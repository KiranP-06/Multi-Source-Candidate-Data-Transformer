"""
Canonical Pydantic v2 schema for the candidate data pipeline.

This module defines three layers of models:
1. CandidateFragment — the intermediate, single-source partial record produced
   by each extractor. Every field is optional.
2. CandidateProfile — the merged canonical record (internal source of truth).
   This is the output of the merge stage and the input to projection.
3. Supporting models — Location, Links, Skill, Experience, Education, Provenance.

Design decisions:
- Validators normalize and sanitize on construction (e.g. lowercase emails,
  validate date formats). Invalid values resolve to None — they never crash.
- CandidateProfile.candidate_id is deterministic (UUID5) — see generate_candidate_id().
- Provenance tracks not just which source won, but HOW the conflict was resolved.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed namespace for deterministic candidate ID generation (UUID5).
# This value never changes across runs — it is part of the pipeline's contract.
CANDIDATE_ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Lightweight email regex — intentionally not full RFC 5322.
# Catches the vast majority of real-world addresses without over-matching.
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# YYYY-MM format validator.
_YYYY_MM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Provenance resolution method labels — kept as a Literal union so typos
# in method names are caught by the type system at model construction time.
ResolutionMethod = Literal[
    "single_source",
    "corroborated",
    "conflict_resolution_majority_vote",
    "conflict_resolution_latest_date",
    "conflict_resolution_higher_source_reliability",
]


# ---------------------------------------------------------------------------
# Deterministic candidate ID
# ---------------------------------------------------------------------------

def generate_candidate_id(
    emails: list[str],
    phones: list[str],
    full_name: str | None,
) -> str:
    """
    Deterministic ID from a merged candidate's stable identifiers.

    Uses the lowest-sorted email or phone as the primary seed, so the result
    is independent of source ordering. Falls back to normalized name if no
    hard identifiers exist.

    Returns a UUID5 string — same inputs always produce the same output.
    """
    # Collect all hard identifiers (already normalized at this point).
    keys = sorted(emails) + sorted(phones)
    if keys:
        seed = keys[0]  # lowest-sorted hard identifier
    elif full_name:
        seed = full_name.strip().lower()
    else:
        # Degenerate case — should be rare after extraction.
        seed = "unknown"
    return str(uuid.uuid5(CANDIDATE_ID_NAMESPACE, seed))


# ---------------------------------------------------------------------------
# Supporting models (used inside CandidateProfile)
# ---------------------------------------------------------------------------

class Location(BaseModel):
    """Geographic location with ISO 3166 alpha-2 country code."""

    city: str | None = None
    region: str | None = None
    country: str | None = None  # ISO 3166 alpha-2 (e.g. "US", "IN")

    @field_validator("country", mode="before")
    @classmethod
    def _validate_country_code(cls, v: Any) -> str | None:
        if v is None:
            return None
        v = str(v).strip().upper()
        # Accept valid 2-letter codes; anything else is left to the
        # normalization stage (which maps names → codes). If it's still
        # not 2 letters after normalization, set to None.
        if len(v) == 2 and v.isalpha():
            return v
        logger.warning("Invalid country code '%s' — setting to None", v)
        return None


class Links(BaseModel):
    """Collection of profile links."""

    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None
    other: list[str] = Field(default_factory=list)


class Skill(BaseModel):
    """
    A single canonical skill with confidence and source traceability.

    `confidence` is computed by the merge stage — not set by extractors.
    `sources` lists which source_ids contributed this skill.
    """

    name: str
    confidence: float = 0.0
    sources: list[str] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    """A single work experience record."""

    company: str | None = None
    title: str | None = None
    start: str | None = None   # YYYY-MM format
    end: str | None = None     # YYYY-MM format, None = current/ongoing
    summary: str | None = None

    @field_validator("start", "end", mode="before")
    @classmethod
    def _validate_date_format(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        v = str(v).strip()
        if _YYYY_MM_RE.match(v):
            return v
        logger.warning("Invalid YYYY-MM date '%s' — setting to None", v)
        return None


class EducationEntry(BaseModel):
    """A single education record."""

    institution: str | None = None
    degree: str | None = None
    field: str | None = None
    end_year: int | None = None

    @field_validator("end_year", mode="before")
    @classmethod
    def _validate_end_year(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        try:
            year = int(v)
            # Sanity-check: reject years that are clearly not graduation years.
            if 1900 <= year <= 2100:
                return year
            logger.warning("end_year %d out of range [1900, 2100] — setting to None", year)
            return None
        except (ValueError, TypeError):
            logger.warning("Invalid end_year '%s' — setting to None", v)
            return None


class Provenance(BaseModel):
    """
    Tracks how a single field value was resolved.

    Every field in the final CandidateProfile should have at least one
    provenance entry explaining where its value came from and how conflicts
    (if any) were resolved.
    """

    field: str          # e.g. "full_name", "experience[0].title"
    value: str | None   # the winning value, serialized to string
    source: str         # the source_id that contributed the winning value
    method: ResolutionMethod


# ---------------------------------------------------------------------------
# Intermediate model — single-source fragment (pre-merge)
# ---------------------------------------------------------------------------

class CandidateFragment(BaseModel):
    """
    A partial, single-source view of a candidate.

    Produced by each extractor. Every field except source_id is optional —
    a fragment may have as little as a name and nothing else. Invalid values
    are silently set to None (never crash).

    Fragments are the INPUT to the blocking/matching/merge stages.
    """

    source_id: str  # e.g. "recruiter_csv", "ats_json", "github", "resume"

    # When the source data was captured/exported — used for staleness penalty.
    # None means "unknown age" which applies zero staleness penalty.
    source_timestamp: datetime | None = None

    full_name: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)

    city: str | None = None
    region: str | None = None
    country: str | None = None

    linkedin_url: str | None = None
    github_url: str | None = None
    portfolio_url: str | None = None
    other_links: list[str] = Field(default_factory=list)

    headline: str | None = None
    years_experience: float | None = None
    skills: list[str] = Field(default_factory=list)

    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)

    @field_validator("emails", mode="before")
    @classmethod
    def _clean_emails(cls, v: Any) -> list[str]:
        """Lowercase, strip, and discard obviously invalid emails."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        cleaned: list[str] = []
        for email in v:
            email = str(email).strip().lower()
            if _EMAIL_RE.match(email):
                cleaned.append(email)
            elif email:  # non-empty but invalid
                logger.warning("Discarding malformed email '%s'", email)
        return cleaned

    @field_validator("phones", mode="before")
    @classmethod
    def _clean_phones(cls, v: Any) -> list[str]:
        """
        Accept raw phone strings — actual E.164 normalization happens
        in the normalizers module, not here. This validator just ensures
        we have a list of non-empty strings.
        """
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(p).strip() for p in v if str(p).strip()]

    @field_validator("full_name", mode="before")
    @classmethod
    def _clean_name(cls, v: Any) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        return v if v else None

    @field_validator("years_experience", mode="before")
    @classmethod
    def _clean_years(cls, v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            val = float(v)
            if val < 0:
                logger.warning("Negative years_experience %s — setting to None", v)
                return None
            return val
        except (ValueError, TypeError):
            logger.warning("Invalid years_experience '%s' — setting to None", v)
            return None


# ---------------------------------------------------------------------------
# Canonical model — the merged, authoritative profile
# ---------------------------------------------------------------------------

class CandidateProfile(BaseModel):
    """
    The merged, canonical candidate profile — internal source of truth.

    This model is READ-ONLY after construction by the merge stage. The
    projection layer reads from it but never mutates it.

    candidate_id is deterministic (UUID5) — see generate_candidate_id().
    """

    candidate_id: str
    full_name: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)  # E.164 format
    location: Location = Field(default_factory=Location)
    links: Links = Field(default_factory=Links)
    headline: str | None = None
    years_experience: float | None = None
    skills: list[Skill] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    overall_confidence: float = 0.0

    @field_validator("overall_confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        """Clamp overall_confidence to [0.0, 1.0]."""
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    @model_validator(mode="after")
    def _sort_experience_by_start_desc(self) -> CandidateProfile:
        """Sort experience entries by start date descending (most recent first)."""
        self.experience = sorted(
            self.experience,
            key=lambda e: e.start or "0000-00",  # None sorts to the bottom
            reverse=True,
        )
        return self

"""
ATS JSON blob extractor.

Maps a semi-structured ATS export into CandidateFragment objects. The ATS schema
uses different field names than our canonical schema — a hardcoded mapping dict
translates them.

Expected top-level structure: {"applicants": [...]}
Each applicant object may have:
  applicant_name, contact_email, contact_phone, current_role, current_employer,
  applied_date, skill_tags, job_history, education_history

Malformed JSON or missing fields are handled gracefully — never crash.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.schema import (
    CandidateFragment,
    EducationEntry,
    ExperienceEntry,
)

logger = logging.getLogger(__name__)


def extract_json(file_path: str) -> List[CandidateFragment]:
    """
    Read an ATS JSON export and return one CandidateFragment per applicant.

    Graceful degradation:
    - Invalid JSON → logs error, returns [].
    - Missing 'applicants' key → logs warning, returns [].
    - Individual applicant parse errors → skips that applicant, continues.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as exc:
        logger.error("Cannot read JSON file '%s': %s", file_path, exc)
        return []

    if not raw.strip():
        logger.warning("JSON file '%s' is empty", file_path)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in '%s': %s", file_path, exc)
        return []

    # Expect a top-level "applicants" or "candidates" key.
    applicants = data.get("applicants") or data.get("candidates")
    if not applicants or not isinstance(applicants, list):
        logger.warning(
            "JSON file '%s' has no 'applicants' or 'candidates' array", file_path
        )
        return []

    fragments: List[CandidateFragment] = []
    for idx, applicant in enumerate(applicants):
        try:
            fragment = _applicant_to_fragment(applicant)
            fragments.append(fragment)
        except Exception as exc:
            logger.warning(
                "Skipping applicant at index %d in '%s': %s", idx, file_path, exc
            )

    logger.info("Extracted %d fragments from JSON '%s'", len(fragments), file_path)
    return fragments


def _applicant_to_fragment(app: Dict[str, Any]) -> CandidateFragment:
    """Convert a single ATS applicant dict into a CandidateFragment."""
    # Parse the applied_date for source_timestamp (staleness tracking).
    source_timestamp = _parse_date(app.get("applied_date"))

    # Map ATS field names → canonical fragment fields.
    full_name = app.get("applicant_name") or None
    email = app.get("contact_email") or None
    phone = app.get("contact_phone") or None
    headline = app.get("current_role") or None

    # Skills
    skills = app.get("skill_tags") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    # Experience from job_history
    experience = _parse_job_history(app.get("job_history"))

    # Education from education_history
    education = _parse_education(app.get("education_history"))

    return CandidateFragment(
        source_id="ats_json",
        source_timestamp=source_timestamp,
        full_name=full_name,
        emails=[email] if email else [],
        phones=[phone] if phone else [],
        headline=headline,
        skills=skills,
        experience=experience,
        education=education,
    )


def _parse_job_history(
    history: Optional[List[Dict[str, Any]]],
) -> List[ExperienceEntry]:
    """Parse ATS job_history array into ExperienceEntry objects."""
    if not history or not isinstance(history, list):
        return []

    entries = []
    for job in history:
        if not isinstance(job, dict):
            continue
        entries.append(
            ExperienceEntry(
                company=job.get("employer") or job.get("company"),
                title=job.get("role") or job.get("title"),
                start=job.get("from") or job.get("start"),
                end=job.get("to") or job.get("end"),
            )
        )
    return entries


def _parse_education(
    history: Optional[List[Dict[str, Any]]],
) -> List[EducationEntry]:
    """Parse ATS education_history array into EducationEntry objects."""
    if not history or not isinstance(history, list):
        return []

    entries = []
    for edu in history:
        if not isinstance(edu, dict):
            continue
        entries.append(
            EducationEntry(
                institution=edu.get("school") or edu.get("institution"),
                degree=edu.get("qualification") or edu.get("degree"),
                field=edu.get("major") or edu.get("field"),
                end_year=edu.get("graduation_year") or edu.get("end_year"),
            )
        )
    return entries


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Best-effort date parsing. Returns None on failure."""
    if not date_str:
        return None
    try:
        from dateutil.parser import parse as dateutil_parse

        return dateutil_parse(date_str)
    except (ValueError, TypeError):
        logger.warning("Could not parse date '%s'", date_str)
        return None

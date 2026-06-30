"""
Normalization module and normalizer registry.

This module provides:
1. Individual normalizer functions for phone numbers, skills, country codes,
   and general text cleanup.
2. A NORMALIZER_REGISTRY (factory pattern): a dict mapping string keys to
   normalizer functions. The projection layer and other modules look up
   normalizers by name from this registry.

Design decision: Adding a new normalizer requires only registering a function
in the registry dict — no core engine changes needed. For example, a future
"uppercase" normalizer can be supported by adding one line:
    NORMALIZER_REGISTRY["uppercase"] = str.upper

Phone normalization uses the `phonenumbers` library for E.164 conversion.
Country normalization uses a hardcoded dict (~30 entries) rather than the
`pycountry` library — simpler, more transparent, and sufficient for the
fixture data. Extending it is a one-line addition.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional

import phonenumbers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phone normalization (E.164)
# ---------------------------------------------------------------------------

def normalize_phone_e164(
    phone: str, default_region: Optional[str] = "US"
) -> Optional[str]:
    """
    Normalize a phone string to E.164 format (e.g. '+14155551234').

    If the phone already has a country code prefix ('+'), it's parsed as-is.
    Otherwise, `default_region` is used as a hint (default: 'US').

    Returns None if the phone cannot be parsed or validated — never guesses.
    A phone like '555-1234' (no area code, no country) returns None.
    """
    if not phone or not phone.strip():
        return None

    phone = phone.strip()

    try:
        # If the number starts with '+', parse without a default region.
        if phone.startswith("+"):
            parsed = phonenumbers.parse(phone, None)
        else:
            parsed = phonenumbers.parse(phone, default_region)

        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
        else:
            logger.warning(
                "Phone '%s' parsed but is not a valid number — returning None",
                phone,
            )
            return None
    except phonenumbers.NumberParseException as exc:
        logger.warning("Cannot parse phone '%s': %s — returning None", phone, exc)
        return None


def normalize_phones_list(phones: List[str]) -> List[str]:
    """
    Normalize a list of phone strings to E.164, dropping any that can't be
    parsed. Deduplicates the result.
    """
    seen = set()
    result = []
    for phone in phones:
        normalized = normalize_phone_e164(phone)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# Skill normalization (canonical names)
# ---------------------------------------------------------------------------

# Synonym table: maps common aliases/abbreviations to canonical skill names.
# All keys must be lowercase. Extend by adding entries — no code changes needed.
_SKILL_SYNONYMS: Dict[str, str] = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "py": "Python",
    "python": "Python",
    "golang": "Go",
    "go": "Go",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "sql": "SQL",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "redis": "Redis",
    "shell": "Shell",
    "bash": "Shell",
    "github actions": "GitHub Actions",
    "html": "HTML",
    "css": "CSS",
}


def normalize_skill_canonical(skill: str) -> str:
    """
    Normalize a skill name to its canonical form via the synonym table.

    If no synonym mapping exists, returns the skill as-is with title casing.
    """
    if not skill:
        return skill
    key = skill.strip().lower()
    return _SKILL_SYNONYMS.get(key, skill.strip())


def normalize_skills_list(skills: List[str]) -> List[str]:
    """
    Normalize a list of skill names to canonical forms.
    Deduplicates (by canonical name) and returns sorted for determinism.
    """
    seen = set()
    result = []
    for skill in skills:
        canonical = normalize_skill_canonical(skill)
        key = canonical.lower()
        if key not in seen:
            seen.add(key)
            result.append(canonical)
    return sorted(result)


# ---------------------------------------------------------------------------
# Country normalization (name/alias → ISO 3166 alpha-2)
# ---------------------------------------------------------------------------

# Hardcoded mapping of common country names/aliases to ISO 3166 alpha-2 codes.
# ~30 entries — sufficient for realistic fixture data. Extend as needed.
_COUNTRY_MAP: Dict[str, str] = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "india": "IN",
    "canada": "CA",
    "united kingdom": "GB",
    "uk": "GB",
    "germany": "DE",
    "france": "FR",
    "australia": "AU",
    "japan": "JP",
    "china": "CN",
    "brazil": "BR",
    "mexico": "MX",
    "singapore": "SG",
    "ireland": "IE",
    "netherlands": "NL",
    "sweden": "SE",
    "switzerland": "CH",
    "israel": "IL",
    "south korea": "KR",
    "spain": "ES",
    "italy": "IT",
    "new zealand": "NZ",
    "poland": "PL",
    "portugal": "PT",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "austria": "AT",
    "belgium": "BE",
}


def normalize_country(country: Optional[str]) -> Optional[str]:
    """
    Normalize a country name or alias to ISO 3166 alpha-2 code.

    Accepts full names ('United States'), common aliases ('USA'), or
    already-valid 2-letter codes ('US'). Returns None for unrecognized input.
    """
    if not country:
        return None
    key = country.strip().lower()
    # If it's already a valid 2-letter code, uppercase and return.
    if len(key) == 2 and key.isalpha():
        return key.upper()
    return _COUNTRY_MAP.get(key)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_name(name: Optional[str]) -> Optional[str]:
    """
    Normalize a full name: Unicode NFC normalization, collapse whitespace,
    title-case.
    """
    if not name:
        return None
    # Unicode NFC normalization — ensures consistent representation.
    name = unicodedata.normalize("NFC", name)
    # Collapse multiple whitespace to single spaces, strip edges.
    name = re.sub(r"\s+", " ", name).strip()
    # Title-case.
    return name.title() if name else None


# ---------------------------------------------------------------------------
# Email normalization
# ---------------------------------------------------------------------------

def normalize_email(email: str) -> Optional[str]:
    """Lowercase, strip whitespace. Already done by schema validator, but
    available here for the normalizer registry."""
    if not email:
        return None
    return email.strip().lower()


# ---------------------------------------------------------------------------
# General-purpose normalizers
# ---------------------------------------------------------------------------

def normalize_lowercase(value: Any) -> str:
    """Convert to lowercase string."""
    return str(value).lower()


def normalize_uppercase(value: Any) -> str:
    """Convert to uppercase string."""
    return str(value).upper()


# ---------------------------------------------------------------------------
# NORMALIZER REGISTRY (factory pattern)
# ---------------------------------------------------------------------------
#
# Maps string keys (used in projection configs) to normalizer functions.
# Adding a new normalizer = adding one entry here. No core engine changes.

NORMALIZER_REGISTRY: Dict[str, Callable] = {
    "E164": normalize_phone_e164,
    "canonical": normalize_skill_canonical,
    "lowercase": normalize_lowercase,
    "uppercase": normalize_uppercase,
    "country": normalize_country,
    "name": normalize_name,
    "email": normalize_email,
}


# ---------------------------------------------------------------------------
# Fragment-level normalization (applied before blocking/matching)
# ---------------------------------------------------------------------------

def normalize_fragment(fragment: "CandidateFragment") -> "CandidateFragment":
    """
    Normalize all fields of a CandidateFragment in preparation for
    blocking, matching, and merge.

    Returns a new CandidateFragment with normalized values. The original
    fragment is not mutated.

    Applied normalizations:
    - full_name: Unicode NFC + whitespace collapse + title-case
    - emails: already lowercased by schema validator (no-op here)
    - phones: E.164 via phonenumbers library; unparseable → dropped
    - skills: canonical names via synonym table, deduplicated
    - country: name/alias → ISO 3166 alpha-2
    """
    # Import here to avoid circular dependency (schema imports nothing from normalizers).
    from pipeline.schema import CandidateFragment as CF

    return CF(
        source_id=fragment.source_id,
        source_timestamp=fragment.source_timestamp,
        full_name=normalize_name(fragment.full_name),
        emails=list(fragment.emails),  # already lowercased by schema validator
        phones=normalize_phones_list(fragment.phones),
        city=fragment.city,
        region=fragment.region,
        country=normalize_country(fragment.country),
        linkedin_url=fragment.linkedin_url,
        github_url=fragment.github_url,
        portfolio_url=fragment.portfolio_url,
        other_links=list(fragment.other_links),
        headline=fragment.headline,
        years_experience=fragment.years_experience,
        skills=normalize_skills_list(fragment.skills),
        experience=list(fragment.experience),
        education=list(fragment.education),
    )


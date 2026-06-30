"""
Recruiter CSV extractor.

Maps a recruiter-exported CSV into CandidateFragment objects. Expected columns:
  name, email, phone, current_company, title

Missing or extra columns are handled gracefully — the extractor logs a warning
and extracts whatever fields are present. Completely empty files return an
empty list. Malformed rows are skipped individually.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import List

from pipeline.schema import CandidateFragment, ExperienceEntry

logger = logging.getLogger(__name__)

# Column name → fragment field mapping.
# Only columns listed here are extracted; extra columns are silently ignored.
_COLUMN_MAP = {
    "name": "full_name",
    "email": "emails",
    "phone": "phones",
    "current_company": "_company",   # handled specially → experience[0].company
    "title": "_title",               # handled specially → experience[0].title
}


def extract_csv(file_path: str) -> List[CandidateFragment]:
    """
    Read a recruiter CSV and return one CandidateFragment per row.

    Graceful degradation:
    - Missing columns → those fields are left as None/empty on each fragment.
    - Empty file → returns [].
    - Unreadable file → logs error, returns [].
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
    except OSError as exc:
        logger.error("Cannot read CSV file '%s': %s", file_path, exc)
        return []

    if not content:
        logger.warning("CSV file '%s' is empty", file_path)
        return []

    reader = csv.DictReader(io.StringIO(content))

    # Check which expected columns are actually present.
    if reader.fieldnames:
        present = set(reader.fieldnames)
        expected = set(_COLUMN_MAP.keys())
        missing = expected - present
        if missing:
            logger.warning(
                "CSV '%s' is missing columns %s — those fields will be null",
                file_path,
                sorted(missing),
            )

    fragments: List[CandidateFragment] = []
    for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
        try:
            fragment = _row_to_fragment(row)
            fragments.append(fragment)
        except Exception as exc:
            # Catch-all: never let a single bad row crash the whole file.
            logger.warning("Skipping CSV row %d in '%s': %s", row_num, file_path, exc)

    logger.info("Extracted %d fragments from CSV '%s'", len(fragments), file_path)
    return fragments


def _row_to_fragment(row: dict) -> CandidateFragment:
    """Convert a single CSV row dict into a CandidateFragment."""
    full_name = row.get("name") or None
    email = row.get("email") or None
    phone = row.get("phone") or None
    company = row.get("current_company") or None
    title = row.get("title") or None

    # Build experience entry from company + title if either is present.
    experience = []
    if company or title:
        experience.append(ExperienceEntry(company=company, title=title))

    return CandidateFragment(
        source_id="recruiter_csv",
        full_name=full_name,
        emails=[email] if email else [],
        phones=[phone] if phone else [],
        experience=experience,
    )

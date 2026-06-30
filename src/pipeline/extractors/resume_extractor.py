"""
Plain-text resume extractor.

Parses a plain-text (.txt) resume into a CandidateFragment using section-based
heuristics and regex patterns. This is intentionally simple — the architecture
allows swapping in an NLP/LLM-based parser later without changing anything
downstream.

Expected sections (case-insensitive):
  CONTACT / header area: name (first line), email, phone
  SUMMARY / OBJECTIVE: → headline
  EXPERIENCE: → experience entries
  EDUCATION: → education entries
  SKILLS: → skill list

Design decision: We use regex and line-by-line parsing rather than ML to keep
dependencies minimal. This works well for structured resumes but will miss
creative/unusual formats — documented as a known limitation.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from pipeline.schema import (
    CandidateFragment,
    EducationEntry,
    ExperienceEntry,
)

logger = logging.getLogger(__name__)

# --- Regex patterns ---

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)

# Section headings — matches lines that are entirely a heading keyword,
# possibly with trailing colon or whitespace.
_SECTION_RE = re.compile(
    r"^\s*(SUMMARY|OBJECTIVE|EXPERIENCE|WORK\s+EXPERIENCE|EDUCATION|SKILLS|"
    r"TECHNICAL\s+SKILLS|CONTACT|PROJECTS)\s*:?\s*$",
    re.IGNORECASE,
)

# Experience entry pattern: "Title, Company" or "Title at Company" on one line,
# followed by a date range line.
_EXPERIENCE_TITLE_RE = re.compile(
    r"^(.+?)\s*(?:,\s*|\s+at\s+)(.+)$", re.IGNORECASE
)
_DATE_RANGE_RE = re.compile(
    r"((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
    r"Nov|Dec)\s+\d{4})\s*(?:-|–|to)\s*(Present|(?:January|February|March|"
    r"April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
    re.IGNORECASE,
)

# Education pattern: "Degree Field, Institution, Year" or similar.
_EDUCATION_RE = re.compile(
    r"(BS|BA|MS|MA|MBA|PhD|Ph\.D|B\.S\.|B\.A\.|M\.S\.|M\.A\.)\s+(.+?),\s*(.+?),\s*(\d{4})",
    re.IGNORECASE,
)

# Month name → number mapping for date conversion.
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
}


def extract_resume(file_path: str) -> List[CandidateFragment]:
    """
    Parse a plain-text resume file into a CandidateFragment.

    Returns a list (always 0 or 1 items) for consistency with other extractors.

    Graceful degradation:
    - Unreadable file → logs error, returns [].
    - Missing sections → those fields are None/empty.
    - Unparseable entries → skipped with warning.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as exc:
        logger.error("Cannot read resume file '%s': %s", file_path, exc)
        return []

    if not content.strip():
        logger.warning("Resume file '%s' is empty", file_path)
        return []

    sections = _split_sections(content)

    # Extract contact info from the header (lines before the first section).
    header = sections.get("_header", "")
    full_name = _extract_name(header)
    emails = _EMAIL_RE.findall(content)  # search entire doc for emails
    phones = _PHONE_RE.findall(content)  # search entire doc for phones

    # Headline from SUMMARY or OBJECTIVE section.
    headline = _extract_headline(sections)

    # Experience entries.
    experience = _extract_experience(sections)

    # Education entries.
    education = _extract_education(sections)

    # Skills from SKILLS section.
    skills = _extract_skills(sections)

    fragment = CandidateFragment(
        source_id="resume",
        full_name=full_name,
        emails=emails,
        phones=phones,
        headline=headline,
        skills=skills,
        experience=experience,
        education=education,
    )

    logger.info("Extracted 1 fragment from resume '%s'", file_path)
    return [fragment]


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _split_sections(content: str) -> dict:
    """
    Split resume text into named sections based on heading lines.

    Returns a dict of {section_name: section_text}. Lines before the first
    recognized heading are stored under the key '_header'.
    """
    sections = {}
    current_section = "_header"
    current_lines: List[str] = []

    for line in content.split("\n"):
        match = _SECTION_RE.match(line)
        if match:
            # Save the previous section.
            sections[current_section] = "\n".join(current_lines).strip()
            current_section = match.group(1).strip().upper()
            # Normalize "WORK EXPERIENCE" → "EXPERIENCE"
            if "EXPERIENCE" in current_section:
                current_section = "EXPERIENCE"
            if "SKILLS" in current_section:
                current_section = "SKILLS"
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section.
    sections[current_section] = "\n".join(current_lines).strip()
    return sections


def _extract_name(header: str) -> Optional[str]:
    """
    Extract the candidate's name from the header area.

    Heuristic: the first non-empty line that doesn't look like an email
    or phone number is assumed to be the name.
    """
    for line in header.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip lines that are clearly email or phone.
        if _EMAIL_RE.search(line) and len(line) < 60:
            # If the line is ONLY an email, skip it. If it contains more
            # text (like "Name — email"), let it through.
            if _EMAIL_RE.fullmatch(line):
                continue
        if _PHONE_RE.fullmatch(line):
            continue
        # This is likely the name.
        return line
    return None


def _extract_headline(sections: dict) -> Optional[str]:
    """Extract a one-line headline from the SUMMARY or OBJECTIVE section."""
    for key in ("SUMMARY", "OBJECTIVE"):
        text = sections.get(key, "").strip()
        if text:
            # Take the first sentence or first 200 chars as the headline.
            # Join multi-line summaries into one string.
            joined = " ".join(text.split())
            return joined[:200]
    return None


def _extract_experience(sections: dict) -> List[ExperienceEntry]:
    """
    Parse the EXPERIENCE section into ExperienceEntry objects.

    Expects patterns like:
      Title, Company
      Month Year - Month Year / Present
      Description lines...
    """
    text = sections.get("EXPERIENCE", "")
    if not text:
        return []

    entries: List[ExperienceEntry] = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Try to match "Title, Company" or "Title at Company".
        title_match = _EXPERIENCE_TITLE_RE.match(line)
        if title_match:
            title = title_match.group(1).strip()
            company = title_match.group(2).strip()

            # Look for a date range on the next line.
            start, end = None, None
            if i + 1 < len(lines):
                date_match = _DATE_RANGE_RE.search(lines[i + 1])
                if date_match:
                    start = _month_year_to_yyyy_mm(date_match.group(1))
                    end_str = date_match.group(2).strip()
                    end = None if end_str.lower() == "present" else _month_year_to_yyyy_mm(end_str)
                    i += 1  # skip the date line

            # Collect subsequent description lines as summary.
            summary_lines = []
            i += 1
            while i < len(lines):
                desc_line = lines[i].strip()
                if not desc_line:
                    i += 1
                    break
                # Stop if we hit what looks like another title line.
                if _EXPERIENCE_TITLE_RE.match(desc_line):
                    break
                summary_lines.append(desc_line)
                i += 1

            entries.append(
                ExperienceEntry(
                    company=company,
                    title=title,
                    start=start,
                    end=end,
                    summary=" ".join(summary_lines) if summary_lines else None,
                )
            )
        else:
            i += 1

    return entries


def _extract_education(sections: dict) -> List[EducationEntry]:
    """Parse the EDUCATION section into EducationEntry objects."""
    text = sections.get("EDUCATION", "")
    if not text:
        return []

    entries: List[EducationEntry] = []
    for line in text.split("\n"):
        match = _EDUCATION_RE.search(line)
        if match:
            entries.append(
                EducationEntry(
                    degree=match.group(1).strip(),
                    field=match.group(2).strip(),
                    institution=match.group(3).strip(),
                    end_year=int(match.group(4)),
                )
            )

    return entries


def _extract_skills(sections: dict) -> List[str]:
    """
    Parse the SKILLS section into a list of skill name strings.

    Handles comma-separated, pipe-separated, and newline-separated formats.
    """
    text = sections.get("SKILLS", "")
    if not text:
        return []

    # Split on commas, pipes, newlines, and semicolons.
    raw = re.split(r"[,|\n;]+", text)
    skills = [s.strip() for s in raw if s.strip()]
    return skills


def _month_year_to_yyyy_mm(text: str) -> Optional[str]:
    """
    Convert 'March 2022' or 'Mar 2022' to '2022-03'.

    Returns None if the format doesn't match.
    """
    text = text.strip()
    parts = text.split()
    if len(parts) != 2:
        return None

    month_str = parts[0].lower().rstrip(".")
    year_str = parts[1]

    month_num = _MONTH_MAP.get(month_str)
    if not month_num:
        return None

    try:
        year = int(year_str)
    except ValueError:
        return None

    return f"{year:04d}-{month_num}"

"""
GitHub profile extractor.

Maps a GitHub user profile (public REST API response) into a CandidateFragment.
Supports two modes:
  1. Offline: reads a pre-fetched JSON file (for tests and offline use).
  2. Live:   fetches from https://api.github.com/users/{username} (optional).

Field mapping:
  name       → full_name
  bio        → headline
  email      → emails
  html_url   → github_url
  location   → city (best-effort split)
  repos[].language → skills (deduplicated)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pipeline.schema import CandidateFragment

logger = logging.getLogger(__name__)


def extract_github_from_file(file_path: str) -> List[CandidateFragment]:
    """
    Read a saved GitHub API JSON response and return a CandidateFragment.

    The JSON should be a single user object (not an array), optionally
    including a 'repos' key with the user's repositories.

    Returns a list (always 0 or 1 items) for consistency with other extractors.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as exc:
        logger.error("Cannot read GitHub JSON file '%s': %s", file_path, exc)
        return []

    if not raw.strip():
        logger.warning("GitHub JSON file '%s' is empty", file_path)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in GitHub file '%s': %s", file_path, exc)
        return []

    if not isinstance(data, dict):
        logger.warning("GitHub JSON '%s' is not an object", file_path)
        return []

    fragment = _profile_to_fragment(data)
    logger.info("Extracted 1 fragment from GitHub file '%s'", file_path)
    return [fragment]


def extract_github_live(username: str) -> List[CandidateFragment]:
    """
    Fetch a GitHub user profile via the public REST API and return a fragment.

    Requires the `requests` library. Rate-limited to 60 req/hr without auth.
    Returns [] on any network or API error.
    """
    try:
        import requests
    except ImportError:
        logger.error("'requests' library is required for live GitHub fetching")
        return []

    user_url = f"https://api.github.com/users/{username}"
    repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"

    try:
        user_resp = requests.get(user_url, timeout=10)
        user_resp.raise_for_status()
        user_data = user_resp.json()

        repos_resp = requests.get(repos_url, timeout=10)
        repos_resp.raise_for_status()
        user_data["repos"] = repos_resp.json()

    except Exception as exc:
        logger.error("GitHub API request failed for '%s': %s", username, exc)
        return []

    fragment = _profile_to_fragment(user_data)
    logger.info("Extracted 1 fragment from GitHub API for user '%s'", username)
    return [fragment]


def _profile_to_fragment(data: Dict[str, Any]) -> CandidateFragment:
    """Convert a GitHub API user object into a CandidateFragment."""
    full_name = data.get("name") or None
    email = data.get("email") or None
    bio = data.get("bio") or None
    github_url = data.get("html_url") or None
    location_raw = data.get("location") or None

    # Best-effort location parsing: "San Francisco, CA" → city, region.
    city: Optional[str] = None
    region: Optional[str] = None
    if location_raw:
        parts = [p.strip() for p in location_raw.split(",")]
        city = parts[0] if len(parts) >= 1 else None
        region = parts[1] if len(parts) >= 2 else None

    # Extract unique programming languages from repos as skills.
    skills = _extract_repo_languages(data.get("repos"))

    return CandidateFragment(
        source_id="github",
        full_name=full_name,
        emails=[email] if email else [],
        github_url=github_url,
        city=city,
        region=region,
        headline=bio,
        skills=skills,
    )


def _extract_repo_languages(repos: Any) -> List[str]:
    """
    Extract unique programming languages from a list of GitHub repo objects.

    Skips repos with null/missing language. Returns deduplicated list
    in a deterministic (sorted) order.
    """
    if not repos or not isinstance(repos, list):
        return []

    languages = set()
    for repo in repos:
        if isinstance(repo, dict):
            lang = repo.get("language")
            if lang and isinstance(lang, str):
                languages.add(lang)

    return sorted(languages)  # sorted for determinism

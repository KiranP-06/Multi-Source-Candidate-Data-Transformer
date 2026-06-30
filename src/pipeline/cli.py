"""
CLI entrypoint for the candidate data transformer pipeline.

Usage:
  python -m pipeline.cli \\
    --inputs fixtures/recruiter.csv fixtures/ats_export.json \\
    --config config/default_output.json \\
    --output results.json

  # With live GitHub fetch:
  python -m pipeline.cli \\
    --inputs fixtures/recruiter.csv \\
    --github-user octocat \\
    --config config/default_output.json

Pipeline stages:
  detect → extract → normalize → block_and_match → merge_with_confidence
  → project_to_config → validate → output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

from pipeline.blocking import get_candidate_groups
from pipeline.extractors.csv_extractor import extract_csv
from pipeline.extractors.github_extractor import (
    extract_github_from_file,
    extract_github_live,
)
from pipeline.extractors.json_extractor import extract_json
from pipeline.extractors.resume_extractor import extract_resume
from pipeline.matching import refine_groups
from pipeline.merge import merge_all_groups
from pipeline.normalizers import normalize_fragment
from pipeline.projection import (
    get_required_field_names,
    load_config,
    project_all,
)
from pipeline.schema import CandidateFragment

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def detect_source_type(file_path: str) -> str | None:
    """
    Detect the source type of an input file by extension + content sniffing.

    Returns: "csv", "ats_json", "github_json", "resume", or None.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return "csv"

    if ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read(4096)  # Read first 4KB for sniffing.
            data = json.loads(raw)
            if isinstance(data, dict):
                if "login" in data or "public_repos" in data:
                    return "github_json"
                if "applicants" in data or "candidates" in data:
                    return "ats_json"
        except (json.JSONDecodeError, OSError):
            pass
        return "ats_json"  # Default JSON → ATS.

    if ext == ".txt":
        return "resume"

    return None


def extract_file(file_path: str) -> List[CandidateFragment]:
    """
    Detect source type and extract fragments from a file.

    Returns an empty list (never crashes) if the file can't be processed.
    """
    source_type = detect_source_type(file_path)

    if source_type is None:
        logger.warning("Unknown file type for '%s' — skipping", file_path)
        return []

    logger.info("Detected '%s' as source type '%s'", file_path, source_type)

    if source_type == "csv":
        return extract_csv(file_path)
    elif source_type == "ats_json":
        return extract_json(file_path)
    elif source_type == "github_json":
        return extract_github_from_file(file_path)
    elif source_type == "resume":
        return extract_resume(file_path)
    else:
        return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_paths: List[str],
    config_path: str,
    output_path: str | None = None,
    canonical_output_path: str | None = None,
    github_user: str | None = None,
) -> None:
    """Run the full pipeline: detect → extract → normalize → block → match → merge → project → output."""

    # --- Load config ---
    config = load_config(config_path)
    required_fields = get_required_field_names(config)
    logger.info("Loaded config from '%s' (%d fields)", config_path, len(config.fields))

    # --- Extract ---
    raw_fragments: List[CandidateFragment] = []

    for path in input_paths:
        try:
            frags = extract_file(path)
            raw_fragments.extend(frags)
            logger.info("Extracted %d fragments from '%s'", len(frags), path)
        except Exception as exc:
            logger.error("Failed to extract from '%s': %s — skipping", path, exc)

    # Live GitHub fetch (optional).
    if github_user:
        try:
            frags = extract_github_live(github_user)
            raw_fragments.extend(frags)
            logger.info("Extracted %d fragments from GitHub user '%s'", len(frags), github_user)
        except Exception as exc:
            logger.error("Failed to fetch GitHub user '%s': %s — skipping", github_user, exc)

    if not raw_fragments:
        logger.error("No fragments extracted from any source — aborting")
        sys.exit(1)

    logger.info("Total raw fragments: %d", len(raw_fragments))

    # --- Normalize ---
    fragments = [normalize_fragment(f) for f in raw_fragments]
    logger.info("Normalized %d fragments", len(fragments))

    # --- Block and match ---
    groups = get_candidate_groups(fragments)
    refined = refine_groups(fragments, groups)
    logger.info("Identity resolution: %d fragments → %d candidate groups", len(fragments), len(refined))

    # --- Merge ---
    profiles = merge_all_groups(fragments, refined, required_fields=required_fields)
    logger.info("Merged into %d candidate profiles", len(profiles))

    # --- Write canonical output (optional) ---
    if canonical_output_path:
        canonical_data = [p.model_dump(mode="json") for p in profiles]
        _write_json(canonical_data, canonical_output_path)
        logger.info("Wrote canonical output to '%s'", canonical_output_path)

    # --- Project ---
    projected = project_all(profiles, config)
    logger.info("Projected %d profiles", len(projected))

    # --- Output ---
    _write_json(projected, output_path)
    if output_path:
        logger.info("Wrote projected output to '%s'", output_path)


def _write_json(data: object, path: str | None) -> None:
    """Write JSON to a file or stdout."""
    output = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")
    else:
        print(output)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Multi-Source Candidate Data Transformer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.cli \\
    --inputs fixtures/recruiter.csv fixtures/ats_export.json \\
    --config config/default_output.json

  python -m pipeline.cli \\
    --inputs fixtures/recruiter.csv fixtures/ats_export.json \\
    fixtures/github_mock.json fixtures/resume_sample.txt \\
    --config config/default_output.json \\
    --output results.json \\
    --canonical-output canonical.json
        """,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input file paths (CSV, JSON, TXT). Source type is auto-detected.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the projection config JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for projected results. Defaults to stdout.",
    )
    parser.add_argument(
        "--canonical-output",
        default=None,
        help="Optional: write full canonical profiles (pre-projection) to this file.",
    )
    parser.add_argument(
        "--github-user",
        default=None,
        help="Optional: fetch live GitHub profile for this username.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    # Configure logging.
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    run_pipeline(
        input_paths=args.inputs,
        config_path=args.config,
        output_path=args.output,
        canonical_output_path=args.canonical_output,
        github_user=args.github_user,
    )


if __name__ == "__main__":
    main()

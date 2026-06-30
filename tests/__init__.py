"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import os
from typing import List

import pytest

from pipeline.schema import CandidateFragment, ExperienceEntry


# Convenience: path to the fixtures directory.
FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
)


def fixture_path(filename: str) -> str:
    """Return the absolute path to a fixture file."""
    return os.path.join(FIXTURES_DIR, filename)

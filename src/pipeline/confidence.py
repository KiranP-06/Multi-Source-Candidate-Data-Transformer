"""
Confidence scoring constants and helpers.

Implements the explicit, inspectable confidence formula:

  confidence(field) = base(source_reliability)
                    + corroboration_bonus(n_sources_agreeing)
                    - staleness_penalty(age_of_value)
                    - conflict_penalty(n_sources_disagreeing)

All weights are named constants with rationale comments, so they are easy to
find, justify, and adjust.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Source reliability rankings
# ---------------------------------------------------------------------------
# These represent our baseline trust in each source type. Values are in [0, 1].

SOURCE_RELIABILITY = {
    "github": 0.80,        # Self-reported but verified by actual public activity
    "ats_json": 0.75,      # Structured system-of-record, semi-automated
    "recruiter_csv": 0.70, # Structured but manually entered, potentially stale
    "resume": 0.65,        # Self-reported, potentially embellished, heuristic parsing
}

# Fallback for unknown source types.
DEFAULT_SOURCE_RELIABILITY = 0.50


def get_source_reliability(source_id: str) -> float:
    """Look up the reliability score for a source_id."""
    return SOURCE_RELIABILITY.get(source_id, DEFAULT_SOURCE_RELIABILITY)


# ---------------------------------------------------------------------------
# Confidence formula constants
# ---------------------------------------------------------------------------

# Each additional source agreeing on the same value adds this bonus.
# Capped at CORROBORATION_MAX_BONUS to prevent runaway scores.
CORROBORATION_BONUS_PER_SOURCE: float = 0.10
CORROBORATION_MAX_BONUS: float = 0.20

# Data older than 1 year loses slight confidence. Capped to avoid
# treating old data as worthless.
STALENESS_PENALTY_PER_YEAR: float = 0.02
STALENESS_MAX_PENALTY: float = 0.10

# Each source that disagrees with the chosen value significantly lowers
# confidence. This is intentionally heavier than corroboration bonus because
# "a wrong-but-confident value is worse than an honestly empty one."
CONFLICT_PENALTY_PER_SOURCE: float = 0.15
CONFLICT_MAX_PENALTY: float = 0.30


# ---------------------------------------------------------------------------
# Confidence calculation functions
# ---------------------------------------------------------------------------

def compute_field_confidence(
    source_id: str,
    n_agreeing: int,
    n_disagreeing: int,
    source_timestamp: Optional[datetime] = None,
) -> float:
    """
    Compute the confidence score for a single field value.

    Args:
        source_id: The source that provided the winning value.
        n_agreeing: Number of OTHER sources that agree with this value.
        n_disagreeing: Number of sources that provided a different value.
        source_timestamp: When the value was captured (for staleness).

    Returns:
        Confidence score clamped to [0.0, 1.0].
    """
    base = get_source_reliability(source_id)

    # Corroboration bonus: each agreeing source adds a bonus (capped).
    corroboration = min(
        n_agreeing * CORROBORATION_BONUS_PER_SOURCE,
        CORROBORATION_MAX_BONUS,
    )

    # Staleness penalty: based on age of the data.
    staleness = 0.0
    if source_timestamp:
        now = datetime.now(timezone.utc)
        # Ensure source_timestamp is timezone-aware for comparison.
        if source_timestamp.tzinfo is None:
            source_timestamp = source_timestamp.replace(tzinfo=timezone.utc)
        age_years = (now - source_timestamp).days / 365.25
        if age_years > 1.0:
            staleness = min(
                (age_years - 1.0) * STALENESS_PENALTY_PER_YEAR,
                STALENESS_MAX_PENALTY,
            )

    # Conflict penalty: each disagreeing source lowers confidence.
    conflict = min(
        n_disagreeing * CONFLICT_PENALTY_PER_SOURCE,
        CONFLICT_MAX_PENALTY,
    )

    confidence = base + corroboration - staleness - conflict
    return max(0.0, min(1.0, confidence))


def compute_overall_confidence(
    field_confidences: dict[str, float],
    required_fields: set[str],
) -> float:
    """
    Compute the overall candidate confidence as a weighted average of
    per-field confidences.

    Fields marked as required in the active output config get weight 2.0;
    optional fields get weight 1.0. This ensures that high-confidence
    required fields contribute more to the overall score.

    Returns a value clamped to [0.0, 1.0].
    """
    if not field_confidences:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for field_name, confidence in field_confidences.items():
        weight = 2.0 if field_name in required_fields else 1.0
        weighted_sum += confidence * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    overall = weighted_sum / total_weight
    return max(0.0, min(1.0, overall))

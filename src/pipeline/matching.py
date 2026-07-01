"""
Fuzzy matching module for identity resolution.

After blocking groups fragments by shared identifiers, this module scores
pairwise similarity within each group and decides whether fragments represent
the same person (merge) or different people (keep separate).

Similarity formula:
  similarity(A, B) = w_name  × jaro_winkler(A.full_name, B.full_name)
                   + w_email × exact_email_overlap(A.emails, B.emails)
                   + w_phone × exact_phone_overlap(A.phones, B.phones)
                   + w_company × fuzzy_company_match(A.companies, B.companies)
                   + w_skills × jaccard_skills(A.skills, B.skills)

Weights and threshold are defined as named constants with rationale comments.
Uses `rapidfuzz` for string similarity (Jaro-Winkler).
"""

from __future__ import annotations

import logging
from typing import List, Set

from rapidfuzz import fuzz

from pipeline.schema import CandidateFragment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Similarity weights — named constants with rationale
# ---------------------------------------------------------------------------

# Names are strong but not unique — common names like "John Smith" exist.
W_NAME: float = 0.30

# Email is a strong identifier but can change over time (job changes).
W_EMAIL: float = 0.30

# Phone is strong but less commonly available in all sources.
W_PHONE: float = 0.20

# Company is a corroborating signal — not a primary identifier.
W_COMPANY: float = 0.10

# Skill overlap is weakly corroborating — many people share skill sets.
W_SKILLS: float = 0.10

# ---------------------------------------------------------------------------
# Merge threshold
# ---------------------------------------------------------------------------
# At 0.60, a pair needs a name match (contributing up to 0.30) PLUS at least
# one shared email (0.30) or phone (0.20) to meet the threshold. A perfect
# name match with a shared email scores exactly 0.60, which is the minimum
# needed — this correctly merges "Jon Doe" (CSV) with "Jon Doe" (GitHub) when
# they share an email. Name alone (max 0.30) is well under 0.60, preventing
# false merges of people with similar names at different companies.
MERGE_THRESHOLD: float = 0.60

# ---------------------------------------------------------------------------
# Orphan matching — higher bar for records without shared hard identifiers
# ---------------------------------------------------------------------------
# Orphans (no email/phone) are only compared against each other. They require
# a very high name similarity AND overlapping company to merge. This prevents
# false merges of "John Smith at Google" and "John Smith at Meta".
ORPHAN_NAME_THRESHOLD: float = 0.92
ORPHAN_COMPANY_THRESHOLD: float = 0.80


MATCH_DIAGNOSTICS: List[Dict[str, Any]] = []

def compute_similarity(a: CandidateFragment, b: CandidateFragment) -> float:
    """
    Compute the weighted similarity score between two fragments.

    Returns a float in [0.0, 1.0].
    """
    name_score = _name_similarity(a.full_name, b.full_name)
    email_score = _set_overlap_score(set(a.emails), set(b.emails))
    phone_score = _set_overlap_score(set(a.phones), set(b.phones))
    company_score = _company_similarity(a, b)
    skills_score = _jaccard_similarity(set(a.skills), set(b.skills))

    total = (
        W_NAME * name_score
        + W_EMAIL * email_score
        + W_PHONE * phone_score
        + W_COMPANY * company_score
        + W_SKILLS * skills_score
    )
    return min(1.0, max(0.0, total))


def should_merge(a: CandidateFragment, b: CandidateFragment) -> bool:
    """
    Determine whether two fragments represent the same person.

    Returns True if the similarity score meets or exceeds MERGE_THRESHOLD.
    """
    score = compute_similarity(a, b)
    result = score >= MERGE_THRESHOLD

    # Capture diagnostic for UI
    MATCH_DIAGNOSTICS.append({
        "source_a": a.source_id,
        "source_b": b.source_id,
        "name_a": a.full_name,
        "name_b": b.full_name,
        "name_similarity": _name_similarity(a.full_name, b.full_name),
        "company_similarity": _company_similarity(a, b),
        "skill_overlap": _jaccard_similarity(set(a.skills), set(b.skills)),
        "final_score": score,
        "threshold": MERGE_THRESHOLD,
        "decision": "merged" if result else "kept separate",
    })

    logger.debug(
        "Similarity('%s', '%s') = %.3f → %s",
        a.full_name,
        b.full_name,
        score,
        "MERGE" if result else "KEEP_SEPARATE",
    )
    return result


def should_merge_orphans(a: CandidateFragment, b: CandidateFragment) -> bool:
    """
    Stricter merge check for orphan fragments (no shared email/phone).

    Requires BOTH high name similarity AND high company similarity.
    This prevents false merges of common names at different companies.
    """
    name_score = _name_similarity(a.full_name, b.full_name)
    if name_score < ORPHAN_NAME_THRESHOLD:
        return False

    company_score = _company_similarity(a, b)
    if company_score < ORPHAN_COMPANY_THRESHOLD:
        return False

    logger.debug(
        "Orphan merge('%s', '%s'): name=%.3f, company=%.3f → MERGE",
        a.full_name,
        b.full_name,
        name_score,
        company_score,
    )
    return True


def refine_groups(
    fragments: List[CandidateFragment],
    groups: List[List[int]],
) -> List[List[int]]:
    """
    Refine candidate groups by applying pairwise similarity within each group.

    For groups with >1 fragment (from blocking), verify that all pairs meet
    the merge threshold. Split any that don't.

    For singleton orphan groups, attempt to merge orphans that pass the
    stricter orphan threshold.

    Returns the final list of fragment-index groups, each representing one
    candidate.
    """
    refined: List[List[int]] = []
    orphan_singletons: List[int] = []

    for group in groups:
        if len(group) == 1:
            # Check if this is truly an orphan (no blocking keys matched
            # anyone else) or just a unique person with identifiers.
            # Either way, we may try to merge orphans below.
            orphan_singletons.append(group[0])
            continue

        # For multi-fragment groups: verify pairwise similarity.
        # Use a simple greedy approach: start with the first fragment,
        # add others if they match any already-accepted fragment.
        sub_groups = _split_group_by_similarity(fragments, group)
        refined.extend(sub_groups)

    # Attempt to merge orphans with each other.
    if orphan_singletons:
        orphan_groups = _merge_orphans(fragments, orphan_singletons)
        refined.extend(orphan_groups)

    # Sort for deterministic output.
    refined.sort(key=lambda g: g[0])
    return refined


def _split_group_by_similarity(
    fragments: List[CandidateFragment],
    group: List[int],
) -> List[List[int]]:
    """
    Within a blocking group, verify pairwise similarity and split if needed.

    Uses greedy clustering: iterate through fragments, assign each to the
    first existing sub-group where it matches any member. If none match,
    start a new sub-group.
    """
    if len(group) <= 1:
        return [group]

    sub_groups: List[List[int]] = [[group[0]]]

    for idx in group[1:]:
        placed = False
        for sg in sub_groups:
            # Check if this fragment matches any fragment in the sub-group.
            if any(should_merge(fragments[idx], fragments[j]) for j in sg):
                sg.append(idx)
                placed = True
                break
        if not placed:
            sub_groups.append([idx])

    return sub_groups


def _merge_orphans(
    fragments: List[CandidateFragment],
    orphan_indices: List[int],
) -> List[List[int]]:
    """
    Attempt to merge orphan fragments using the stricter orphan threshold.

    Greedy clustering approach, same as _split_group_by_similarity but
    using should_merge_orphans instead.
    """
    if not orphan_indices:
        return []

    groups: List[List[int]] = [[orphan_indices[0]]]

    for idx in orphan_indices[1:]:
        placed = False
        for g in groups:
            if any(
                should_merge_orphans(fragments[idx], fragments[j]) for j in g
            ):
                g.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])

    return groups


# ---------------------------------------------------------------------------
# Similarity sub-functions
# ---------------------------------------------------------------------------

def _name_similarity(name_a: str | None, name_b: str | None) -> float:
    """
    Jaro-Winkler similarity between two names. Returns 0.0 if either is None.

    Uses rapidfuzz for performance and licensing (MIT, no GPL).
    Normalizes to [0.0, 1.0].
    """
    if not name_a or not name_b:
        return 0.0
    # rapidfuzz.fuzz returns 0-100 scale; normalize to 0-1.
    # Use token_sort_ratio for robustness against word order differences
    # (e.g. "Doe, Jon" vs "Jon Doe").
    jw = fuzz.token_sort_ratio(name_a.lower(), name_b.lower()) / 100.0
    return jw


def _set_overlap_score(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Binary overlap: 1.0 if the sets share any element, 0.0 otherwise.

    For emails and phones, any shared value is a strong identity signal.
    """
    if not set_a or not set_b:
        return 0.0
    return 1.0 if set_a & set_b else 0.0


def _company_similarity(a: CandidateFragment, b: CandidateFragment) -> float:
    """
    Fuzzy match on company names from experience entries.

    Returns the highest pairwise similarity between any company in A's
    experience and any company in B's experience. Uses token_sort_ratio
    for robustness against formatting differences.
    """
    companies_a = {e.company.lower() for e in a.experience if e.company}
    companies_b = {e.company.lower() for e in b.experience if e.company}

    if not companies_a or not companies_b:
        return 0.0

    best = 0.0
    for ca in companies_a:
        for cb in companies_b:
            score = fuzz.token_sort_ratio(ca, cb) / 100.0
            best = max(best, score)
    return best


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard index: |intersection| / |union|. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    # Case-insensitive comparison.
    a = {s.lower() for s in set_a}
    b = {s.lower() for s in set_b}
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0

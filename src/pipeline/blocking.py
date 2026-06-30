"""
Blocking module for identity resolution.

Generates blocking keys from CandidateFragment records to narrow the
comparison space from O(n²) all-pairs to O(n × block_size). Records sharing
at least one blocking key are placed in the same block for pairwise scoring.

Blocking keys:
  1. Normalized email (lowercased) — e.g. "jon.doe@example.com"
  2. Normalized phone (E.164)     — e.g. "+14155551234"

Records with NO email and NO phone are placed in a special orphan pool,
handled separately by the matching module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from pipeline.normalizers import normalize_phone_e164
from pipeline.schema import CandidateFragment

logger = logging.getLogger(__name__)


def generate_blocking_keys(fragment: CandidateFragment) -> Set[str]:
    """
    Generate blocking keys for a single fragment.

    Returns a set of string keys — one per email and one per parseable phone.
    An empty set means the fragment has no hard identifiers (orphan).
    """
    keys: Set[str] = set()

    # Email keys (already lowercased by the schema validator).
    for email in fragment.emails:
        email = email.strip().lower()
        if email:
            keys.add(f"email:{email}")

    # Phone keys (normalized to E.164).
    for phone in fragment.phones:
        normalized = normalize_phone_e164(phone)
        if normalized:
            keys.add(f"phone:{normalized}")

    return keys


def build_blocks(
    fragments: List[CandidateFragment],
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Assign fragment indices to blocks based on shared blocking keys.

    Returns:
      blocks: dict mapping blocking_key → list of fragment indices
      orphans: list of fragment indices with no blocking keys

    Fragments may appear in multiple blocks (if they have both email and phone).
    The matching module uses connected-component logic to merge across blocks.
    """
    blocks: Dict[str, List[int]] = defaultdict(list)
    orphans: List[int] = []

    for idx, fragment in enumerate(fragments):
        keys = generate_blocking_keys(fragment)
        if not keys:
            orphans.append(idx)
            logger.debug(
                "Fragment %d ('%s') has no blocking keys — orphaned",
                idx,
                fragment.full_name,
            )
        else:
            for key in keys:
                blocks[key].append(idx)

    logger.info(
        "Built %d blocks from %d fragments (%d orphans)",
        len(blocks),
        len(fragments),
        len(orphans),
    )
    return dict(blocks), orphans


def get_candidate_groups(
    fragments: List[CandidateFragment],
) -> List[List[int]]:
    """
    Resolve blocks into connected-component groups of fragment indices.

    If fragment A shares a blocking key with fragment B, and B shares a
    (possibly different) blocking key with C, then A, B, C are all in the
    same group. This handles the case where the same person has different
    emails/phones across sources but a shared one links them transitively.

    Orphans are each returned as singleton groups — the matching module
    may attempt to merge orphans with other orphans via name+company similarity.
    """
    blocks, orphans = build_blocks(fragments)

    # Union-Find for connected components.
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Initialize all fragment indices.
    for idx in range(len(fragments)):
        parent[idx] = idx

    # Union all indices within each block.
    for key, indices in blocks.items():
        for i in range(1, len(indices)):
            union(indices[0], indices[i])

    # Build groups from the union-find structure.
    groups: Dict[int, List[int]] = defaultdict(list)
    # Only group non-orphan indices via union-find.
    blocked_indices = set()
    for indices in blocks.values():
        blocked_indices.update(indices)

    for idx in blocked_indices:
        root = find(idx)
        groups[root].append(idx)

    # Sort within each group for determinism.
    result = [sorted(group) for group in groups.values()]

    # Add orphans as singleton groups.
    for idx in orphans:
        result.append([idx])

    # Sort groups by their first index for deterministic output order.
    result.sort(key=lambda g: g[0])

    logger.info(
        "Resolved %d blocks + %d orphans into %d candidate groups",
        len(blocks),
        len(orphans),
        len(result),
    )
    return result

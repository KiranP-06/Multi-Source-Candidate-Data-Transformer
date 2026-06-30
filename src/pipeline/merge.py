"""
Merge engine with confidence scoring and provenance tracking.

Takes groups of CandidateFragment objects (already identified as the same
person by blocking + matching) and merges each group into a single
CandidateProfile with:
- Field-level conflict resolution with explicit method labels
- Per-field confidence scoring via the confidence formula
- Full provenance tracking showing which source won and how
- Deterministic candidate_id (UUID5)

Conflict resolution priority (from design doc §3.6):
1. single_source — only one source provides the field
2. corroborated — multiple sources agree on the same value
3. conflict_resolution_higher_source_reliability — one source is more reliable
4. conflict_resolution_latest_date — reliability tied, one value is more recent
5. conflict_resolution_majority_vote — 3+ sources, majority agrees
6. Deterministic tie-break: sort by source_id alphabetically, pick first.
   Apply full conflict_penalty. (See design doc §3.6, item 3d.)
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.confidence import (
    compute_field_confidence,
    compute_overall_confidence,
    get_source_reliability,
)
from pipeline.schema import (
    CandidateFragment,
    CandidateProfile,
    EducationEntry,
    ExperienceEntry,
    Links,
    Location,
    Provenance,
    ResolutionMethod,
    Skill,
    generate_candidate_id,
)

logger = logging.getLogger(__name__)


def merge_fragments(
    fragments: List[CandidateFragment],
    required_fields: Optional[Set[str]] = None,
) -> CandidateProfile:
    """
    Merge a group of fragments (same person) into one CandidateProfile.

    Args:
        fragments: List of CandidateFragment objects to merge.
        required_fields: Field names marked required by the active config
                         (used for overall_confidence weighting).

    Returns:
        A fully populated CandidateProfile with provenance.
    """
    if not fragments:
        raise ValueError("Cannot merge an empty list of fragments")

    if required_fields is None:
        required_fields = set()

    provenance: List[Provenance] = []
    field_confidences: Dict[str, float] = {}

    # --- Scalar fields: resolve conflicts ---
    full_name, name_prov, name_conf = _resolve_scalar(
        fragments, "full_name", lambda f: f.full_name
    )
    if name_prov:
        provenance.append(name_prov)
        field_confidences["full_name"] = name_conf

    headline, hl_prov, hl_conf = _resolve_scalar(
        fragments, "headline", lambda f: f.headline
    )
    if hl_prov:
        provenance.append(hl_prov)
        field_confidences["headline"] = hl_conf

    years_exp, ye_prov, ye_conf = _resolve_scalar(
        fragments, "years_experience",
        lambda f: str(f.years_experience) if f.years_experience is not None else None,
    )
    if ye_prov:
        provenance.append(ye_prov)
        field_confidences["years_experience"] = ye_conf

    # --- Location: resolve each sub-field independently ---
    city, city_prov, city_conf = _resolve_scalar(
        fragments, "location.city", lambda f: f.city
    )
    if city_prov:
        provenance.append(city_prov)
        field_confidences["location.city"] = city_conf

    region, region_prov, region_conf = _resolve_scalar(
        fragments, "location.region", lambda f: f.region
    )
    if region_prov:
        provenance.append(region_prov)
        field_confidences["location.region"] = region_conf

    country, country_prov, country_conf = _resolve_scalar(
        fragments, "location.country", lambda f: f.country
    )
    if country_prov:
        provenance.append(country_prov)
        field_confidences["location.country"] = country_conf

    # --- Links: resolve each sub-field ---
    linkedin, li_prov, _ = _resolve_scalar(
        fragments, "links.linkedin", lambda f: f.linkedin_url
    )
    if li_prov:
        provenance.append(li_prov)

    github_url, gh_prov, _ = _resolve_scalar(
        fragments, "links.github", lambda f: f.github_url
    )
    if gh_prov:
        provenance.append(gh_prov)

    portfolio, pf_prov, _ = _resolve_scalar(
        fragments, "links.portfolio", lambda f: f.portfolio_url
    )
    if pf_prov:
        provenance.append(pf_prov)

    other_links = _merge_string_lists(fragments, lambda f: f.other_links)

    # --- List fields: union + deduplicate ---
    emails = _merge_string_lists(fragments, lambda f: f.emails)
    phones = _merge_string_lists(fragments, lambda f: f.phones)

    if emails:
        field_confidences["emails"] = max(
            get_source_reliability(f.source_id) for f in fragments if f.emails
        )
    if phones:
        field_confidences["phones"] = max(
            get_source_reliability(f.source_id) for f in fragments if f.phones
        )

    # --- Skills: union by canonical name, track sources ---
    skills = _merge_skills(fragments)
    if skills:
        field_confidences["skills"] = sum(s.confidence for s in skills) / len(skills)

    # --- Experience: union, deduplicate by (company, title, start) ---
    experience = _merge_experience(fragments)

    # --- Education: union, deduplicate by (institution, degree) ---
    education = _merge_education(fragments)

    # --- Generate deterministic candidate_id ---
    candidate_id = generate_candidate_id(emails, phones, full_name)

    # --- Compute overall confidence ---
    overall = compute_overall_confidence(field_confidences, required_fields)

    # --- Parse years_experience back to float ---
    years_experience_val: Optional[float] = None
    if years_exp is not None:
        try:
            years_experience_val = float(years_exp)
        except (ValueError, TypeError):
            pass

    return CandidateProfile(
        candidate_id=candidate_id,
        full_name=full_name,
        emails=emails,
        phones=phones,
        location=Location(city=city, region=region, country=country),
        links=Links(
            linkedin=linkedin,
            github=github_url,
            portfolio=portfolio,
            other=other_links,
        ),
        headline=headline,
        years_experience=years_experience_val,
        skills=skills,
        experience=experience,
        education=education,
        provenance=provenance,
        overall_confidence=overall,
    )


def merge_all_groups(
    fragments: List[CandidateFragment],
    groups: List[List[int]],
    required_fields: Optional[Set[str]] = None,
) -> List[CandidateProfile]:
    """
    Merge all candidate groups into CandidateProfile objects.

    Args:
        fragments: All fragments (indexed by position).
        groups: List of index-groups from blocking+matching.
        required_fields: Fields marked required by the active config.

    Returns:
        List of CandidateProfile objects, one per group.
    """
    profiles = []
    for group in groups:
        group_fragments = [fragments[i] for i in group]
        try:
            profile = merge_fragments(group_fragments, required_fields)
            profiles.append(profile)
        except Exception as exc:
            logger.error(
                "Failed to merge group %s: %s — skipping", group, exc
            )
    return profiles


# ---------------------------------------------------------------------------
# Scalar field conflict resolution
# ---------------------------------------------------------------------------

def _resolve_scalar(
    fragments: List[CandidateFragment],
    field_name: str,
    extractor,
) -> Tuple[Optional[str], Optional[Provenance], float]:
    """
    Resolve a scalar field across multiple fragments.

    Returns (winning_value, provenance_entry, confidence).
    """
    # Collect (value, source_id, source_timestamp) for each fragment that
    # has a non-None value for this field.
    candidates: List[Tuple[str, str, Optional[datetime]]] = []
    for frag in fragments:
        val = extractor(frag)
        if val is not None:
            candidates.append((str(val), frag.source_id, frag.source_timestamp))

    if not candidates:
        return None, None, 0.0

    # --- Case 1: single source ---
    if len(candidates) == 1:
        value, source_id, ts = candidates[0]
        conf = compute_field_confidence(source_id, n_agreeing=0, n_disagreeing=0, source_timestamp=ts)
        prov = Provenance(
            field=field_name, value=value, source=source_id,
            method="single_source",
        )
        return value, prov, conf

    # Group by normalized value (case-insensitive for comparison).
    value_groups: Dict[str, List[Tuple[str, str, Optional[datetime]]]] = {}
    for val, sid, ts in candidates:
        key = val.strip().lower()
        value_groups.setdefault(key, []).append((val, sid, ts))

    # --- Case 2: all sources agree ---
    if len(value_groups) == 1:
        entries = list(value_groups.values())[0]
        value = entries[0][0]  # take the first (they all agree)
        # Pick the source with highest reliability for provenance.
        best_source = max(entries, key=lambda e: get_source_reliability(e[1]))
        n_agreeing = len(entries) - 1
        conf = compute_field_confidence(
            best_source[1], n_agreeing=n_agreeing, n_disagreeing=0,
            source_timestamp=best_source[2],
        )
        prov = Provenance(
            field=field_name, value=value, source=best_source[1],
            method="corroborated",
        )
        return value, prov, conf

    # --- Case 3: sources disagree ---
    return _resolve_conflict(field_name, candidates, value_groups)


def _resolve_conflict(
    field_name: str,
    candidates: List[Tuple[str, str, Optional[datetime]]],
    value_groups: Dict[str, List[Tuple[str, str, Optional[datetime]]]],
) -> Tuple[Optional[str], Optional[Provenance], float]:
    """
    Resolve a conflict where sources provide different values.

    Priority chain:
    3a. Higher source reliability
    3b. More recent timestamp (if reliability tied)
    3c. Majority vote (3+ sources)
    3d. Deterministic tie-break (alphabetical source_id)
    """
    n_total = len(candidates)

    # Find the best candidate for each unique value.
    value_representatives: List[Tuple[str, str, Optional[datetime], int]] = []
    for key, entries in value_groups.items():
        # Best entry for this value = highest reliability source.
        best = max(entries, key=lambda e: (
            get_source_reliability(e[1]),
            e[2] or datetime.min,  # secondary: most recent timestamp
            # Deterministic tie-break within same value: alphabetical source_id (reversed for max)
        ))
        value_representatives.append(
            (best[0], best[1], best[2], len(entries))  # value, source, timestamp, count
        )

    # --- 3c: Majority vote (if 3+ total candidates) ---
    if n_total >= 3:
        majority_threshold = n_total / 2
        majority = [vr for vr in value_representatives if vr[3] > majority_threshold]
        if len(majority) == 1:
            val, src, ts, count = majority[0]
            n_disagreeing = n_total - count
            conf = compute_field_confidence(
                src, n_agreeing=count - 1, n_disagreeing=n_disagreeing,
                source_timestamp=ts,
            )
            prov = Provenance(
                field=field_name, value=val, source=src,
                method="conflict_resolution_majority_vote",
            )
            return val, prov, conf

    # --- 3a: Higher source reliability ---
    # Sort by (reliability DESC, timestamp DESC, source_id ASC for determinism).
    value_representatives.sort(
        key=lambda vr: (
            -get_source_reliability(vr[1]),
            -(vr[2] or datetime.min).timestamp() if vr[2] else 0,
            vr[1],  # alphabetical source_id for deterministic tie-break
        )
    )

    best = value_representatives[0]
    second = value_representatives[1] if len(value_representatives) > 1 else None

    best_reliability = get_source_reliability(best[1])
    second_reliability = get_source_reliability(second[1]) if second else 0.0

    n_disagreeing = n_total - best[3]

    if best_reliability > second_reliability:
        # --- 3a: Clear reliability winner ---
        method: ResolutionMethod = "conflict_resolution_higher_source_reliability"
    elif second and best[2] and second[2]:
        # Make timestamps tz-aware for comparison.
        best_ts = best[2].replace(tzinfo=None) if best[2] else datetime.min
        second_ts = second[2].replace(tzinfo=None) if second[2] else datetime.min
        if best_ts > second_ts:
            # --- 3b: Reliability tied, but best is more recent ---
            method = "conflict_resolution_latest_date"
        else:
            # --- 3d: True tie — deterministic alphabetical tie-break ---
            # (Already sorted by source_id alphabetically as tertiary key)
            method = "conflict_resolution_higher_source_reliability"
    else:
        # --- 3d: True tie, no timestamps to compare ---
        method = "conflict_resolution_higher_source_reliability"

    conf = compute_field_confidence(
        best[1], n_agreeing=best[3] - 1, n_disagreeing=n_disagreeing,
        source_timestamp=best[2],
    )
    prov = Provenance(
        field=field_name, value=best[0], source=best[1], method=method,
    )
    return best[0], prov, conf


# ---------------------------------------------------------------------------
# List field merging
# ---------------------------------------------------------------------------

def _merge_string_lists(
    fragments: List[CandidateFragment],
    extractor,
) -> List[str]:
    """
    Union and deduplicate string lists across fragments.
    Preserves insertion order, deduplicates case-insensitively.
    """
    seen: Set[str] = set()
    result: List[str] = []
    for frag in fragments:
        for item in extractor(frag):
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(item)
    return sorted(result)  # sorted for determinism


# ---------------------------------------------------------------------------
# Skill merging
# ---------------------------------------------------------------------------

def _merge_skills(fragments: List[CandidateFragment]) -> List[Skill]:
    """
    Merge skills across fragments: union by canonical name, track which
    sources contributed each skill, compute per-skill confidence.
    """
    # skill_name_lower → {canonical_name, sources}
    skill_map: Dict[str, Dict[str, Any]] = {}

    for frag in fragments:
        for skill_name in frag.skills:
            key = skill_name.strip().lower()
            if key not in skill_map:
                skill_map[key] = {"name": skill_name, "sources": set()}
            skill_map[key]["sources"].add(frag.source_id)

    skills = []
    for key in sorted(skill_map.keys()):  # sorted for determinism
        info = skill_map[key]
        n_sources = len(info["sources"])
        # Confidence: average reliability of contributing sources + corroboration bonus.
        avg_reliability = sum(
            get_source_reliability(s) for s in info["sources"]
        ) / n_sources
        corroboration = min(
            (n_sources - 1) * 0.10, 0.20
        )  # same constants as confidence.py
        conf = min(1.0, avg_reliability + corroboration)

        skills.append(Skill(
            name=info["name"],
            confidence=round(conf, 3),
            sources=sorted(info["sources"]),  # sorted for determinism
        ))

    return skills


# ---------------------------------------------------------------------------
# Experience merging
# ---------------------------------------------------------------------------

def _merge_experience(
    fragments: List[CandidateFragment],
) -> List[ExperienceEntry]:
    """
    Union experience entries across fragments, deduplicated by
    (company_lower, title_lower, start) composite key.
    """
    seen: Set[Tuple[str, str, Optional[str]]] = set()
    result: List[ExperienceEntry] = []

    for frag in fragments:
        for exp in frag.experience:
            key = (
                (exp.company or "").lower().strip(),
                (exp.title or "").lower().strip(),
                exp.start,
            )
            if key not in seen:
                seen.add(key)
                result.append(exp)

    return result  # CandidateProfile's model_validator sorts by start desc


# ---------------------------------------------------------------------------
# Education merging
# ---------------------------------------------------------------------------

def _merge_education(
    fragments: List[CandidateFragment],
) -> List[EducationEntry]:
    """
    Union education entries across fragments, deduplicated by
    (institution_lower, degree_lower) composite key.
    """
    seen: Set[Tuple[str, str]] = set()
    result: List[EducationEntry] = []

    for frag in fragments:
        for edu in frag.education:
            key = (
                (edu.institution or "").lower().strip(),
                (edu.degree or "").lower().strip(),
            )
            if key not in seen:
                seen.add(key)
                result.append(edu)

    return result

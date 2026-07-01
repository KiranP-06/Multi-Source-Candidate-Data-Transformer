"""
Edge case tests for the candidate data transformer pipeline.

Each test maps directly to one of the 6 required edge cases from the spec:
1. Conflicting identity, same person
2. Near-duplicate, different people
3. Malformed/empty source
4. Cross-format phone normalization
5. Invalid/missing config request
6. Genuine value conflict, no clear winner

These are integration-level tests: they exercise multiple pipeline stages
together (extract → normalize → block → match → merge → project) to prove
the edge case is handled end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.blocking import get_candidate_groups
from pipeline.confidence import (
    CONFLICT_PENALTY_PER_SOURCE,
    compute_field_confidence,
    get_source_reliability,
)
from pipeline.extractors.csv_extractor import extract_csv
from pipeline.extractors.json_extractor import extract_json
from pipeline.matching import refine_groups
from pipeline.merge import merge_all_groups, merge_fragments
from pipeline.normalizers import normalize_fragment, normalize_phone_e164
from pipeline.projection import ProjectionConfig, load_config, project
from pipeline.schema import (
    CandidateFragment,
    CandidateProfile,
    ExperienceEntry,
    Provenance,
)

from tests import fixture_path


# =========================================================================
# Test 1: Conflicting identity, same person
# =========================================================================

class TestConflictingIdentitySamePerson:
    """
    Two sources spell a name differently but share a normalized email.
    They should be correctly merged into one profile, with provenance
    showing both source values and the resolution method.
    """

    def test_conflicting_identity_same_person(self):
        # Setup: two fragments with same email, different name spellings.
        frag_csv = CandidateFragment(
            source_id="recruiter_csv",
            full_name="Jon Doe",
            emails=["jon.doe@example.com"],
            phones=["(415) 555-1234"],
            experience=[ExperienceEntry(company="Acme Corp", title="Software Engineer")],
        )
        frag_ats = CandidateFragment(
            source_id="ats_json",
            full_name="Jonathan Doe",
            emails=["jon.doe@example.com"],
            phones=["+14155551234"],
            experience=[ExperienceEntry(company="Acme Corp", title="Staff Engineer")],
        )

        # Normalize.
        fragments = [normalize_fragment(frag_csv), normalize_fragment(frag_ats)]

        # Block and match.
        groups = get_candidate_groups(fragments)
        refined = refine_groups(fragments, groups)

        # Assert: merged into ONE group.
        assert len(refined) == 1, (
            f"Expected 1 group (same person), got {len(refined)}: {refined}"
        )
        assert sorted(refined[0]) == [0, 1]

        # Merge.
        profile = merge_fragments([fragments[i] for i in refined[0]])

        # Assert: one profile with shared email appearing once.
        assert profile.emails == ["jon.doe@example.com"]

        # Assert: provenance for full_name shows the resolution method.
        name_provs = [p for p in profile.provenance if p.field == "full_name"]
        assert len(name_provs) == 1
        prov = name_provs[0]
        assert prov.method in (
            "conflict_resolution_higher_source_reliability",
            "conflict_resolution_latest_date",
            "conflict_resolution_majority_vote",
            "corroborated",
        )
        # The winning value should be one of the two names.
        assert prov.value in ("Jon Doe", "Jonathan Doe")

        # Assert: phones are deduplicated (both normalize to same E.164).
        assert len(profile.phones) == 1
        assert profile.phones[0] == "+14155551234"


# =========================================================================
# Test 2: Near-duplicate, different people
# =========================================================================

class TestNearDuplicateDifferentPeople:
    """
    Similar names, no shared email/phone, different companies.
    They should land in different blocks entirely — no similarity score
    is ever computed — and produce two separate candidate_ids.
    """

    def test_near_duplicate_different_people(self):
        # Setup: two fragments with same name but different emails/companies.
        frag_a = CandidateFragment(
            source_id="recruiter_csv",
            full_name="Jane Smith",
            emails=["jane.s@acme.com"],
            phones=["(212) 555-9876"],
            experience=[ExperienceEntry(company="Acme Corp", title="Product Manager")],
        )
        frag_b = CandidateFragment(
            source_id="recruiter_csv",
            full_name="Jane Smith",
            emails=["jsmith@globex.com"],
            phones=["(312) 555-4321"],
            experience=[ExperienceEntry(company="Globex Inc", title="Product Manager")],
        )

        fragments = [normalize_fragment(frag_a), normalize_fragment(frag_b)]

        # Block and match.
        groups = get_candidate_groups(fragments)
        refined = refine_groups(fragments, groups)

        # Assert: two separate groups — fragments never share a block.
        assert len(refined) == 2, (
            f"Expected 2 groups (different people), got {len(refined)}: {refined}"
        )

        # Merge each group separately.
        profiles = merge_all_groups(fragments, refined)
        assert len(profiles) == 2

        # Assert: different candidate_ids.
        ids = {p.candidate_id for p in profiles}
        assert len(ids) == 2, "Two different people should have different IDs"

        # Assert: each profile has only its own email.
        emails_a = {p.emails[0] for p in profiles}
        assert "jane.s@acme.com" in emails_a
        assert "jsmith@globex.com" in emails_a


# =========================================================================
# Test 3: Malformed/empty source
# =========================================================================

class TestMalformedEmptySource:
    """
    A CSV missing columns, a truncated JSON, and an empty file should all
    be handled gracefully — pipeline completes, affected fields are null,
    and at least one valid source's data is present.
    """

    def test_malformed_csv_missing_columns(self):
        """CSV with missing 'name', 'current_company', 'title' columns."""
        fragments = extract_csv(fixture_path("malformed.csv"))
        # Should extract what's available (email, phone) without crashing.
        assert len(fragments) >= 1
        frag = fragments[0]
        assert frag.full_name is None  # 'name' column was missing
        assert len(frag.emails) > 0     # 'email' column was present
        assert len(frag.experience) == 0  # 'current_company' was missing

    def test_truncated_json(self):
        """Invalid/truncated JSON file should not crash the pipeline."""
        fragments = extract_json(fixture_path("truncated.json"))
        assert fragments == []  # No fragments extracted, no crash.

    def test_empty_csv(self):
        """Empty CSV file should return empty list."""
        fragments = extract_csv(fixture_path("empty.csv"))
        assert fragments == []

    def test_pipeline_with_mix_of_valid_and_invalid_sources(self):
        """Full pipeline with malformed + valid sources should complete."""
        # Extract from a mix of valid and invalid sources.
        all_fragments = []
        all_fragments.extend(extract_csv(fixture_path("malformed.csv")))
        all_fragments.extend(extract_json(fixture_path("truncated.json")))
        all_fragments.extend(extract_csv(fixture_path("empty.csv")))
        all_fragments.extend(extract_csv(fixture_path("recruiter.csv")))

        # Should have fragments from the valid CSV at least.
        assert len(all_fragments) >= 4  # recruiter.csv has 4 rows

        # Normalize and merge should not crash.
        fragments = [normalize_fragment(f) for f in all_fragments]
        groups = get_candidate_groups(fragments)
        refined = refine_groups(fragments, groups)
        profiles = merge_all_groups(fragments, refined)

        # Should produce at least some profiles from the valid source.
        assert len(profiles) >= 1


# =========================================================================
# Test 4: Cross-format phone normalization
# =========================================================================

class TestCrossFormatPhoneNormalization:
    """
    Same candidate has a US-format phone from one source and an Indian-format
    phone from another. Both should normalize to E.164. Ambiguous phones
    (no country info) should be dropped, not guessed.
    """

    def test_us_and_indian_phone_normalization(self):
        frag_csv = CandidateFragment(
            source_id="recruiter_csv",
            full_name="Alice Johnson",
            emails=["alice.j@techstart.io"],
            phones=["+91-9876543210"],
        )
        frag_ats = CandidateFragment(
            source_id="ats_json",
            full_name="Alice Johnson",
            emails=["alice.j@techstart.io"],
            phones=["(415) 555-8888"],
        )

        fragments = [normalize_fragment(frag_csv), normalize_fragment(frag_ats)]

        # Both phones should be in E.164 format after normalization.
        all_phones = []
        for f in fragments:
            all_phones.extend(f.phones)

        assert "+919876543210" in all_phones, f"Indian phone not in E.164: {all_phones}"
        assert "+14155558888" in all_phones, f"US phone not in E.164: {all_phones}"

        # Merge and check.
        groups = get_candidate_groups(fragments)
        refined = refine_groups(fragments, groups)
        profiles = merge_all_groups(fragments, refined)
        assert len(profiles) == 1
        profile = profiles[0]
        assert "+919876543210" in profile.phones
        assert "+14155558888" in profile.phones

    def test_ambiguous_phone_dropped(self):
        """A phone like '555-1234' with no country info should be dropped."""
        result = normalize_phone_e164("555-1234")
        assert result is None, f"Ambiguous phone should be None, got {result}"

    def test_short_phone_dropped(self):
        """Very short phone numbers should not be guessed."""
        result = normalize_phone_e164("1234")
        assert result is None, f"Short phone should be None, got {result}"


# =========================================================================
# Test 5: Invalid/missing config request
# =========================================================================

class TestInvalidMissingConfigRequest:
    """
    Config asks for a field/path that doesn't exist in canonical schema,
    or marks a field required that resolves to null. Should produce clean
    behavior per on_missing setting, never silent garbage or a crash.
    """

    def _make_profile(self) -> CandidateProfile:
        """Create a minimal profile for projection testing."""
        return CandidateProfile(
            candidate_id="test-id",
            full_name="Test User",
            emails=["test@example.com"],
        )

    def test_nonexistent_field_on_missing_null(self):
        """Config asks for nonexistent field with on_missing='null'."""
        config = ProjectionConfig(
            fields=[
                {"path": "nonexistent_field", "from_path": "does_not_exist", "required": True},
            ],
            on_missing="null",
        )
        result = project(self._make_profile(), config)
        assert result["nonexistent_field"] is None

    def test_nonexistent_field_on_missing_omit(self):
        """Config asks for nonexistent field with on_missing='omit'."""
        config = ProjectionConfig(
            fields=[
                {"path": "nonexistent_field", "from_path": "does_not_exist"},
            ],
            on_missing="omit",
        )
        result = project(self._make_profile(), config)
        assert "nonexistent_field" not in result

    def test_nonexistent_field_on_missing_error(self):
        """Config asks for required nonexistent field with on_missing='error'."""
        config = ProjectionConfig(
            fields=[
                {"path": "nonexistent_field", "from_path": "does_not_exist", "required": True},
            ],
            on_missing="error",
        )
        with pytest.raises(Exception):  # ValidationError or similar
            project(self._make_profile(), config)

    def test_valid_field_still_works(self):
        """Valid fields should still resolve correctly alongside invalid ones."""
        config = ProjectionConfig(
            fields=[
                {"path": "full_name", "required": True},
                {"path": "primary_email", "from_path": "emails[0]", "required": True},
                {"path": "ghost", "from_path": "does_not_exist"},
            ],
            on_missing="null",
        )
        result = project(self._make_profile(), config)
        assert result["full_name"] == "Test User"
        assert result["primary_email"] == "test@example.com"
        assert result["ghost"] is None


# =========================================================================
# Test 6: Genuine value conflict, no clear winner
# =========================================================================

class TestGenuineValueConflictNoClearWinner:
    """
    Two sources with equal reliability and similar timestamps report
    different values for the same field. Confidence should be lowered,
    and provenance should record the conflict explicitly.
    """

    def test_genuine_conflict_lowers_confidence(self):
        # Setup: two fragments from sources. We mock get_source_reliability
        # so they have EXACTLY equal reliability to force a true tie.
        from unittest.mock import patch
        
        frag_a = CandidateFragment(
            source_id="source_z",            # Alphabetically second
            full_name="Conflict Person",
            emails=["conflict@example.com"],
            headline="Senior Engineer",
            source_timestamp=None,           # No timestamp to force tie
        )
        frag_b = CandidateFragment(
            source_id="source_a",            # Alphabetically first (will win tie-break)
            full_name="Conflict Person",
            emails=["conflict@example.com"],
            headline="Staff Engineer",
            source_timestamp=None,           # No timestamp to force tie
        )

        fragments = [normalize_fragment(frag_a), normalize_fragment(frag_b)]
        
        with patch('pipeline.merge.get_source_reliability', return_value=0.50), \
             patch('pipeline.confidence.get_source_reliability', return_value=0.50):
            profile_conflict = merge_fragments(fragments)

        # Assert: provenance for headline shows the deterministic tie-break method.
        hl_provs = [p for p in profile_conflict.provenance if p.field == "headline"]
        assert len(hl_provs) == 1
        prov = hl_provs[0]
        # In a true tie (same reliability, same/no timestamp, <3 sources), it falls through
        # to the deterministic alphabetical source_id tie-breaker.
        assert prov.method == "conflict_resolution_higher_source_reliability", \
            f"Expected alphabetical tie-break method, got '{prov.method}'"
        assert prov.value == "Staff Engineer"  # source_a comes before source_z
        assert prov.source == "source_a"

        # Assert: overall_confidence is lower than it would be without conflict.
        # Create a no-conflict version (both sources agree).
        frag_agree = CandidateFragment(
            source_id="source_a",
            full_name="Conflict Person",
            emails=["conflict@example.com"],
            headline="Senior Engineer",      # same as frag_a
            source_timestamp=None,
        )
        fragments_agree = [normalize_fragment(frag_a), normalize_fragment(frag_agree)]
        
        with patch('pipeline.merge.get_source_reliability', return_value=0.50), \
             patch('pipeline.confidence.get_source_reliability', return_value=0.50):
            profile_agree = merge_fragments(fragments_agree)

        assert profile_conflict.overall_confidence < profile_agree.overall_confidence, (
            f"Conflict confidence ({profile_conflict.overall_confidence:.3f}) should be "
            f"lower than agreement confidence ({profile_agree.overall_confidence:.3f})"
        )

    def test_conflict_penalty_in_formula(self):
        """Verify the conflict penalty directly reduces field confidence."""
        # No conflict.
        conf_no_conflict = compute_field_confidence(
            source_id="ats_json", n_agreeing=0, n_disagreeing=0,
        )
        # With 1 disagreeing source.
        conf_with_conflict = compute_field_confidence(
            source_id="ats_json", n_agreeing=0, n_disagreeing=1,
        )

        expected_diff = CONFLICT_PENALTY_PER_SOURCE
        actual_diff = conf_no_conflict - conf_with_conflict

        assert abs(actual_diff - expected_diff) < 0.001, (
            f"Expected penalty of {expected_diff}, got {actual_diff}"
        )

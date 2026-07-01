# Multi-Source Candidate Data Transformer

A production-quality pipeline that ingests candidate data from multiple messy sources (structured + unstructured), resolves identity across sources via blocking & fuzzy matching, merges conflicting values with an explicit confidence formula, and outputs one clean canonical profile per candidate — projected through a config-driven output layer.

## Quick Start

```bash
# Install dependencies
pip install pydantic rapidfuzz phonenumbers jsonpath-ng requests python-dateutil eval-type-backport pytest

# Run the pipeline (output to stdout)
PYTHONPATH=src python -m pipeline.cli \
  --inputs fixtures/recruiter.csv fixtures/ats_export.json \
           fixtures/github_mock.json fixtures/resume_sample.txt \
  --config config/default_output.json

# Run with output files
PYTHONPATH=src python -m pipeline.cli \
  --inputs fixtures/recruiter.csv fixtures/ats_export.json \
           fixtures/github_mock.json fixtures/resume_sample.txt \
  --config config/default_output.json \
  --output results.json \
  --canonical-output canonical.json

# Run tests
PYTHONPATH=src python -m pytest tests/ -v

# Run the interactive demo UI
PYTHONPATH=src streamlit run src/pipeline/ui.py
```

## Architecture

### Pipeline Stages

```
detect → extract → normalize → block_and_match → merge_with_confidence
       → project_to_config → validate → output
```

| Stage | Module | Description |
|-------|--------|-------------|
| Detect | `cli.py` | Auto-detect source type by extension + content sniffing |
| Extract | `extractors/` | One adapter per source type → `CandidateFragment` |
| Normalize | `normalizers.py` | Phones → E.164, skills → canonical names, names → NFC title-case |
| Block | `blocking.py` | Group by shared email/phone (union-find for transitive merges) |
| Match | `matching.py` | Pairwise fuzzy similarity within blocks, threshold-based merge decisions |
| Merge | `merge.py` | Field-level conflict resolution with provenance tracking |
| Project | `projection.py` | Config-driven JSONPath-based output shaping |
| Output | `cli.py` | JSON to stdout or file |

### Source Types

| Source | Type | Adapter | Notes |
|--------|------|---------|-------|
| Recruiter CSV | Structured | `csv_extractor.py` | Maps `name`, `email`, `phone`, `current_company`, `title` |
| ATS JSON | Structured | `json_extractor.py` | Non-canonical field names (`applicant_name`, `contact_email`, `job_history`) |
| GitHub Profile | Unstructured | `github_extractor.py` | Offline mock or live API; repo languages → skills |
| Resume (.txt) | Unstructured | `resume_extractor.py` | Section-based regex parsing for experience, education, skills |

### Canonical Schema

Defined in `src/pipeline/schema.py` as Pydantic v2 models:

```
CandidateProfile:
  candidate_id: str             # Deterministic UUID5
  full_name: str | None
  emails: list[str]
  phones: list[str]             # E.164 format
  location: {city, region, country}  # country = ISO 3166 alpha-2
  links: {linkedin, github, portfolio, other[]}
  headline: str | None
  years_experience: float | None
  skills: [{name, confidence, sources[]}]
  experience: [{company, title, start, end, summary}]  # dates as YYYY-MM
  education: [{institution, degree, field, end_year}]
  provenance: [{field, value, source, method}]
  overall_confidence: float     # 0.0 – 1.0
```

---

## Identity Resolution: Matching Threshold & Rationale

### Blocking

Records are grouped by shared **blocking keys** to avoid O(n²) all-pairs comparison:
- **Email**: `email:jon.doe@example.com`
- **Phone**: `phone:+14155551234` (E.164)

Records sharing at least one key are connected via **union-find** (transitive: A↔B + B↔C → A,B,C in one group).

Records with **no email and no phone** (orphans) are compared pairwise only if they pass a strict threshold: Jaro-Winkler name ≥ 0.92 AND company similarity ≥ 0.80.

### Pairwise Similarity Formula

```
similarity(A, B) = 0.30 × name_similarity(A, B)
                 + 0.30 × email_overlap(A, B)
                 + 0.20 × phone_overlap(A, B)
                 + 0.10 × company_similarity(A, B)
                 + 0.10 × skill_jaccard(A, B)
```

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Name (token-sort-ratio) | 0.30 | Strong but not unique — common names exist |
| Email (exact overlap) | 0.30 | Strong identifier, can change over time |
| Phone (exact E.164 overlap) | 0.20 | Strong but less commonly available |
| Company (fuzzy) | 0.10 | Corroborating signal only |
| Skills (Jaccard) | 0.10 | Weakly corroborating — many people share skill sets |

### Merge Threshold: **0.60**

At 0.60, a pair needs a name match (up to 0.30) **PLUS** at least one shared email (0.30) or phone (0.20). A perfect name match alone (0.30) is well under 0.60, preventing false merges of people with similar names at different companies.

**Why this works:**
- "Jon Doe" (CSV) + "Jon Doe" (GitHub), shared email → name(0.30) + email(0.30) = **0.60 → MERGE** ✓
- "Jane Smith" (Acme) + "Jane Smith" (Globex), no shared email/phone → **different blocks, never compared** ✓
- "Jon Doe" + "Jonathan Doe", shared email → name(0.22) + email(0.30) = **0.52** + phone/company/skills push over threshold → **MERGE** ✓

---

## Confidence Formula & Weights

```
confidence(field) = base(source_reliability)
                  + corroboration_bonus(n_agreeing)
                  - staleness_penalty(age)
                  - conflict_penalty(n_disagreeing)
```

### Constants (defined in `src/pipeline/confidence.py`)

| Constant | Value | Rationale |
|----------|-------|-----------|
| **Source reliability** | | |
| `github` | 0.80 | Verified by actual public activity |
| `ats_json` | 0.75 | Structured system-of-record |
| `recruiter_csv` | 0.70 | Manually entered, potentially stale |
| `resume` | 0.65 | Self-reported, heuristic parsing |
| **Formula weights** | | |
| `CORROBORATION_BONUS_PER_SOURCE` | +0.10 | Each agreeing source (capped at +0.20) |
| `STALENESS_PENALTY_PER_YEAR` | −0.02 | Data >1 year old (capped at −0.10) |
| `CONFLICT_PENALTY_PER_SOURCE` | −0.15 | Each disagreeing source (capped at −0.30) |

**Design decision**: The conflict penalty (−0.15) is intentionally heavier than the corroboration bonus (+0.10) because **a wrong-but-confident value is worse than an honestly empty one**.

### Conflict Resolution Priority

1. **`single_source`** — only one source provides the field
2. **`corroborated`** — multiple sources agree
3. **`conflict_resolution_higher_source_reliability`** — one source has strictly higher reliability
4. **`conflict_resolution_latest_date`** — reliability tied, one value is more recent
5. **`conflict_resolution_majority_vote`** — 3+ sources, majority agrees
6. **Deterministic tie-break** — sort by `source_id` alphabetically, pick first; apply full `conflict_penalty`

### Overall Confidence

Weighted average of per-field confidences: fields marked `required` in the active config get weight 2.0, optional fields get 1.0. Clamped to [0.0, 1.0].

---

## Determinism

The pipeline is fully deterministic — identical inputs always produce identical outputs. This is enforced at two critical points:

1. **`candidate_id`**: Uses UUID5 (not UUID4) with a fixed namespace, seeded from the lowest-sorted normalized email or phone. Same identity signals → same ID across runs.
2. **Conflict tie-breaks**: When all resolution signals are exhausted (equal reliability, similar recency, no majority), values are sorted by `source_id` alphabetically and the first is picked. Arbitrary but stable.
3. **Output ordering**: All lists (skills, emails, phones, groups) are sorted for deterministic output.

---

## Config-Driven Projection

Runtime config (JSON) reshapes canonical records into custom output without code changes:

```json
{
  "fields": [
    { "path": "full_name", "type": "string", "required": true },
    { "path": "primary_email", "from": "emails[0]", "type": "string", "required": true },
    { "path": "phone", "from": "phones[0]", "type": "string" },
    { "path": "skills", "from": "skills[].name", "type": "string[]", "normalize": "canonical" }
  ],
  "include_confidence": true,
  "include_provenance": true,
  "on_missing": "null"
}
```

**Features:**
- **JSONPath resolution** via `jsonpath-ng` — `emails[0]`, `skills[*].name`, `links.github`, etc.
- **Normalizer registry** — `"normalize": "E164"` dispatches to the registered function. New normalizers added by registering one function, no core changes.
- **on_missing**: `"null"` (set None), `"omit"` (exclude field), `"error"` (raise if required)
- **Canonical record is read-only** — projection never mutates it

---

## Sample Output

Running the pipeline on all 4 fixture files produces 5 candidate profiles. Here's the first (Jon/Jonathan Doe, merged from 4 sources):

```json
{
  "full_name": "Jon Doe",
  "primary_email": "jon.doe@example.com",
  "phone": "+14155551234",
  "headline": "Backend engineer. Open-source contributor. Coffee addict.",
  "location": { "city": "San Francisco", "region": "CA", "country": null },
  "github": "https://github.com/jondoe",
  "skills": ["AWS", "Docker", "GitHub Actions", "Go", "Kubernetes", "PostgreSQL", "Python", "Redis", "Shell"],
  "years_experience": null,
  "overall_confidence": 0.702,
  "provenance": [
    { "field": "full_name", "value": "Jon Doe", "source": "github", "method": "conflict_resolution_higher_source_reliability" },
    { "field": "headline", "value": "Backend engineer. Open-source contributor. Coffee addict.", "source": "github", "method": "conflict_resolution_higher_source_reliability" },
    { "field": "location.city", "value": "San Francisco", "source": "github", "method": "single_source" },
    { "field": "location.region", "value": "CA", "source": "github", "method": "single_source" },
    { "field": "links.github", "value": "https://github.com/jondoe", "source": "github", "method": "single_source" }
  ]
}
```

**What happened**: "Jon Doe" (CSV), "Jonathan Doe" (ATS + resume), and "Jon Doe" (GitHub) all share `jon.doe@example.com`. Blocking grouped them, similarity scoring confirmed the match, and merge resolved the name conflict in favor of GitHub (highest source reliability: 0.80). Provenance records every resolution decision.

---

## Running Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

### Edge Case Tests (15 tests)

| # | Test | What It Proves |
|---|------|----------------|
| 1 | `test_conflicting_identity_same_person` | Same email + different name spellings → merged, provenance shows method |
| 2 | `test_near_duplicate_different_people` | Same name + different emails/companies → different blocks, never compared |
| 3 | `test_malformed_csv_missing_columns` | Missing CSV columns → extract what's available, rest is null |
| 3 | `test_truncated_json` | Invalid JSON → returns empty list, no crash |
| 3 | `test_empty_csv` | Empty file → returns empty list |
| 3 | `test_pipeline_with_mix_of_valid_and_invalid_sources` | Valid + invalid sources → pipeline completes |
| 4 | `test_us_and_indian_phone_normalization` | US + Indian formats → both E.164 in merged profile |
| 4 | `test_ambiguous_phone_dropped` | `555-1234` → None (never guessed) |
| 4 | `test_short_phone_dropped` | `1234` → None |
| 5 | `test_nonexistent_field_on_missing_null` | Nonexistent JSONPath + `on_missing: null` → field is null |
| 5 | `test_nonexistent_field_on_missing_omit` | `on_missing: omit` → field absent from output |
| 5 | `test_nonexistent_field_on_missing_error` | `on_missing: error` + required → raises error |
| 5 | `test_valid_field_still_works` | Valid fields resolve correctly alongside invalid ones |
| 6 | `test_genuine_conflict_lowers_confidence` | Equal-reliability conflict → lower confidence than agreement |
| 6 | `test_conflict_penalty_in_formula` | Penalty arithmetic directly verified |

---

## Project Structure

```
├── src/pipeline/
│   ├── schema.py          # Pydantic v2 canonical + intermediate models
│   ├── extractors/
│   │   ├── csv_extractor.py     # Recruiter CSV
│   │   ├── json_extractor.py    # ATS JSON blob
│   │   ├── github_extractor.py  # GitHub public API
│   │   └── resume_extractor.py  # Plain-text resume
│   ├── normalizers.py     # Normalizer registry + functions
│   ├── blocking.py        # Blocking-key generation + union-find
│   ├── matching.py        # Pairwise similarity + threshold
│   ├── merge.py           # Field-level merge + confidence + provenance
│   ├── confidence.py      # Confidence formula constants + computation
│   ├── projection.py      # Config-driven JSONPath projection
│   └── cli.py             # CLI entrypoint
├── config/
│   └── default_output.json    # Sample projection config
├── fixtures/                  # Sample data (includes deliberate conflicts)
│   ├── recruiter.csv          # 4 candidates
│   ├── ats_export.json        # 3 applicants (non-canonical field names)
│   ├── github_mock.json       # Saved GitHub API response
│   ├── resume_sample.txt      # Plain-text resume
│   ├── malformed.csv          # Missing columns
│   ├── truncated.json         # Invalid JSON
│   └── empty.csv              # Empty file
├── tests/
│   └── test_edge_cases.py     # 15 tests for all 6 edge cases
├── pyproject.toml
└── README.md
```

---

## Assumptions & Descoped

| Item | Status | Rationale |
|------|--------|-----------|
| **Non-Latin name transliteration** | Descoped | Requires ICU/Unicode CLDR libraries; out of scope. Pipeline handles Unicode NFC normalization but not transliteration (e.g. Cyrillic → Latin). |
| **OCR for scanned PDF resumes** | Descoped | Requires Tesseract + image processing. We support plain-text resumes only. |
| **PDF/DOCX binary parsing** | Descoped | Would add `pdfplumber`/`python-docx` dependencies. Plain-text `.txt` resumes demonstrate the same extraction architecture without binary complexity. The extractor interface (`file_path → list[CandidateFragment]`) supports swapping in a PDF parser with zero downstream changes. |
| **Live LinkedIn scraping** | Descoped (per spec) | LinkedIn's ToS prohibits scraping. Mocked/fixture data used instead. |
| **ML-based entity extraction** | Descoped | Regex + heuristic parsing is sufficient for the demonstrated scope. The extractor architecture supports swapping in an NLP/LLM-based parser later. |
| **Database / persistent storage** | Descoped | Pipeline is stateless, file-in → file-out. A production system would persist profiles to a database. |
| **Country code on Location** | Partial | Country normalization uses a hardcoded ~30-entry dict (US, IN, UK, etc.) rather than `pycountry`. Sufficient for fixture data; extending is a one-line addition. |
| **Greedy clustering** | Intentional | `_split_group_by_similarity` uses single-linkage greedy clustering. If A matches B, and B matches C, all three land in one group even if A and C never get compared directly. This is a deliberate tradeoff for O(n) merging rather than exhaustive cliques. |
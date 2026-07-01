import io
import json
import logging
import tempfile
import sys
from typing import List, Dict, Any, Tuple
import os

import streamlit as st
from pydantic import ValidationError

from pipeline.schema import CandidateProfile
from pipeline.extractors.csv_extractor import extract_csv
from pipeline.extractors.json_extractor import extract_json
from pipeline.extractors.github_extractor import extract_github_from_file
from pipeline.extractors.resume_extractor import extract_resume
from pipeline.normalizers import normalize_fragment
from pipeline.blocking import get_candidate_groups
from pipeline.matching import refine_groups, MATCH_DIAGNOSTICS
from pipeline.merge import merge_all_groups
from pipeline.projection import load_config, project_all, get_required_field_names, ProjectionConfig
from pipeline.cli import extract_file

# =========================================================================
# Backend UI Wrapper
# =========================================================================

class PipelineResult:
    def __init__(
        self,
        canonical_records: List[CandidateProfile],
        projected_output: List[Dict[str, Any]],
        match_diagnostics: List[Dict[str, Any]],
        pipeline_log: str,
    ):
        self.canonical_records = canonical_records
        self.projected_output = projected_output
        self.match_diagnostics = match_diagnostics
        self.pipeline_log = pipeline_log

def run_pipeline_ui(input_paths: List[str], config_json: str) -> PipelineResult:
    """Run the pipeline and return programmatic objects for the UI to render."""
    # Capture logs
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    
    # Remove old handlers to avoid duplicate logs in capture
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.addHandler(handler)

    MATCH_DIAGNOSTICS.clear()  # Clear global state from matching.py

    try:
        # Load config from JSON string instead of file
        raw_config = json.loads(config_json)
        # Handle "from" alias
        for field_spec in raw_config.get("fields", []):
            if "from" in field_spec and "from_path" not in field_spec:
                field_spec["from_path"] = field_spec.pop("from")
        config = ProjectionConfig(**raw_config)
        required_fields = get_required_field_names(config)
        logger.info(f"Loaded config with {len(config.fields)} fields.")

        # Extract
        raw_fragments = []
        for path in input_paths:
            try:
                frags = extract_file(path)
                raw_fragments.extend(frags)
                if not frags:
                    logger.warning(f"Extracted 0 fragments from {path}")
                else:
                    logger.info(f"Extracted {len(frags)} fragments from {path}")
            except Exception as exc:
                logger.error(f"Failed to extract from '{path}': {exc} — skipping")
        
        if not raw_fragments:
            logger.warning("No fragments extracted from any source.")

        # Normalize
        fragments = [normalize_fragment(f) for f in raw_fragments]
        
        # Block and Match
        groups = get_candidate_groups(fragments)
        refined = refine_groups(fragments, groups)
        logger.info(f"Identity resolution: {len(fragments)} fragments → {len(refined)} candidate groups")

        # Merge
        profiles = merge_all_groups(fragments, refined, required_fields=required_fields)
        logger.info(f"Merged into {len(profiles)} candidate profiles")

        # Project
        projected = []
        for profile in profiles:
            # We project manually here to allow catching ValidationError per candidate if needed, 
            # or just project all and let the error bubble up to the UI if on_missing="error".
            from pipeline.projection import project
            projected.append(project(profile, config))
        
        logger.info(f"Projected {len(projected)} profiles")

        return PipelineResult(
            canonical_records=profiles,
            projected_output=projected,
            match_diagnostics=list(MATCH_DIAGNOSTICS),
            pipeline_log=log_capture.getvalue()
        )
    finally:
        logger.removeHandler(handler)

# =========================================================================
# Streamlit App
# =========================================================================

st.set_page_config(page_title="Eightfold Candidate Pipeline Demo", layout="wide")

st.title("Eightfold Multi-Source Candidate Data Transformer")

# --- Default Configs ---
DEFAULT_CONFIG = """{
  "fields": [
    { "path": "full_name", "type": "string", "required": true },
    { "path": "primary_email", "from": "emails[0]", "type": "string", "required": true },
    { "path": "phone", "from": "phones[0]", "type": "string", "normalize": "E164" },
    { "path": "skills", "from": "skills[*].name", "type": "string[]", "normalize": "canonical" }
  ],
  "include_confidence": true,
  "include_provenance": true,
  "on_missing": "null"
}"""

INVALID_CONFIG = """{
  "fields": [
    { "path": "full_name", "type": "string", "required": true },
    { "path": "bad_field", "from": "nonexistent_field[0]", "type": "string", "required": true }
  ],
  "include_confidence": true,
  "include_provenance": true,
  "on_missing": "error"
}"""

SCENARIOS = {
    "1. Same person, conflicting name spelling (shared email)": {
        "files": ["fixtures/recruiter.csv", "fixtures/ats_export.json"],
        "desc": "Proves two different name spellings ('Jon Doe' vs 'Jonathan Doe') merge because they share an email.",
        "config": DEFAULT_CONFIG
    },
    "2. Near-duplicate names, different people (no shared ID)": {
        "files": ["fixtures/recruiter.csv"],
        "desc": "Proves two 'Jane Smith's at different companies stay separate because they lack shared identifiers.",
        "config": DEFAULT_CONFIG
    },
    "3. Malformed/empty source (graceful degradation)": {
        "files": ["fixtures/malformed.csv", "fixtures/truncated.json", "fixtures/empty.csv", "fixtures/recruiter.csv"],
        "desc": "Proves the pipeline survives bad inputs, logs errors, and processes the valid files.",
        "config": DEFAULT_CONFIG
    },
    "4. Cross-format phone (US + India → E.164)": {
        "files": ["fixtures/recruiter.csv"],
        "desc": "Proves phones like '(415) 555-1234' and '+91-9876543210' both normalize correctly to E.164.",
        "config": DEFAULT_CONFIG
    },
    "5. Invalid config (bad path / required-but-null field)": {
        "files": ["fixtures/recruiter.csv"],
        "desc": "Proves the config-driven projection layer correctly halts on 'on_missing': 'error' for required fields.",
        "config": INVALID_CONFIG
    },
    "6. Genuine conflict, no clear winner (title conflict)": {
        "files": ["fixtures/recruiter.csv", "fixtures/resume_sample.txt"],
        "desc": "Proves that equal-reliability sources fall through to deterministic tie-break and lower confidence.",
        "config": DEFAULT_CONFIG
    },
    "Custom upload": {
        "files": [],
        "desc": "Upload your own CSV, JSON, or TXT files.",
        "config": DEFAULT_CONFIG
    }
}

# --- Sidebar ---
with st.sidebar:
    st.header("Demo Controls")
    selected_scenario = st.selectbox("Demo Scenarios", list(SCENARIOS.keys()))
    scenario = SCENARIOS[selected_scenario]
    
    st.caption(f"**Goal**: {scenario['desc']}")
    
    uploaded_files = []
    if selected_scenario == "Custom upload":
        uploaded_files = st.file_uploader("Upload sources", accept_multiple_files=True)
    
    st.subheader("Projection Config")
    config_input = st.text_area("JSON Config", value=scenario["config"], height=250)
    
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

# --- Execution ---
if run_btn:
    input_paths = []
    temp_files = []
    
    # Setup inputs
    if selected_scenario == "Custom upload":
        for uf in uploaded_files:
            # Save to tempfile so extractors can read from disk
            ext = os.path.splitext(uf.name)[1]
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tf.write(uf.read())
            tf.close()
            input_paths.append(tf.name)
            temp_files.append(tf.name)
    else:
        # Verify fixtures exist
        for f in scenario["files"]:
            if os.path.exists(f):
                input_paths.append(f)
            else:
                st.error(f"Fixture not found: {f}")
    
    with st.spinner("Running pipeline..."):
        try:
            result = run_pipeline_ui(input_paths, config_input)
            
            # --- TABS ---
            tab_overview, tab_identity, tab_confidence, tab_config, tab_log = st.tabs([
                "Overview", 
                "Identity Resolution", 
                "Confidence & Provenance", 
                "Config & Projection", 
                "Pipeline Log"
            ])
            
            # 1. Overview Tab
            with tab_overview:
                if not result.canonical_records:
                    st.info("No candidates generated.")
                else:
                    for p in result.canonical_records:
                        with st.container(border=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.subheader(p.full_name or "Unknown")
                                if p.headline:
                                    st.write(f"*{p.headline}*")
                                st.write(f"**Emails**: {', '.join(p.emails) if p.emails else 'None'}")
                                st.write(f"**Phones**: {', '.join(p.phones) if p.phones else 'None'}")
                            with col2:
                                st.metric("Confidence", f"{p.overall_confidence:.2%}")
            
            # 2. Identity Resolution Tab
            with tab_identity:
                if not result.match_diagnostics:
                    st.info("No pairs were compared (only one fragment, or only disjoint blocks).")
                else:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame(result.match_diagnostics),
                        column_config={
                            "decision": st.column_config.TextColumn(
                                "Decision",
                                help="Merged if final_score >= threshold"
                            ),
                        },
                        use_container_width=True,
                        hide_index=True
                    )
            
            # 3. Confidence & Provenance Tab
            with tab_confidence:
                if not result.canonical_records:
                    st.info("No candidates.")
                else:
                    for p in result.canonical_records:
                        st.subheader(f"Provenance for {p.full_name or p.candidate_id}")
                        
                        # --- Math Breakdown ---
                        st.markdown("##### Confidence Math Breakdown")
                        for prov in p.provenance:
                            if prov.components:
                                c = prov.components
                                final_val = c.get('base', 0) + c.get('corroboration', 0) - c.get('staleness', 0) - c.get('conflict', 0)
                                final_val = max(0.0, min(1.0, final_val))
                                st.markdown(
                                    f"- **{prov.field}**: `{c.get('base', 0):.2f}` (base) "
                                    f"+ `{c.get('corroboration', 0):.2f}` (corroboration) "
                                    f"- `{c.get('staleness', 0):.2f}` (staleness) "
                                    f"- `{c.get('conflict', 0):.2f}` (conflict) "
                                    f"= **`{final_val:.2f}`**"
                                )
                        
                        # --- Provenance Table ---
                        st.markdown("##### Provenance Log")
                        prov_data = []
                        for prov in p.provenance:
                            # Build the primary row
                            row = {
                                "Field": prov.field,
                                "Value": prov.value,
                                "Source": prov.source,
                                "Resolution Method": prov.method
                            }
                            # Add alternatives into the value cell if present
                            if prov.alternatives and str(prov.method).startswith("conflict_resolution"):
                                alts_str = " | ".join([f"'{a['value']}' ({a['source']})" for a in prov.alternatives])
                                row["Value"] = f"{prov.value}\n(Also reported: {alts_str} — not selected)"
                                
                            prov_data.append(row)
                            
                        import pandas as pd
                        df_prov = pd.DataFrame(prov_data)
                        
                        def highlight_conflict(val):
                            if isinstance(val, str) and val.startswith("conflict_resolution"):
                                return 'background-color: #ffcccc; color: #900;'
                            return ''
                            
                        if not df_prov.empty and 'Resolution Method' in df_prov.columns:
                            styler = df_prov.style.applymap(highlight_conflict, subset=['Resolution Method'])
                            st.dataframe(styler, use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(df_prov, use_container_width=True, hide_index=True)
            
            # 4. Config & Projection Tab
            with tab_config:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Canonical Record (Internal)")
                    if result.canonical_records:
                        st.json(result.canonical_records[0].model_dump(mode="json"))
                with col2:
                    st.subheader("Projected Output (Final)")
                    if result.projected_output:
                        st.json(result.projected_output[0])
            
            # 5. Pipeline Log Tab
            with tab_log:
                if result.pipeline_log.strip():
                    st.code(result.pipeline_log, language="log")
                else:
                    st.write("No issues in this run.")
                    
        except ValidationError as e:
            st.error("Validation Error during Projection")
            for err in e.errors():
                loc = " -> ".join(str(l) for l in err["loc"])
                st.error(f"Field '{loc}': {err['msg']}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON Config: {e}")
        except Exception as e:
            st.error(f"Pipeline crashed: {type(e).__name__} - {e}")
        finally:
            # Cleanup temp files
            for tf in temp_files:
                try:
                    os.remove(tf)
                except OSError:
                    pass
else:
    st.info("Select a scenario and click 'Run Pipeline' to see results.")

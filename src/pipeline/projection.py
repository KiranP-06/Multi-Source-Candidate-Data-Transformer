"""
Config-driven projection layer.

Takes a CandidateProfile (read-only) and a JSON config, and produces a
projected output dict. The canonical record is NEVER mutated.

Features:
- Field subset selection (only output fields listed in config)
- Rename/remap via "from" (JSONPath expression resolved by jsonpath-ng)
- Per-field normalize via NORMALIZER_REGISTRY
- include_confidence toggle
- include_provenance toggle
- on_missing behavior: "null" | "omit" | "error"
- Dynamic Pydantic model generation for output validation

Uses jsonpath-ng for generic path resolution — no hardcoded nested dict/list
lookups per field.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JsonPathParserError
from pydantic import BaseModel, ValidationError, create_model

from pipeline.normalizers import NORMALIZER_REGISTRY
from pipeline.schema import CandidateProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config schema (Pydantic model for the projection config itself)
# ---------------------------------------------------------------------------

class FieldSpec(BaseModel):
    """Specification for a single projected output field."""
    model_config = {"populate_by_name": True}

    path: str                            # output field name
    type: str = "string"                 # "string", "string[]", "number", etc.
    required: bool = False
    # JSONPath expression to resolve from canonical record.
    # If absent, `path` is used as both the output key and the JSONPath.
    from_path: Optional[str] = None      # aliased from "from" in JSON

    # Normalizer key from NORMALIZER_REGISTRY (e.g. "E164", "canonical").
    normalize: Optional[str] = None


class ProjectionConfig(BaseModel):
    """Runtime projection configuration."""
    fields: List[FieldSpec]
    include_confidence: bool = True
    include_provenance: bool = True
    on_missing: str = "null"   # "null" | "omit" | "error"


def load_config(config_path: str) -> ProjectionConfig:
    """Load a projection config from a JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load config from '{config_path}': {exc}") from exc

    # Handle "from" → "from_path" alias since "from" is a Python keyword.
    for field_spec in raw.get("fields", []):
        if "from" in field_spec and "from_path" not in field_spec:
            field_spec["from_path"] = field_spec.pop("from")

    return ProjectionConfig(**raw)


# ---------------------------------------------------------------------------
# Projection engine
# ---------------------------------------------------------------------------

def project(
    profile: CandidateProfile,
    config: ProjectionConfig,
) -> Dict[str, Any]:
    """
    Project a canonical CandidateProfile into a config-shaped output dict.

    The canonical record is read-only — this function never mutates it.

    Returns a dict matching the config's field specifications.
    Raises ValidationError if on_missing="error" and a required field is missing.
    """
    # Serialize the canonical profile to a dict for JSONPath resolution.
    canonical_dict = profile.model_dump(mode="json")

    output: Dict[str, Any] = {}
    missing_required: List[str] = []

    for field_spec in config.fields:
        output_key = field_spec.path
        jsonpath_expr = field_spec.from_path or field_spec.path

        # Resolve the value via JSONPath.
        value = _resolve_jsonpath(canonical_dict, jsonpath_expr)

        # Apply normalizer if specified.
        if value is not None and field_spec.normalize:
            value = _apply_normalizer(value, field_spec.normalize)

        # Handle missing values.
        if value is None or (isinstance(value, list) and len(value) == 0):
            if field_spec.required:
                if config.on_missing == "error":
                    missing_required.append(output_key)
                elif config.on_missing == "omit":
                    continue  # skip this field entirely
                else:  # "null"
                    output[output_key] = None
            else:
                if config.on_missing == "omit":
                    continue
                else:
                    output[output_key] = None
        else:
            output[output_key] = value

    # Raise if on_missing="error" and required fields are missing.
    if missing_required:
        raise ValidationError.from_exception_data(
            title="ProjectionValidationError",
            line_errors=[
                {
                    "type": "missing",
                    "loc": (field,),
                    "msg": f"Required field '{field}' resolved to null",
                    "input": None,
                }
                for field in missing_required
            ],
        )

    # Optionally include confidence and provenance.
    if config.include_confidence:
        output["overall_confidence"] = profile.overall_confidence
    if config.include_provenance:
        output["provenance"] = [p.model_dump(mode="json") for p in profile.provenance]

    return output


def project_all(
    profiles: List[CandidateProfile],
    config: ProjectionConfig,
) -> List[Dict[str, Any]]:
    """Project a list of profiles, collecting errors without crashing."""
    results = []
    for profile in profiles:
        try:
            projected = project(profile, config)
            results.append(projected)
        except Exception as exc:
            logger.error(
                "Projection failed for candidate '%s': %s — skipping",
                profile.candidate_id,
                exc,
            )
    return results


def get_required_field_names(config: ProjectionConfig) -> Set[str]:
    """Extract the set of field names marked required in the config."""
    return {
        field_spec.path
        for field_spec in config.fields
        if field_spec.required
    }


# ---------------------------------------------------------------------------
# JSONPath resolution
# ---------------------------------------------------------------------------

def _resolve_jsonpath(data: Dict[str, Any], expression: str) -> Any:
    """
    Resolve a JSONPath expression against a data dict.

    Handles:
    - Simple paths: "full_name" → data["full_name"]
    - Indexed paths: "emails[0]" → data["emails"][0]
    - Wildcard paths: "skills[*].name" or "skills[].name" → list of values

    Uses jsonpath-ng for generic resolution. Returns None if the path
    doesn't match anything.
    """
    # Normalize shorthand: "skills[].name" → "skills[*].name"
    # jsonpath-ng uses [*] for wildcard array iteration.
    expression = expression.replace("[]", "[*]")

    # Ensure the expression is rooted — jsonpath-ng needs "$.field" format
    # for top-level fields, but simple names work too.
    if not expression.startswith("$"):
        expression = f"$.{expression}"

    try:
        parsed = jsonpath_parse(expression)
    except (JsonPathParserError, Exception) as exc:
        logger.warning("Invalid JSONPath '%s': %s", expression, exc)
        return None

    matches = parsed.find(data)
    if not matches:
        return None

    # If multiple matches (wildcard), return list of values.
    if len(matches) > 1:
        return [m.value for m in matches]

    # Single match — return the value directly.
    return matches[0].value


def _apply_normalizer(value: Any, normalizer_key: str) -> Any:
    """
    Apply a normalizer function from the registry to a value.

    If the value is a list, applies the normalizer to each element.
    """
    normalizer = NORMALIZER_REGISTRY.get(normalizer_key)
    if normalizer is None:
        logger.warning(
            "Unknown normalizer '%s' — returning value unchanged",
            normalizer_key,
        )
        return value

    try:
        if isinstance(value, list):
            return [normalizer(v) for v in value if v is not None]
        return normalizer(value)
    except Exception as exc:
        logger.warning(
            "Normalizer '%s' failed on value '%s': %s — returning unchanged",
            normalizer_key,
            value,
            exc,
        )
        return value

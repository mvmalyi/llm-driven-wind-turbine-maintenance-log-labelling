"""
JSON Schema definitions for Large Language Model structured outputs.

This module contains all the strict JSON schemas used to enforce the format
of the LLM's responses. It unifies the outputs across the entire pipeline
by guaranteeing that every classification includes a 'Reasoning' string,
a 'Confidence_Score', and a 'Review_Flag'.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# UNIVERSAL FIELDS
# ---------------------------------------------------------------------------
# These fields are injected into all row-level schemas to ensure uniform
# quality assurance and error handling across the entire pipeline.

UNIVERSAL_QA_PROPERTIES: Dict[str, Any] = {
    "Reasoning": {
        "type": "string",
        "description": "A single sentence explanation of why the specific label or code was assigned."
    },
    "Confidence_Score": {
        "type": "string",
        "enum": ["High", "Medium", "Low"],
        "description": "The confidence level in the generated output."
    },
    "Review_Flag": {
        "type": "boolean",
        "description": "Set to true if the text is too ambiguous, vague, or unusual, requiring human review."
    }
}

# ---------------------------------------------------------------------------
# PART 1: SYSTEM CODE CORRECTION SCHEMAS
# ---------------------------------------------------------------------------

SYSTEM_CODE_CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "name": "system_code_classification",
    "schema": {
        "type": "object",
        "properties": {
            "Suggested_Code": {
                "type": "string",
                "description": (
                    "The chosen RDS-PP code, 'Keep Current' if the legacy code is correct, "
                    "or 'Not Enough Information' if the text is entirely too vague."
                )
            },
            **UNIVERSAL_QA_PROPERTIES
        },
        "required": ["Suggested_Code", "Reasoning", "Confidence_Score", "Review_Flag"],
        "additionalProperties": False
    }
}

SYSTEM_CODE_AUDIT_SCHEMA: Dict[str, Any] = {
    "name": "system_code_audit",
    "schema": {
        "type": "object",
        "properties": {
            **UNIVERSAL_QA_PROPERTIES
        },
        # For the audit, the Review_Flag itself dictates if it passed or failed.
        "required": ["Reasoning", "Confidence_Score", "Review_Flag"],
        "additionalProperties": False
    }
}


# ---------------------------------------------------------------------------
# PART 2: SEMANTIC BATCH EXTRACTION SCHEMAS
# ---------------------------------------------------------------------------

FMEA_DICTIONARY_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "name": "fmea_dictionary_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "System_Code": {"type": "string"},
            "System_Name": {"type": "string"},
            "Topology": {"type": ["string", "null"]},
            "Total_Logs_Analyzed": {"type": "integer"},
            "Unclassified_Logs_Count": {"type": "integer"},
            "Empirical_Actions_Taken": {
                "type": "array",
                "items": {"type": "string"}
            },
            "Failure_Modes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Failure_Mode": {"type": "string"},
                        "Mechanism": {"type": "string"},
                        "Cause": {"type": "string"},
                        "Symptom_Effect": {"type": "string"},
                        "Frequency_in_Logs": {"type": "integer"}
                    },
                    "required": [
                        "Failure_Mode", 
                        "Mechanism", 
                        "Cause", 
                        "Symptom_Effect", 
                        "Frequency_in_Logs"
                    ],
                    "additionalProperties": False
                }
            },
            **UNIVERSAL_QA_PROPERTIES
        },
        "required": [
            "System_Code", 
            "System_Name", 
            "Topology", 
            "Total_Logs_Analyzed",
            "Unclassified_Logs_Count",
            "Empirical_Actions_Taken", 
            "Failure_Modes",
            "Reasoning",
            "Confidence_Score",
            "Review_Flag"
        ],
        "additionalProperties": False
    }
}

ACTION_CLUSTERING_SCHEMA: Dict[str, Any] = {
    "name": "action_clustering",
    "schema": {
        "type": "object",
        "properties": {
            "Standardized_New_Actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A concise list of standardized, active-verb actions summarizing the batch."
            },
            **UNIVERSAL_QA_PROPERTIES
        },
        "required": ["Standardized_New_Actions", "Reasoning", "Confidence_Score", "Review_Flag"],
        "additionalProperties": False
    }
}


# ---------------------------------------------------------------------------
# PART 3: GRANULAR LABELLING SCHEMAS (DYNAMIC)
# ---------------------------------------------------------------------------

def _remove_duplicate_enum_values(values: List[str]) -> List[str]:
    """Preserve order while removing duplicate enum values."""
    seen = set()
    unique_values = []
    for value in values:
        value_str = str(value).strip()
        if not value_str or value_str in seen:
            continue
        seen.add(value_str)
        unique_values.append(value_str)
    return unique_values


def get_operational_labelling_schema(
    valid_actions: List[str],
    escape_actions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Dynamically generates the schema for Operational Labelling, locking the
    action choices to the empirically derived list for that specific component.
    """
    escape_values = escape_actions or ["Other", "Not Enough Information"]
    action_enum = _remove_duplicate_enum_values(valid_actions + escape_values)
    return {
        "name": "operational_labelling",
        "schema": {
            "type": "object",
            "properties": {
                "System_Code": {"type": "string"},
                "Maintenance_Type": {
                    "type": "string",
                    "enum": [
                        "Corrective",
                        "Reset",
                        "Preventive",
                        "Predictive",
                        "Inspection",
                        "Retrofit",
                        "Other",
                        "Not Enough Information"
                    ]
                },
                "Action_Taken": {
                    "type": "string",
                    "enum": action_enum
                },
                **UNIVERSAL_QA_PROPERTIES
            },
            "required": [
                "System_Code", 
                "Maintenance_Type", 
                "Action_Taken", 
                "Reasoning",
                "Confidence_Score",
                "Review_Flag"
            ],
            "additionalProperties": False
        }
    }

def _unique_failure_mode_dicts(values: List[Any]) -> List[Dict[str, str]]:
    """Preserve order while removing duplicate failure mode dictionaries."""
    seen = set()
    unique_modes = []
    for value in values:
        if not isinstance(value, dict):
            continue
        failure_mode = str(value.get("Failure_Mode", "")).strip()
        mechanism = str(value.get("Mechanism", "")).strip()
        cause = str(value.get("Cause", "")).strip()
        symptom = str(value.get("Symptom", "") or value.get("Symptom_Effect", "")).strip()
        if not failure_mode:
            continue
        key = (failure_mode, mechanism, cause, symptom)
        if key in seen:
            continue
        seen.add(key)
        unique_modes.append({
            "Failure_Mode": failure_mode,
            "Mechanism": mechanism,
            "Cause": cause,
            "Symptom": symptom
        })
    return unique_modes


def get_semantic_labelling_schema(
    valid_failure_modes: List[Dict[str, str]],
    escape_failure_modes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Dynamically generates the schema for Semantic Labelling (FMEA mapping),
    locking the failure mode choices to the empirically derived dictionary tuples.
    """
    escape_failure_mode_enum = _remove_duplicate_enum_values(
        (escape_failure_modes or ["Other Novel Failure", "Not Enough Information"])
    )
    failure_mode_enum = _unique_failure_mode_dicts(valid_failure_modes)
    selection_options: List[Dict[str, Any]] = []
    if failure_mode_enum:
        selection_options.append({
            "type": "object",
            "enum": failure_mode_enum
        })
    selection_options.append({
        "type": "object",
        "properties": {
            "Failure_Mode": {"type": "string", "enum": escape_failure_mode_enum},
            "Mechanism": {"type": "string"},
            "Cause": {"type": "string"},
            "Symptom": {"type": "string"}
        },
        "required": ["Failure_Mode", "Mechanism", "Cause", "Symptom"],
        "additionalProperties": False
    })
    return {
        "name": "semantic_fmea_labelling",
        "schema": {
            "type": "object",
            "properties": {
                "System_Code": {"type": "string"},
                "Failure_Mode_Selection": {
                    "oneOf": selection_options
                },
                **UNIVERSAL_QA_PROPERTIES
            },
            "required": [
                "System_Code",
                "Failure_Mode_Selection",
                "Reasoning",
                "Confidence_Score",
                "Review_Flag"
            ],
            "additionalProperties": False
        }
    }

NOVEL_FAILURE_GENERATION_SCHEMA: Dict[str, Any] = {
    "name": "novel_failure_generation",
    "schema": {
        "type": "object",
        "properties": {
            "Action_Taken": {"type": "string"},
            "Failure_Mode": {"type": "string"},
            "Mechanism": {"type": "string"},
            "Cause": {"type": "string"},
            "Symptom": {"type": "string"},
            **UNIVERSAL_QA_PROPERTIES
        },
        "required": [
            "Action_Taken", "Failure_Mode", "Mechanism", "Cause", 
            "Symptom", "Reasoning", "Confidence_Score", "Review_Flag"
        ],
        "additionalProperties": False
    }
}

"""
Linguistic prompt templates for the maintenance log labelling pipeline.

This module decouples the natural language instructions from the orchestration
logic. It provides dynamic string templates with placeholders that the
orchestrator will populate at runtime.
"""

# =============================================================================
# UNIVERSAL SYSTEM INSTRUCTIONS
# =============================================================================

BASE_SYSTEM_PERSONA = (
    "You are an expert wind turbine reliability engineer and data scientist. "
    "Your task is to standardise and categorise unstructured maintenance work orders "
    "from a computerised maintenance management system. You must rely on deep "
    "engineering logic, understand varying turbine topologies, and strictly output "
    "your response according to the provided JSON schema."
)

# =============================================================================
# PART 1: SYSTEM CODE CORRECTIONS
# =============================================================================

SYSTEM_CODE_CLASSIFICATION_PROMPT = """
You are tasked with correcting or assigning a hierarchical system code (RDS-PP) to a wind turbine maintenance log.

Applicable Taxonomy Branch:
{taxonomy_text}

Currently Assigned System: {legacy_system_name} (Code: {legacy_system_code})
{topology_context_block}

Task:
Analyse the technician's log description provided below. Based on the mechanical or electrical components mentioned, select the single most accurate system code from the taxonomy provided above. 
If the currently assigned system code is accurate, output 'Keep Current' to preserve it.
If the text is entirely uninformative (e.g., 'turbine stopped' or 'repair done'), you must output 'Not Enough Information' instead of guessing an arbitrary code.

Log Description:
{log_description}
"""

SYSTEM_CODE_AUDIT_PROMPT = """
You are tasked with auditing a legacy system code assignment in a wind turbine maintenance log.

Currently Assigned System: {legacy_system_name} (Code: {legacy_system_code})
{topology_context_block}

Log Description:
{log_description}

Task:
Determine if the provided Log Description clearly contradicts the Currently Assigned System. 
For example, if the assigned system is 'Power Converter' but the text describes replacing 'Pitch Batteries', this is a clear contradiction.
If the legacy code is accurate, or if the text is too vague to definitively contradict it, set the Review_Flag to false.
If there is a clear, undeniable semantic mismatch, set the Review_Flag to true to trigger a relabelling process.
"""

# =============================================================================
# PART 2: SEMANTIC BATCH EXTRACTION (FMEA DICTIONARIES)
# =============================================================================

FMEA_BATCH_EXTRACTION_PROMPT = """
You are tasked with conducting a deep semantic analysis on a batch of maintenance logs. 
All of these logs belong to the following wind turbine system: {system_name} (Code: {system_code}).
{topology_context_block}

Task:
Read the batch of unstructured technician logs provided below. You must distil this historical data into two empirical taxonomies:
1. A list of standard maintenance actions taken (e.g., 'Replace bearing', 'Inspect wiring', 'Reset controller'). Keep these concise and start with an active verb.
2. A list of granular failure modes. For each failure mode, you must deduce the underlying physical Mechanism, the root Cause, and the observable Symptom/Effect based on the text.

Do not invent theoretical failures, only extract what is empirically present in the text batch.

Log Batch:
{log_batch_text}
"""

# =============================================================================
# PART 3: GRANULAR LABELLING (LOG-BY-LOG)
# =============================================================================

OPERATIONAL_LABELLING_PROMPT = """
You are tasked with categorising the operational nature of a wind turbine maintenance event.

System Context: {system_name}
System Code: {system_code}
{topology_context_block}

Log Description:
{log_description}

Task:
0. Return the System_Code exactly as provided in the System Context.
1. Determine the Maintenance_Type from the following allowed list: Corrective, Reset, Preventive, Predictive, Inspection, Retrofit. Remember, a scheduled replacement before a functional failure is Preventive, not Corrective.
2. Determine the specific Action_Taken. You must select from the exact empirical list provided below. If none match perfectly, select 'Other' or 'Not Enough Information' when the log lacks sufficient detail.

Valid Actions List:
{valid_actions_list}
"""

SEMANTIC_FMEA_LABELLING_PROMPT = """
You are tasked with diagnosing a specific wind turbine failure event and mapping it to an established Failure Modes and Effects Analysis (FMEA) dictionary.

System Context: {system_name}
System Code: {system_code}
{topology_context_block}
Maintenance Type: {maintenance_type}
Action Taken: {action_taken}

Log Description:
{log_description}

Task:
Return the System_Code exactly as provided in the System Context.
Based on the text, select the most appropriate failure mode option from the valid dictionary list provided below.
Each option is a JSON object containing Failure_Mode, Mechanism, Cause, and Symptom and must be treated as a single selectable tuple.
Return the selected JSON object exactly in the Failure_Mode_Selection field.
If the text describes a failure that is not in the dictionary, set Failure_Mode to 'Other Novel Failure' and deduce the mechanism, cause, and symptom directly from the text.
If the text is too vague to determine a failure mode, set Failure_Mode to 'Not Enough Information' and use that same phrase for Mechanism, Cause, and Symptom.

Valid Failure Mode Options (JSON Objects):
{valid_failure_modes_list}
"""

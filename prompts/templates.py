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
    "ROLE: a reliability engineer with expertise in wind turbine operations and maintenance. \n"
    "CONTEXT: standardisation and categorisation of unstructured maintenance work orders from a computerised maintenance management system. \n"
    "DIRECTIVE: you must rely on deep engineering logic, understand varying turbine topologies, and strictly output. \n"
    "OUTPUT FORMAT: your response according to the provided JSON schema. \n"
    "A detailed task description is provided below. \n"
)

# =============================================================================
# PART 1: SYSTEM CODE CORRECTIONS
# =============================================================================

SYSTEM_CODE_CLASSIFICATION_PROMPT = """
TASK: correcting (or assigning if missing) a hierarchical system code from the RDS-PP taxonomy provided below to a wind turbine maintenance log.

Currently Assigned System: {legacy_system_name} (Code: {legacy_system_code})
{topology_context_block}

Log Description:
{log_description}

STEPS:
- Analyse the technician's log description provided below.
- Compare the currently assiged system code (if not missing) to the descriptions of the conducted maintenance activities.
- Based on the information provided in the description, select the single most accurate system code from the Applicable Taxonomy Branch provided below with a one-sentence explanation of the reason for this choice.
  Note 1: If the currently assigned system code is accurate, output 'Keep Current' to preserve it.
  Note 2: If the text is entirely uninformative (e.g., 'turbine stopped' or 'repair done'), you must output 'Not Enough Information' instead of guessing an arbitrary code.
- Evaluate the level of confidence in your selection, and determine if the assigned system code needs further human review, all in accordance with the JSON schema used for the output.

***

Applicable Taxonomy Branch:
{taxonomy_text}
"""

SYSTEM_CODE_AUDIT_PROMPT = """
TASK: auditing a legacy system code assignment in a wind turbine maintenance log.

Currently Assigned System: {legacy_system_name} (Code: {legacy_system_code})
{topology_context_block}

Log Description:
{log_description}

STEPS:
- Analyse the technician's log description provided below.
- Determine if the provided Log Description clearly contradicts the Currently Assigned System. 
  For example, if the assigned system is 'Power Converter' but the text describes replacing 'Pitch Batteries', this is a clear contradiction.
  The results of this analysis should be summarised with a one-sentence explanation of the reason for this choice.
- If the legacy code is accurate, or if the text is too vague to definitively contradict it, set the Review_Flag to false.
- If there is a clear, undeniable semantic mismatch, set the Review_Flag to true.
- Evaluate the level of confidence in your output, and determine if the assigned system code needs further human review, all in accordance with the JSON schema used for the output.
"""

# =============================================================================
# PART 2: SEMANTIC BATCH EXTRACTION (FMEA DICTIONARIES)
# =============================================================================

FMEA_BATCH_EXTRACTION_PROMPT = """
TASK: conducting a deep semantic analysis on a batch of maintenance logs to extract an empirical failure modes dictionary, as well as a standardized list of maintenance actions taken.
All of the logs in this batch belong to the following wind turbine system: {system_name} (Code: {system_code}).
{topology_context_block}

STEPS:
1. READ AND RETAIN: Read through all provided logs. 
   - ZERO DATA LOSS RULE: You must account for 100% of the logs. Never filter out mechanical failures, severe faults, or single-occurrence anomalies. Every discrete failure mode must be represented.

2. STANDARDIZE ACTIONS: Extract the distinct physical actions technicians took.
   - VERB SYNTAX RULE: Every action in the `Empirical_Actions_Taken` list MUST begin with an active, present-tense imperative verb (e.g., "Replace", "Repair", "Inspect").
   - ISOLATION RULE: Keep mechanical, electrical, and software actions conceptually distinct. Never merge them into overly broad categories.

3. DEDUCE FAILURE MODE PARAMETERS: Group logs describing similar faults and deduce the failure mode parameters:
   - Failure_Mode: What functionally broke or failed?
   - Mechanism: The physical, chemical, or electrical process of the failure.
   - Cause: The root reason for the mechanism we can infer from the semantic patterns in the historical descriptions of maintenance actions taken and the domain knowledge of the failure mode addressed. 
     ANTI-TAUTOLOGY RULE: Never use circular logic (e.g., you cannot write "Failed pitch motor" as the cause for a motor failure).
     If supported by the historical information, you must state the underlying stressor from the information provided (e.g., "Thermal degradation", "Fatigue wear", "Vibration loosening").
   - Symptom_Effect: How the failure typically manifested. 

4. FAILURE MODE SPLIT RULE: If a single Failure Mode has multiple distinct Causes or Mechanisms observed in the logs, split into separate failure mode rows to preserve the exact causal chain.

5. FREQUENCY: Quantify the number of logs that correspond to each failure mode row to determine their frequency in the provided batch.

6. THE ESCAPE HATCH (UNCLASSIFIABLE LOGS): If a log contains absolutely no technical information regarding a failure, mechanism, or specific action (e.g., "Done", "Inspected turbine", "OK", "No issues found"), do not force it into an failure modes row.
   You must count these non-informative logs separately as unclassified to prevent hallucinations in the failure mode taxonomy created. However, never use this as a reason to skip complex logs; only use it for logs devoid of semantic value. 
   Thus, if a log mentions a specific broken physical component (e.g., 'broken bolt', 'replaced bearing') but without details, you must still classify it as a failure mode and deduce the likely generilised mechanism/cause based on engineering principles.

***

Log Batch:
{log_batch_text}
"""

# =============================================================================
# PART 3: GRANULAR LABELLING (LOG-BY-LOG)
# =============================================================================

OPERATIONAL_LABELLING_PROMPT = """
TASK: analyze a single maintenance log and classify the exact maintenance action taken and the overarching maintenance type.

System Context: {system_name}
System Code: {system_code}
{topology_context_block}

Log Description:
{log_description}

STEPS:
1. Identify the Action Taken
    - Review the log description and select the exact physical action the technician performed from the Approved Actions list below. 
    - You MUST choose an action EXACTLY as it is written in the list above. Never invent or rephrase actions. 
    - If a certain clear action occurred but is not on the list, select "Other".
    - If the log doesn't contain sufficient technical information to infer action taken, select "Not Enough Information".

Approved Actions for this Component:
{valid_actions_list}
   - Other
   - Not Enough Information

2. Classify the Maintenance Type
Based on the initial description and action you identified, classify the Maintenance Type using one of the labels from the list provided below adhering to the following definitions:
    - Corrective: Repairing or replacing a physically failed, damaged, or severely degraded component.
    - Reset: Clearing software faults, resetting breakers, or rebooting systems. No physical parts were repaired/replaced.
    - Preventive: Routine, calendar-based or usage-based servicing (e.g., annual greasing, filter changes).
    - Predictive: Targeted action taken because condition monitoring (CMS) or inspections predicted an imminent failure.
    - Inspection: Visual, auditory, or diagnostic checks ONLY. No immediate repair is executed.
    - Retrofit: Fleet-wide campaigns, OEM upgrades, or design modifications.
    - Other: A clear maintenance action occurred, but it strictly defies the categories above.
    - Not Enough Information: The log is garbage text or lacks sufficient detail to classify the work.

3. Reasoning and Confidence Score
    - Provide a 1-sentence brief explanation for your classifications within Task 1 and Task 2 in the reasoning field with reference to the provided log description. If you select "Other" or "Not Enough Information" for either field, you still need to explain why in the 'Reasoning' field.
    - Evaluate the level of confidence in your selection, and determine if the assigned system code needs further human review, all in accordance with the JSON schema used for the output.
"""

SEMANTIC_FMEA_LABELLING_PROMPT = """
TASK: map a specific maintenance log to the exact set of defined Failure Mode, Mechanism, Cause, and Symptom chain that best describes it.

System Context: {system_name}
System Code: {system_code}
{topology_context_block}
Maintenance Type: {maintenance_type}
Action Taken: {action_taken}

Log Description:
{log_description}

STEPS: 
1. Analyze the log description and the available failure modes taxonomy in the Approved Empirical Failure Modes Dictionary provided below
    - When evaluating each Failure Mode chain in the dictionary, look for matches in the symptom, cause, or mechanism as determined combinations.
    - Then, select the most accurate failure chain best matching the log description.

Rule 1: You MUST choose a failure mode chain EXACTLY as it is written in the dictionary above. Do not mix and match the combinations of Mechanism, Cause, and Symptom or alter their contents.
Rule 2: If the maintenance log described has information but absolutely does not match any provided failure modes options, select "Other" for all four fields.
Rule 3: If the log clearly describes a non-failure event where no component broke (e.g., a routine inspection where everything was fine, or an administrative action), select "Not Applicable" for all four fields.
Rule 4: If the log contains no technical details about what broke or no information to infer what it was for, select "Not Enough Information" for all four fields.

2. Reasoning and Confidence Score
    - Provide a 1-sentence brief explanation for your classifications in the reasoning field with reference to the provided log description. If you select "Other" or "Not Enough Information" for either field, you still need to explain why in the 'Reasoning' field.
    - Evaluate the level of confidence in your selection, and determine if the assigned system code needs further human review, all in accordance with the JSON schema used for the output.

***

Approved Empirical Failure Modes Dictionary:
{valid_failure_modes_list}

Failure_Mode: Other
  - Mechanism: Other
  - Cause: Other
  - Symptom: Other

Failure_Mode: Not Applicable
  - Mechanism: Not Applicable
  - Cause: Not Applicable
  - Symptom: Not Applicable

Failure_Mode: Not Enough Information
  - Mechanism: Not Enough Information
  - Cause: Not Enough Information
  - Symptom: Not Enough Information
"""

"""
Unified pipeline orchestration module for maintenance log labelling.

This module contains the PipelineOrchestrator class, which integrates
the configuration, data management, taxonomy handling, and LLM API client
into a cohesive, four-phase execution workflow with robust checkpointing,
dynamic taxonomy extraction, and post-processing quality assurance.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.config_manager import DataConfig
from src.data_manager import DataManager
from src.llm_clients.openai_client import AsyncOpenAIClient
from src.taxonomy_handler import TaxonomyHandler
from src.performance_logger import PerformanceLogger
from src.exceptions import ConfigurationError, LabellingPipelineError

from prompts.templates import (
    SYSTEM_CODE_CLASSIFICATION_PROMPT,
    SYSTEM_CODE_AUDIT_PROMPT,
    FMEA_BATCH_EXTRACTION_PROMPT,
    OPERATIONAL_LABELLING_PROMPT,
    SEMANTIC_FMEA_LABELLING_PROMPT
)
from prompts.schemas import (
    SYSTEM_CODE_CLASSIFICATION_SCHEMA,
    SYSTEM_CODE_AUDIT_SCHEMA,
    FMEA_DICTIONARY_EXTRACTION_SCHEMA,
    get_operational_labelling_schema,
    get_semantic_labelling_schema
)


# Configure basic logging for the module
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger: logging.Logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Central orchestrator coordinating the four-phase labelling pipeline.
    
    Attributes:
        config (DataConfig): The master configuration object.
        data_manager (DataManager): The robust I/O and merging handler.
        api_client (AsyncOpenAIClient): The schema-agnostic LLM executor.
        taxonomy_handler (TaxonomyHandler): The hierarchical taxonomy manager.
        chunk_size (int): The number of rows to process before saving a checkpoint.
        performance_logger (Optional[PerformanceLogger]): The cost tracking utility.
    """
    # Default non-failure maintenance types are configured in PipelineRulesConfig.
    BATCH_ENTRY_PREFIX = "- "
    BATCH_ENTRY_SEPARATOR = "\n"
    BATCH_TRUNCATION_SUFFIX = "..."
    LEGACY_SYSTEM_CODE_COL = "Legacy_System_Code"
    MERGED_SYSTEM_CODE_COL = "Merged_System_Code"
    SYSTEM_CODE_CORRECTED_FLAG_COL = "System_Code_Corrected_Flag"
    LEGACY_MAINTENANCE_TYPE_COL = "Legacy_Maintenance_Type"
    LEGACY_ACTION_TAKEN_COL = "Legacy_Action_Taken"
    MERGED_MAINTENANCE_TYPE_COL = "Merged_Maintenance_Type"
    MERGED_ACTION_TAKEN_COL = "Merged_Action_Taken"
    MAINTENANCE_TYPE_CORRECTED_FLAG_COL = "Maintenance_Type_Corrected_Flag"
    ACTION_TAKEN_CORRECTED_FLAG_COL = "Action_Taken_Corrected_Flag"
    MERGED_FAILURE_MODE_COL = "Merged_Failure_Mode"
    MERGED_MECHANISM_COL = "Merged_Mechanism"
    MERGED_CAUSE_COL = "Merged_Cause"
    MERGED_SYMPTOM_COL = "Merged_Symptom"
    P1_AUDIT_EXEMPT_COL = "P1_Audit_Exempt"
    P1_WORKFLOW_COL = "P1_Workflow"
    STANDARD_TOPOLOGY_KEY = "Standard Topology"

    def __init__(
        self,
        config: DataConfig,
        data_manager: DataManager,
        api_client: AsyncOpenAIClient,
        taxonomy_handler: TaxonomyHandler,
        chunk_size: int = 50
    ) -> None:
        """Initialise the PipelineOrchestrator with all necessary subsystem modules."""
        self.config: DataConfig = config
        self.data_manager: DataManager = data_manager
        self.api_client: AsyncOpenAIClient = api_client
        self.taxonomy_handler: TaxonomyHandler = taxonomy_handler
        self.chunk_size: int = chunk_size
        self.performance_logger: Optional[PerformanceLogger] = None

    def initialize_performance_logger(self, output_dir: str) -> None:
        """Initialise performance logging for notebook or full-pipeline execution."""
        perf_log_path = os.path.join(output_dir, "performance_log.csv")
        self.performance_logger = PerformanceLogger(self.config.llm, perf_log_path)

    def _ensure_taxonomy_loaded(self) -> None:
        """Ensure the taxonomy tree is loaded before prompt construction."""
        if not self.taxonomy_handler.taxonomy_tree:
            self.taxonomy_handler.load_and_parse_taxonomy()

    def _get_active_system_code(self, row: pd.Series) -> Optional[str]:
        """Retrieve the active system code, preferring merged corrections when available."""
        merged_code = row.get(self.MERGED_SYSTEM_CODE_COL)
        if isinstance(merged_code, str) and merged_code.strip():
            return merged_code.strip()
        legacy_code = row.get(self.config.columns.system_code_col)
        if isinstance(legacy_code, str) and legacy_code.strip():
            return legacy_code.strip()
        return None

    def _normalise_prefixes(self, prefixes: List[str]) -> List[str]:
        """Return non-empty taxonomy prefixes with whitespace removed."""
        return [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]

    def _extract_system_prefix(self, system_code: Optional[str]) -> Optional[str]:
        """Extract the system-level prefix from an RDS-PP code."""
        if not isinstance(system_code, str) or not system_code.strip():
            return None
        parts = system_code.strip().split()
        if len(parts) >= 2:
            return parts[1]
        return parts[0]

    def _prefix_matches(self, system_code: Optional[str], prefixes: List[str]) -> bool:
        """Check whether a system code or its system-level prefix matches any configured prefix."""
        cleaned_prefixes = self._normalise_prefixes(prefixes)
        if not cleaned_prefixes:
            return False
        cleaned_code = system_code.strip() if isinstance(system_code, str) else ""
        system_prefix = self._extract_system_prefix(cleaned_code)
        for prefix in cleaned_prefixes:
            if system_prefix and system_prefix.startswith(prefix):
                return True
            if cleaned_code and cleaned_code.startswith(prefix):
                return True
        return False

    def _field_contains_any(self, value: Any, keywords: List[str]) -> bool:
        """Case-insensitive keyword lookup for system names and descriptions."""
        if value is None:
            return False
        text = str(value).lower()
        return any(keyword in text for keyword in keywords)

    def _resolve_topology_columns(
        self,
        row: pd.Series,
        system_code: Optional[str] = None,
        allow_description_hints: bool = False
    ) -> List[str]:
        """
        Return topology columns relevant to the active system only.

        Topology is intentionally scoped to systems where hardware variants change
        the expected fault profile: pitch, yaw brakes, generator, converter, and
        drivetrain. Other systems are not split by turbine-wide specification fields.
        """
        columns = self.config.columns
        taxonomy = self.config.taxonomy
        active_code = system_code or self._get_active_system_code(row)
        system_name = row.get(columns.system_name_col, "")
        description = row.get(columns.description_col, "")

        def matched(prefixes: List[str], keywords: List[str]) -> bool:
            return (
                self._prefix_matches(active_code, prefixes)
                or self._field_contains_any(system_name, keywords)
                or (
                    allow_description_hints
                    and self._field_contains_any(description, keywords)
                )
            )

        if matched(taxonomy.pitch_system_prefixes, ["pitch"]):
            return [columns.pitch_type_col]
        if matched(taxonomy.yaw_brake_system_prefixes, ["yaw brake", "yaw"]):
            return [columns.yaw_brake_type_col]
        if matched(taxonomy.converter_system_prefixes, ["converter", "inverter"]):
            return [columns.generator_type_col]
        if matched(taxonomy.generator_system_prefixes, ["generator"]):
            return [columns.generator_type_col]
        if matched(taxonomy.drivetrain_system_prefixes, ["drive train", "drivetrain", "drive-train"]):
            return [columns.drive_type_col]
        return []

    def _get_topology_values(
        self,
        row: pd.Series,
        system_code: Optional[str] = None,
        allow_description_hints: bool = False
    ) -> Dict[str, str]:
        """Return populated topology values relevant to the active system."""
        topology_values: Dict[str, str] = {}
        for spec_col in self._resolve_topology_columns(
            row,
            system_code,
            allow_description_hints
        ):
            val = row.get(spec_col)
            if pd.notna(val) and str(val).strip():
                topology_values[spec_col] = str(val).strip()
        return topology_values

    def _build_topology_context(
        self,
        row: pd.Series,
        system_code: Optional[str] = None,
        allow_description_hints: bool = False
    ) -> str:
        """Build a deterministic topology context string for prompts and grouping."""
        topology_values = self._get_topology_values(
            row,
            system_code,
            allow_description_hints
        )
        if not topology_values:
            return self.STANDARD_TOPOLOGY_KEY
        return ", ".join(
            f"{spec_col}: {value}"
            for spec_col, value in topology_values.items()
        )

    def _build_topology_signature(
        self,
        row: pd.Series,
        system_code: Optional[str] = None,
        allow_description_hints: bool = False
    ) -> str:
        """Build a stable topology signature key for dictionary lookup."""
        return self._build_topology_context(
            row,
            system_code,
            allow_description_hints
        )

    def _build_topology_prompt_block(
        self,
        row: pd.Series,
        system_code: Optional[str] = None,
        allow_description_hints: bool = False
    ) -> str:
        """Build an optional topology prompt block, omitting it for non-topology systems."""
        topology_values = self._get_topology_values(
            row,
            system_code,
            allow_description_hints
        )
        if not topology_values:
            return ""
        context = ", ".join(
            f"{spec_col}: {value}"
            for spec_col, value in topology_values.items()
        )
        return f"Topology Context: {context}\n"

    def _has_pitch_mentions(self, text: str) -> bool:
        """Detect pitch-related keywords in a description string."""
        if not text:
            return False
        keywords = [str(term).strip() for term in self.config.taxonomy.pitch_keywords if str(term).strip()]
        if not keywords:
            return False
        pattern = r"\b(" + "|".join(re.escape(term) for term in keywords) + r")\w*\b"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    def _get_pitch_prefixes(self, require: bool = False) -> List[str]:
        """Return configured pitch system prefixes, optionally failing when absent."""
        prefixes = self._normalise_prefixes(self.config.taxonomy.pitch_system_prefixes)
        if require and not prefixes:
            raise ConfigurationError(
                "Pitch-sensitive taxonomy routing requires "
                "TaxonomyConfig.pitch_system_prefixes to be configured."
            )
        return prefixes

    def _coerce_review_flag(self, value: Any) -> bool:
        """Safely normalise review flags loaded from API responses or CSV checkpoints."""
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n", "", "nan", "none"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return bool(value)

    def _is_unprocessed_value(self, value: Any) -> bool:
        """Identify missing or explicit false completion markers."""
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            return not value
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in {"", "false", "0", "nan", "none"}
        return False

    def _passes_threshold(
        self,
        confidence: Optional[str],
        review_flag: Any,
        threshold_config: Any
    ) -> bool:
        """Evaluate a confidence/review pair against configured thresholds."""
        conf_map = {"High": 3, "Medium": 2, "Low": 1, None: 0}
        required_score = conf_map.get(getattr(threshold_config, "required_confidence_level", "High"), 3)
        score = conf_map.get(confidence, 0)
        if score < required_score:
            return False
        if self._coerce_review_flag(review_flag) and not getattr(threshold_config, "allow_human_review_flags", False):
            return False
        return True

    def _resolve_taxonomy_branch(
        self,
        row: pd.Series,
        use_full_taxonomy: bool = False,
        pitch_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Resolve the taxonomy branch to use for system code classification."""
        self._ensure_taxonomy_loaded()

        if use_full_taxonomy:
            return self.taxonomy_handler.taxonomy_tree

        facility_prefix = self.config.taxonomy.facility_prefix
        system_prefix = self.config.taxonomy.system_prefix
        description = str(row.get(self.config.columns.description_col, "") or "")
        pitch_prefixes = self._get_pitch_prefixes(require=pitch_sensitive)
        has_pitch_mention = pitch_sensitive and self._has_pitch_mentions(description)

        legacy_code = row.get(self.config.columns.system_code_col)
        if isinstance(legacy_code, str) and legacy_code.strip():
            parts = legacy_code.strip().split()
            if len(parts) >= 2:
                if not facility_prefix:
                    facility_prefix = parts[0]
                if not system_prefix:
                    system_prefix = parts[1]

        if has_pitch_mention:
            if facility_prefix:
                filtered = self.taxonomy_handler.filter_taxonomy_tree(
                    include_prefixes=pitch_prefixes,
                    facility_prefix=facility_prefix
                )
            else:
                filtered = self.taxonomy_handler.filter_taxonomy_tree(
                    include_prefixes=pitch_prefixes
                )
            if not filtered:
                raise ConfigurationError(
                    "Pitch mention detected, but no matching pitch taxonomy branch "
                    "was found for the configured pitch_system_prefixes."
                )
            return filtered

        branch: Dict[str, Any]
        if facility_prefix and system_prefix:
            branch = self.taxonomy_handler.get_system_branch(facility_prefix, system_prefix)
        elif facility_prefix:
            branch = self.taxonomy_handler.get_facility_branch(facility_prefix)
        else:
            branch = self.taxonomy_handler.taxonomy_tree

        if pitch_sensitive:
            if branch is self.taxonomy_handler.taxonomy_tree:
                filtered = self.taxonomy_handler.filter_taxonomy_tree(
                    exclude_prefixes=pitch_prefixes
                )
            else:
                filtered = self.taxonomy_handler.filter_system_branch(
                    branch,
                    exclude_prefixes=pitch_prefixes
                )
            if not filtered:
                raise ConfigurationError(
                    "Restricted non-pitch taxonomy routing produced an empty branch. "
                    "Check facility_prefix, system_prefix, and pitch_system_prefixes."
                )
            branch = filtered

        return branch or self.taxonomy_handler.taxonomy_tree

    def _redact_text(self, text: str) -> str:
        """Apply configurable redaction rules to a text string."""
        if not text:
            return text

        redaction = self.config.redaction
        if not redaction.enabled:
            return text

        redacted_text = text
        if redaction.custom_redactor:
            try:
                redacted_text = redaction.custom_redactor(redacted_text)
            except Exception as exc:
                logger.warning("Custom redactor failed; continuing without it: %s", exc)

        if redaction.patterns:
            for pattern in redaction.patterns:
                try:
                    redacted_text = re.sub(pattern, redaction.replacement, redacted_text)
                except re.error as exc:
                    logger.warning("Invalid redaction pattern skipped (%s): %s", pattern, exc)

        return redacted_text

    def _build_batched_log_text(self, descriptions: list[str], system_code: str) -> str:
        """Construct a complete Phase 3 prompt batch without arbitrary truncation."""
        max_descriptions = self.config.rules.max_batch_descriptions
        max_chars = self.config.rules.max_batch_chars

        if max_descriptions is not None or max_chars is not None:
            raise ConfigurationError(
                "Phase 3 full-snapshot forwarding is required. "
                "Set max_batch_descriptions and max_batch_chars to None."
            )

        selected = [
            f"{self.BATCH_ENTRY_PREFIX}{desc.strip()}"
            for desc in descriptions
            if desc.strip()
        ]
        return self.BATCH_ENTRY_SEPARATOR.join(selected)

    def _resolve_fmea_dictionary(
        self,
        system_code: Optional[str],
        topology_key: str,
        fmea_dictionaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve the topology-aware FMEA dictionary with backward compatibility."""
        if not system_code:
            return {}
        sys_dict = fmea_dictionaries.get(system_code)
        if not isinstance(sys_dict, dict):
            return {}
        if "Actions" in sys_dict or "Failure_Modes" in sys_dict:
            return sys_dict
        if topology_key in sys_dict and isinstance(sys_dict[topology_key], dict):
            return sys_dict[topology_key]
        standard_key = "Standard Topology"
        if standard_key in sys_dict and isinstance(sys_dict[standard_key], dict):
            return sys_dict[standard_key]
        if len(sys_dict) == 1:
            only_value = next(iter(sys_dict.values()))
            if isinstance(only_value, dict):
                return only_value
        return {}

    def _format_failure_mode_options(self, failure_modes: List[Dict[str, Any]]) -> str:
        """Format failure mode dictionaries for prompt injection."""
        if not failure_modes:
            return ""
        return "\n".join(f"- {json.dumps(mode, ensure_ascii=False)}" for mode in failure_modes)

    async def _process_in_chunks(
        self,
        df: pd.DataFrame,
        target_column: str,
        processor_func: Callable[[pd.Series], Any],
        checkpoint_path: str,
        phase_name: str,
        row_filter: Optional[pd.Series] = None,
        max_failures: Optional[int] = None
    ) -> pd.DataFrame:
        """Helper method to process dataframe rows concurrently with retries and checkpoints."""
        checkpoint_df = self.data_manager.load_checkpoint(checkpoint_path)
        if checkpoint_df is not None:
            if len(checkpoint_df) == len(df):
                df = checkpoint_df
            else:
                logger.warning(
                    "Ignoring checkpoint at %s because its row count (%s) "
                    "does not match the current dataframe (%s).",
                    checkpoint_path,
                    len(checkpoint_df),
                    len(df)
                )

        attempts_column = f"{target_column}_Attempts"
        failed_column = f"{target_column}_Failed"
        failure_reason_column = f"{target_column}_Failure_Reason"

        if attempts_column not in df.columns:
            df[attempts_column] = 0
        if failed_column not in df.columns:
            df[failed_column] = False
        if failure_reason_column not in df.columns:
            df[failure_reason_column] = None

        failed_mask = df[failed_column].apply(self._coerce_review_flag)
        unprocessed_mask = df[target_column].apply(self._is_unprocessed_value) & (~failed_mask)
        if row_filter is not None:
            unprocessed_mask &= row_filter.fillna(False)

        indices_to_process = df[unprocessed_mask].index.tolist()
        
        if not indices_to_process:
            logger.info(f"All rows for {phase_name} already processed. Skipping.")
            return df
            
        logger.info(f"Found {len(indices_to_process)} rows requiring processing for {phase_name}.")
        
        max_failures = max_failures if max_failures is not None else self.config.rules.max_row_retries
        chunk_index = 0

        while indices_to_process:
            chunk_indices = indices_to_process[:self.chunk_size]
            indices_to_process = indices_to_process[self.chunk_size:]
            chunk_index += 1
            logger.info(f"Processing chunk {chunk_index}...")
            
            start_time = time.time()
            start_in_tokens = self.api_client.total_prompt_tokens
            start_out_tokens = self.api_client.total_completion_tokens
            
            tasks = [processor_func(df.loc[idx]) for idx in chunk_indices]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            retry_indices = []
            for idx, result in zip(chunk_indices, results):
                if isinstance(result, Exception):
                    failures = int(df.at[idx, attempts_column]) + 1
                    df.at[idx, attempts_column] = failures
                    if failures < max_failures:
                        logger.warning(
                            "Retrying row %s after failure %s/%s: %s",
                            idx,
                            failures,
                            max_failures,
                            result
                        )
                        retry_indices.append(idx)
                    else:
                        logger.error(
                            "Skipping row %s after %s failed attempts: %s",
                            idx,
                            failures,
                            result
                        )
                        df.at[idx, failed_column] = True
                        df.at[idx, failure_reason_column] = str(result)
                    continue

                for key, value in result.items():
                    df.at[idx, key] = value
                    
            self.data_manager.save_checkpoint(df, checkpoint_path)
            
            end_time = time.time()
            if self.performance_logger:
                self.performance_logger.log_execution_chunk(
                    phase_name=phase_name,
                    chunk_index=chunk_index,
                    rows_processed=len(chunk_indices),
                    input_tokens=self.api_client.total_prompt_tokens - start_in_tokens,
                    output_tokens=self.api_client.total_completion_tokens - start_out_tokens,
                    runtime_seconds=end_time - start_time
                )
            
            indices_to_process.extend(retry_indices)

        return df

    # =========================================================================
    # PHASE 1 & 2: SYSTEM CODE CORRECTIONS AND AUDIT
    # =========================================================================

    async def run_phase_1_system_code_corrections(self, df: pd.DataFrame, checkpoint_path: str) -> pd.DataFrame:
        """Phase 1: Sequential Pitch Correction and Missing Code Recovery workflows."""
        phase_name = "Phase 1: System Code Corrections"
        logger.info(f"Starting {phase_name}")

        # Initialize tracking columns if they don't exist
        if "P1_Complete" not in df.columns:
            df["P1_Complete"] = None
        for col in [
            "P1_System_Code", "P1_Suggested_Code", "P1_Reasoning", 
            "P1_Confidence", "P1_Review_Flag", self.P1_WORKFLOW_COL,
            self.P1_AUDIT_EXEMPT_COL
        ]:
            if col not in df.columns:
                df[col] = None
        
        async def process_row(row: pd.Series, workflow_name: str) -> Dict[str, Any]:
            # The existing routing logic dynamically handles BOTH workflows perfectly.
            # If a row has pitch mentions, it returns the Pitch-Only branch.
            # If a row lacks pitch mentions, it returns the Restricted (Non-Pitch) branch.
            branch = self._resolve_taxonomy_branch(row, pitch_sensitive=True)
            formatted_taxonomy = self.taxonomy_handler.format_branch_for_prompt(branch)
            log_description = self._redact_text(str(row.get(self.config.columns.description_col, "")))
            
            legacy_system_name = str(row.get(self.config.columns.system_name_col, "")).strip() or "Unknown"
            legacy_system_code = str(row.get(self.config.columns.system_code_col, "")).strip() or "Unknown"
            
            prompt = SYSTEM_CODE_CLASSIFICATION_PROMPT.format(
                taxonomy_text=formatted_taxonomy,
                log_description=log_description,
                legacy_system_name=legacy_system_name,
                legacy_system_code=legacy_system_code,
                topology_context_block=self._build_topology_prompt_block(
                    row,
                    allow_description_hints=True
                )
            )
            
            result, _ = await self.api_client.generate_structured_output(
                system_prompt="You are an expert wind turbine reliability engineer.",
                user_message=prompt,
                schema=SYSTEM_CODE_CLASSIFICATION_SCHEMA
            )

            suggested_code = result.get("Suggested_Code")
            existing_code = legacy_system_code if legacy_system_code != "Unknown" else None
            
            final_code = suggested_code
            if isinstance(suggested_code, str) and suggested_code.strip().lower() == "keep current":
                final_code = existing_code

            return {
                "P1_Complete": True,
                "P1_System_Code": final_code,
                "P1_Suggested_Code": suggested_code,
                "P1_Reasoning": result.get("Reasoning"),
                "P1_Confidence": result.get("Confidence_Score"),
                "P1_Review_Flag": result.get("Review_Flag"),
                self.P1_WORKFLOW_COL: workflow_name,
                self.P1_AUDIT_EXEMPT_COL: True
            }

        async def process_pitch_row(row: pd.Series) -> Dict[str, Any]:
            return await process_row(row, "Pitch Correction")

        async def process_recovery_row(row: pd.Series) -> Dict[str, Any]:
            return await process_row(row, "Missing Code Recovery")

        # =====================================================================
        # Step 1: Pitch-Only Workflow
        # =====================================================================
        pitch_mask = df[self.config.columns.description_col].fillna("").astype(str).apply(self._has_pitch_mentions)
        if pitch_mask.any():
            self._get_pitch_prefixes(require=True)
        logger.info(f"Phase 1a: Identified {pitch_mask.sum()} pitch-related logs.")
        
        df = await self._process_in_chunks(
            df, 
            "P1_Complete", 
            process_pitch_row, 
            checkpoint_path, 
            "Phase 1a: Pitch Correction",
            row_filter=pitch_mask
        )

        # =====================================================================
        # Step 2: Missing Code Recovery (Restricted Taxonomy)
        # =====================================================================
        missing_code_mask = (
            df[self.config.columns.system_code_col].isna() | 
            df[self.config.columns.system_code_col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "unknown"])
        )
        recovery_mask = missing_code_mask & ~pitch_mask
        if recovery_mask.any():
            self._get_pitch_prefixes(require=True)
        logger.info(f"Phase 1b: Identified {recovery_mask.sum()} logs requiring missing code recovery.")

        df = await self._process_in_chunks(
            df, 
            "P1_Complete", 
            process_recovery_row, 
            checkpoint_path, 
            "Phase 1b: Missing Code Recovery",
            row_filter=recovery_mask
        )

        # =====================================================================
        # Step 3: Fast-forward remaining logs to Phase 2 (Discrepancy Audit)
        # =====================================================================
        skip_mask = ~(pitch_mask | recovery_mask)
        df.loc[skip_mask, "P1_Complete"] = True
        df.loc[skip_mask, self.P1_WORKFLOW_COL] = "Bypassed"
        df.loc[skip_mask, self.P1_AUDIT_EXEMPT_COL] = False
        df.loc[pitch_mask | recovery_mask, self.P1_AUDIT_EXEMPT_COL] = True
        logger.info(f"Phase 1 complete. Bypassed {skip_mask.sum()} populated logs directly to Phase 2 Audit.")

        return df

    async def run_phase_2_system_code_audit(self, df: pd.DataFrame, checkpoint_path: str) -> pd.DataFrame:
        """Phase 2: Audit legacy system codes for semantic contradictions."""
        phase_name = "Phase 2: System Code Audit"
        logger.info(f"Starting {phase_name}")
        
        if "P2_Audit_Complete" not in df.columns:
            df["P2_Audit_Complete"] = None
        for col in ["P2_Reasoning", "P2_Confidence", "P2_Requires_Relabel"]:
            if col not in df.columns:
                df[col] = None

        audit_exempt_mask = (
            df[self.P1_AUDIT_EXEMPT_COL].apply(self._coerce_review_flag)
            if self.P1_AUDIT_EXEMPT_COL in df.columns
            else pd.Series(False, index=df.index)
        )
        if audit_exempt_mask.any():
            df.loc[audit_exempt_mask, "P2_Audit_Complete"] = True
            df.loc[audit_exempt_mask, "P2_Requires_Relabel"] = False
            df.loc[audit_exempt_mask, "P2_Confidence"] = "High"
            df.loc[audit_exempt_mask, "P2_Reasoning"] = (
                "Skipped because the row was handled by Phase 1 targeted "
                "pitch or missing-code correction."
            )
        audit_mask = ~audit_exempt_mask
            
        async def process_row(row: pd.Series) -> Dict[str, Any]:
            log_description = self._redact_text(str(row[self.config.columns.description_col]))
            prompt = SYSTEM_CODE_AUDIT_PROMPT.format(
                legacy_system_name=row.get(self.config.columns.system_name_col, "Unknown"),
                legacy_system_code=row.get(self.config.columns.system_code_col, "Unknown"),
                log_description=log_description,
                topology_context_block=self._build_topology_prompt_block(
                    row,
                    allow_description_hints=True
                )
            )
            
            result, _ = await self.api_client.generate_structured_output(
                system_prompt="You are an expert data auditor.",
                user_message=prompt,
                schema=SYSTEM_CODE_AUDIT_SCHEMA
            )
            return {
                "P2_Audit_Complete": True,
                "P2_Reasoning": result.get("Reasoning"),
                "P2_Confidence": result.get("Confidence_Score"),
                "P2_Requires_Relabel": result.get("Review_Flag")
            }

        return await self._process_in_chunks(
            df,
            "P2_Audit_Complete",
            process_row,
            checkpoint_path,
            phase_name,
            row_filter=audit_mask
        )

    async def run_phase_2_relabel_failed_audits(self, df: pd.DataFrame, checkpoint_path: str) -> pd.DataFrame:
        """Phase 2b: Relabel system codes for rows that failed the audit."""
        phase_name = "Phase 2b: System Code Relabelling"
        logger.info(f"Starting {phase_name}")

        if "P2_Relabel_Complete" not in df.columns:
            df["P2_Relabel_Complete"] = None
        for col in [
            "P2_Relabel_System_Code",
            "P2_Relabel_Suggested_Code",
            "P2_Relabel_Reasoning",
            "P2_Relabel_Confidence",
            "P2_Relabel_Review_Flag"
        ]:
            if col not in df.columns:
                df[col] = None

        if "P2_Requires_Relabel" not in df.columns:
            logger.info("Audit relabel flag not found; skipping relabel step.")
            return df

        relabel_mask = df["P2_Requires_Relabel"].apply(self._coerce_review_flag)
        if self.P1_AUDIT_EXEMPT_COL in df.columns:
            relabel_mask &= ~df[self.P1_AUDIT_EXEMPT_COL].apply(self._coerce_review_flag)
        if not relabel_mask.any():
            logger.info("No audit failures detected; skipping relabel step.")
            return df

        async def process_row(row: pd.Series) -> Dict[str, Any]:
            branch = self._resolve_taxonomy_branch(row, use_full_taxonomy=True)
            formatted_taxonomy = self.taxonomy_handler.format_branch_for_prompt(branch)
            log_description = self._redact_text(str(row[self.config.columns.description_col]))
            legacy_system_name = row.get(self.config.columns.system_name_col)
            if not isinstance(legacy_system_name, str) or not legacy_system_name.strip():
                legacy_system_name = "Unknown"
            else:
                legacy_system_name = legacy_system_name.strip()
            legacy_system_code = row.get(self.config.columns.system_code_col)
            if not isinstance(legacy_system_code, str) or not legacy_system_code.strip():
                legacy_system_code = "Unknown"
            else:
                legacy_system_code = legacy_system_code.strip()

            prompt = SYSTEM_CODE_CLASSIFICATION_PROMPT.format(
                taxonomy_text=formatted_taxonomy,
                log_description=log_description,
                legacy_system_name=legacy_system_name,
                legacy_system_code=legacy_system_code,
                topology_context_block=self._build_topology_prompt_block(
                    row,
                    allow_description_hints=True
                )
            )

            result, _ = await self.api_client.generate_structured_output(
                system_prompt="You are an expert wind turbine reliability engineer.",
                user_message=prompt,
                schema=SYSTEM_CODE_CLASSIFICATION_SCHEMA
            )

            suggested_code = result.get("Suggested_Code")
            existing_code = row.get(self.config.columns.system_code_col)
            final_code = existing_code
            if isinstance(suggested_code, str) and suggested_code.strip().lower() != "keep current":
                final_code = suggested_code

            return {
                "P2_Relabel_Complete": True,
                "P2_Relabel_System_Code": final_code,
                "P2_Relabel_Suggested_Code": suggested_code,
                "P2_Relabel_Reasoning": result.get("Reasoning"),
                "P2_Relabel_Confidence": result.get("Confidence_Score"),
                "P2_Relabel_Review_Flag": result.get("Review_Flag")
            }

        return await self._process_in_chunks(
            df,
            "P2_Relabel_Complete",
            process_row,
            checkpoint_path,
            phase_name,
            row_filter=relabel_mask
        )

    def apply_system_code_merges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge system code corrections using configured confidence/review thresholds.
        """
        merged_col = self.MERGED_SYSTEM_CODE_COL
        corrected_flag_col = self.SYSTEM_CODE_CORRECTED_FLAG_COL
        if self.LEGACY_SYSTEM_CODE_COL not in df.columns:
            df[self.LEGACY_SYSTEM_CODE_COL] = df.get(self.config.columns.system_code_col)
        if merged_col not in df.columns:
            df[merged_col] = None
        if corrected_flag_col not in df.columns:
            df[corrected_flag_col] = False

        thresholds = self.config.rules.system_code_thresholds

        def normalize_code(value: Any) -> Optional[str]:
            if isinstance(value, str):
                cleaned = value.strip()
                return cleaned if cleaned else None
            return None

        def is_invalid_code(value: Optional[str]) -> bool:
            if not value:
                return True
            lowered = value.strip().lower()
            return lowered in {"not enough information", "keep current"}

        for idx, row in df.iterrows():
            legacy_code = normalize_code(row.get(self.config.columns.system_code_col))
            p2_code = normalize_code(row.get("P2_Relabel_System_Code"))
            p2_conf = row.get("P2_Relabel_Confidence")
            p2_review = self._coerce_review_flag(row.get("P2_Relabel_Review_Flag", False))
            p1_code = normalize_code(row.get("P1_System_Code"))
            p1_conf = row.get("P1_Confidence")
            p1_review = self._coerce_review_flag(row.get("P1_Review_Flag", False))
            p1_audit_exempt = self._coerce_review_flag(row.get(self.P1_AUDIT_EXEMPT_COL, False))

            selected_code = legacy_code
            if (
                p1_audit_exempt
                and not is_invalid_code(p1_code)
                and self._passes_threshold(p1_conf, p1_review, thresholds)
            ):
                selected_code = p1_code
            elif not is_invalid_code(p2_code) and self._passes_threshold(p2_conf, p2_review, thresholds):
                selected_code = p2_code
            elif not is_invalid_code(p1_code) and self._passes_threshold(p1_conf, p1_review, thresholds):
                selected_code = p1_code

            merged_code = selected_code or legacy_code
            df.at[idx, merged_col] = merged_code
            df.at[idx, corrected_flag_col] = bool(
                (legacy_code != merged_code) and (merged_code is not None)
            )

        return df

    # =========================================================================
    # PHASE 3: SEMANTIC BATCH EXTRACTION
    # =========================================================================

    async def run_phase_3_batch_extraction(self, df: pd.DataFrame, dict_path: str) -> Dict[str, Any]:
        """Phase 3: Extract empirical FMEA dictionaries from batched system logs."""
        phase_name = "Phase 3: Semantic Batch Extraction"
        logger.info(f"Starting {phase_name}")

        # Task 2: Check for existing JSON cache
        if os.path.exists(dict_path):
            try:
                extracted_dictionaries = self.data_manager.load_json(dict_path)
                logger.info("Loaded existing FMEA dictionaries from disk. Skipping API extraction.")
                return extracted_dictionaries
            except Exception as e:
                logger.warning(f"Failed to load existing dictionary: {e}. Regenerating...")

        start_time = time.time()
        start_in_tokens = self.api_client.total_prompt_tokens
        start_out_tokens = self.api_client.total_completion_tokens

        extracted_dictionaries = {}
        working_df = df.copy()
        working_df["_Active_System_Code"] = working_df.apply(self._get_active_system_code, axis=1)
        working_df["_Topology_Key"] = working_df.apply(
            lambda row: self._build_topology_signature(
                row,
                row.get("_Active_System_Code")
            ),
            axis=1
        )
        grouped_items = list(working_df.groupby(["_Active_System_Code", "_Topology_Key"], dropna=False))
        phase_concurrency = (
            self.config.llm.phase3_max_concurrent_requests
            or self.config.llm.max_concurrent_requests
        )
        phase_semaphore = asyncio.Semaphore(phase_concurrency)
        original_reasoning_effort = self.api_client.config.reasoning_effort
        if self.config.llm.phase3_reasoning_effort:
            self.api_client.config.reasoning_effort = self.config.llm.phase3_reasoning_effort

        async def process_system_group(
            system_code: Optional[str],
            topology_key: str,
            group_df: pd.DataFrame
        ) -> tuple[tuple[str, str], Dict[str, Any]]:
            """Process a single system group asynchronously."""
            async with phase_semaphore:
                descriptions = (
                    group_df[self.config.columns.description_col]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
                redacted_descriptions = [
                    self._redact_text(desc).strip()
                    for desc in descriptions
                    if desc.strip()
                ]
                system_code_value = (
                    system_code.strip()
                    if isinstance(system_code, str) and system_code.strip()
                    else None
                )
                system_code_value = system_code_value or "Unknown"
                batch_text = self._build_batched_log_text(redacted_descriptions, system_code_value)
                topology_context = topology_key or self.STANDARD_TOPOLOGY_KEY
                sys_name = str(group_df[self.config.columns.system_name_col].iloc[0]) if self.config.columns.system_name_col in group_df else "Unknown System"
                representative_row = group_df.iloc[0]

                prompt = FMEA_BATCH_EXTRACTION_PROMPT.format(
                    system_name=sys_name,
                    system_code=system_code_value,
                    topology_context_block=self._build_topology_prompt_block(
                        representative_row,
                        system_code_value
                    ),
                    log_batch_text=batch_text
                )

                result, _ = await self.api_client.generate_structured_output(
                    system_prompt="You are an expert wind turbine reliability engineer.",
                    user_message=prompt,
                    schema=FMEA_DICTIONARY_EXTRACTION_SCHEMA
                )

                actions = result.get("Empirical_Actions_Taken", [])
                failures = result.get("Failure_Modes", [])
                cleaned_failures = []
                if isinstance(failures, list):
                    for failure in failures:
                        if not isinstance(failure, dict):
                            continue
                        failure_mode = str(failure.get("Failure_Mode", "")).strip()
                        mechanism = str(failure.get("Mechanism", "")).strip()
                        cause = str(failure.get("Cause", "")).strip()
                        symptom = str(failure.get("Symptom_Effect", "") or failure.get("Symptom", "")).strip()
                        frequency = failure.get("Frequency_in_Logs")
                        if not failure_mode:
                            continue
                        cleaned_failure = {
                            "Failure_Mode": failure_mode,
                            "Mechanism": mechanism,
                            "Cause": cause,
                            "Symptom": symptom
                        }
                        if frequency is not None:
                            cleaned_failure["Frequency_in_Logs"] = frequency
                        cleaned_failures.append(cleaned_failure)

                return (system_code_value, topology_context), {
                    "Actions": actions if isinstance(actions, list) else [],
                    "Failure_Modes": cleaned_failures
                }

        logger.info(f"Extracting dictionaries for {len(grouped_items)} system topology batches...")
        tasks = [
            process_system_group(system_code, topology_key, group)
            for (system_code, topology_key), group in grouped_items
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.api_client.config.reasoning_effort = original_reasoning_effort

        for res in results:
            if isinstance(res, Exception):
                if isinstance(res, ConfigurationError):
                    raise res
                logger.error(f"Error extracting dictionary for a system group: {res}")
            else:
                (code, topology_key), sys_dict = res
                extracted_dictionaries.setdefault(code, {})
                extracted_dictionaries[code][topology_key] = sys_dict

        # Save the freshly extracted dictionaries to disk
        self.data_manager.save_json(extracted_dictionaries, dict_path)

        end_time = time.time()
        if self.performance_logger:
            self.performance_logger.log_execution_chunk(
                phase_name=phase_name,
                chunk_index=1,
                rows_processed=len(df),
                input_tokens=self.api_client.total_prompt_tokens - start_in_tokens,
                output_tokens=self.api_client.total_completion_tokens - start_out_tokens,
                runtime_seconds=end_time - start_time
            )
            
        return extracted_dictionaries

    # =========================================================================
    # PHASE 4: GRANULAR LABELLING
    # =========================================================================

    async def run_phase_4_granular_labelling(
        self, 
        df: pd.DataFrame, 
        fmea_dictionaries: Dict[str, Any],
        checkpoint_path: str
    ) -> pd.DataFrame:
        """Phase 4: Apply granular operational and FMEA labels log-by-log."""
        phase_name = "Phase 4: Granular Labelling"
        logger.info(f"Starting {phase_name}")
        
        target_columns = [
            "P4_System_Code", "P4_Maintenance_Type", "P4_Action_Taken", "P4_Failure_Mode",
            "P4_Mechanism", "P4_Cause", "P4_Symptom", "P4_Reasoning", 
            "P4_Confidence", "P4_Review_Flag",
            "P4_Operational_Confidence", "P4_Operational_Review_Flag",
            "P4_Semantic_Confidence", "P4_Semantic_Review_Flag"
        ]
        for col in target_columns:
            if col not in df.columns:
                df[col] = None

        # Normalize configured skip types and drop null/blank entries.
        configured_skip_types = {
            cleaned
            for item in self.config.rules.non_failure_maintenance_types
            if item is not None and (cleaned := str(item).strip())
        }
        skip_types = configured_skip_types
                
        async def process_row(row: pd.Series) -> Dict[str, Any]:
            desc = str(row.get(self.config.columns.description_col, "")).strip()
            sys_code = self._get_active_system_code(row) or "Unknown"
            topology_context = self._build_topology_context(row, sys_code)
            topology_key = self._build_topology_signature(row, sys_code)
            topology_context_block = self._build_topology_prompt_block(row, sys_code)
            
            if self.config.rules.skip_missing_descriptions and (not desc or len(desc) < 5):
                garbage = self.config.rules.garbage_data_label
                return {
                    "P4_System_Code": sys_code,
                    "P4_Maintenance_Type": garbage,
                    "P4_Action_Taken": garbage,
                    "P4_Failure_Mode": garbage,
                    "P4_Mechanism": garbage,
                    "P4_Cause": garbage,
                    "P4_Symptom": garbage,
                    "P4_Reasoning": "Deterministically skipped due to missing or uninformative description.",
                    "P4_Confidence": "High",
                    "P4_Review_Flag": False,
                    "P4_Operational_Confidence": "High",
                    "P4_Operational_Review_Flag": False,
                    "P4_Semantic_Confidence": "High",
                    "P4_Semantic_Review_Flag": False
                }

            redacted_desc = self._redact_text(desc)

            sys_dict = self._resolve_fmea_dictionary(sys_code, topology_key, fmea_dictionaries)
            raw_actions = sys_dict.get("Actions")
            raw_failures = sys_dict.get("Failure_Modes")
            valid_actions = raw_actions if isinstance(raw_actions, list) else []
            valid_failures = []
            if isinstance(raw_failures, list):
                for failure in raw_failures:
                    if isinstance(failure, dict):
                        failure_mode = str(failure.get("Failure_Mode", "")).strip()
                        if not failure_mode:
                            continue
                        valid_failures.append({
                            "Failure_Mode": failure_mode,
                            "Mechanism": str(failure.get("Mechanism", "")).strip(),
                            "Cause": str(failure.get("Cause", "")).strip(),
                            "Symptom": str(failure.get("Symptom", "") or failure.get("Symptom_Effect", "")).strip()
                        })
                    elif isinstance(failure, str) and failure.strip():
                        placeholder = self.config.rules.garbage_data_label
                        valid_failures.append({
                            "Failure_Mode": failure.strip(),
                            "Mechanism": placeholder,
                            "Cause": placeholder,
                            "Symptom": placeholder
                        })
            action_prompt_values = (
                valid_actions
                if valid_actions
                else self.config.rules.action_escape_categories
            )
            failure_prompt_values = self._format_failure_mode_options(valid_failures)
            if not failure_prompt_values:
                failure_prompt_values = ", ".join(self.config.rules.failure_escape_categories)
            
            op_schema = get_operational_labelling_schema(
                valid_actions,
                escape_actions=self.config.rules.action_escape_categories
            )
            op_prompt = OPERATIONAL_LABELLING_PROMPT.format(
                system_name=row.get(self.config.columns.system_name_col, "Unknown"),
                system_code=sys_code,
                topology_context_block=topology_context_block,
                log_description=redacted_desc,
                valid_actions_list=", ".join(str(action) for action in action_prompt_values)
            )
            
            op_result, _ = await self.api_client.generate_structured_output(
                system_prompt="You are an expert wind turbine reliability engineer.",
                user_message=op_prompt,
                schema=op_schema
            )
            
            op_system_code = op_result.get("System_Code", sys_code)
            if isinstance(op_system_code, str):
                op_system_code = op_system_code.strip()
            if op_system_code != sys_code:
                logger.warning(
                    "Operational labelling returned mismatched System_Code '%s' for work order %s; using '%s'.",
                    op_system_code,
                    row.get(self.config.columns.wo_id_col, "Unknown"),
                    sys_code
                )
                op_system_code = sys_code
            maint_type = op_result.get("Maintenance_Type", "Other")
            if not isinstance(maint_type, str):
                maint_type = str(maint_type)
            maint_type = maint_type.strip()
            action_taken = op_result.get("Action_Taken", "Other")
            op_review_flag = self._coerce_review_flag(op_result.get("Review_Flag", False))
            op_confidence = op_result.get("Confidence_Score", "Low")
            op_reasoning = op_result.get("Reasoning", "")
            
            output_dict = {
                "P4_System_Code": op_system_code,
                "P4_Maintenance_Type": maint_type,
                "P4_Action_Taken": action_taken,
                "P4_Reasoning": f"Operational: {op_reasoning}",
                "P4_Confidence": op_confidence,
                "P4_Review_Flag": op_review_flag,
                "P4_Operational_Confidence": op_confidence,
                "P4_Operational_Review_Flag": op_review_flag,
                "P4_Semantic_Confidence": None,
                "P4_Semantic_Review_Flag": None
            }
            
            if self.config.rules.skip_preventive_maintenance and maint_type in skip_types:
                healthy = self.config.rules.healthy_turbine_label
                output_dict.update({
                    "P4_Failure_Mode": healthy,
                    "P4_Mechanism": healthy,
                    "P4_Cause": healthy,
                    "P4_Symptom": healthy,
                    "P4_Reasoning": f"{output_dict['P4_Reasoning']} | FMEA: Skipped deterministically as maintenance type is {maint_type}.",
                    "P4_Semantic_Confidence": "High",
                    "P4_Semantic_Review_Flag": False
                })
                return output_dict

            sem_schema = get_semantic_labelling_schema(
                valid_failures,
                escape_failure_modes=self.config.rules.failure_escape_categories
            )
            sem_prompt = SEMANTIC_FMEA_LABELLING_PROMPT.format(
                system_name=row.get(self.config.columns.system_name_col, "Unknown"),
                system_code=sys_code,
                topology_context_block=topology_context_block,
                maintenance_type=maint_type,
                action_taken=action_taken,
                log_description=redacted_desc,
                valid_failure_modes_list=failure_prompt_values
            )
            
            sem_result, _ = await self.api_client.generate_structured_output(
                system_prompt="You are an expert wind turbine reliability engineer.",
                user_message=sem_prompt,
                schema=sem_schema
            )
            
            sem_system_code = sem_result.get("System_Code", sys_code)
            if isinstance(sem_system_code, str):
                sem_system_code = sem_system_code.strip()
            if sem_system_code != sys_code:
                logger.warning(
                    "Semantic labelling returned mismatched System_Code '%s' for work order %s; using '%s'.",
                    sem_system_code,
                    row.get(self.config.columns.wo_id_col, "Unknown"),
                    sys_code
                )
                sem_system_code = sys_code
            sem_review_flag = self._coerce_review_flag(sem_result.get("Review_Flag", False))
            sem_confidence = sem_result.get("Confidence_Score", "Low")
            sem_reasoning = sem_result.get("Reasoning", "")
            selection = sem_result.get("Failure_Mode_Selection") if isinstance(sem_result, dict) else None
            if not isinstance(selection, dict):
                selection = {
                    "Failure_Mode": sem_result.get("Failure_Mode"),
                    "Mechanism": sem_result.get("Mechanism"),
                    "Cause": sem_result.get("Cause"),
                    "Symptom": sem_result.get("Symptom")
                }
            failure_mode = selection.get("Failure_Mode")
            mechanism = selection.get("Mechanism")
            cause = selection.get("Cause")
            symptom = selection.get("Symptom")
            
            # Carry over the highest review flag risk
            final_review_flag = op_review_flag or sem_review_flag
            
            # Carry over the lowest confidence score to ensure strict QA
            conf_map = {"High": 3, "Medium": 2, "Low": 1, None: 0}
            final_confidence = op_confidence if conf_map.get(op_confidence, 0) < conf_map.get(sem_confidence, 0) else sem_confidence
            
            output_dict.update({
                "P4_Failure_Mode": failure_mode,
                "P4_Mechanism": mechanism,
                "P4_Cause": cause,
                "P4_Symptom": symptom,
                "P4_Reasoning": f"{output_dict['P4_Reasoning']} | FMEA: {sem_reasoning}",
                "P4_Confidence": final_confidence,
                "P4_Review_Flag": final_review_flag,
                "P4_Semantic_Confidence": sem_confidence,
                "P4_Semantic_Review_Flag": sem_review_flag
            })

            return output_dict

        return await self._process_in_chunks(df, "P4_Maintenance_Type", process_row, checkpoint_path, phase_name)

    def apply_granular_merges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Phase 4 labels into merged output columns based on configured thresholds.
        """
        merge_columns = [
            self.LEGACY_MAINTENANCE_TYPE_COL,
            self.LEGACY_ACTION_TAKEN_COL,
            self.MERGED_MAINTENANCE_TYPE_COL,
            self.MERGED_ACTION_TAKEN_COL,
            self.MAINTENANCE_TYPE_CORRECTED_FLAG_COL,
            self.ACTION_TAKEN_CORRECTED_FLAG_COL,
            self.MERGED_FAILURE_MODE_COL,
            self.MERGED_MECHANISM_COL,
            self.MERGED_CAUSE_COL,
            self.MERGED_SYMPTOM_COL
        ]
        for col in merge_columns:
            if col not in df.columns:
                df[col] = None
        if self.LEGACY_MAINTENANCE_TYPE_COL in df.columns:
            df[self.LEGACY_MAINTENANCE_TYPE_COL] = df[self.LEGACY_MAINTENANCE_TYPE_COL].where(
                df[self.LEGACY_MAINTENANCE_TYPE_COL].notna(),
                df.get(self.config.columns.maintenance_type_col)
            )
        if self.LEGACY_ACTION_TAKEN_COL in df.columns:
            df[self.LEGACY_ACTION_TAKEN_COL] = df[self.LEGACY_ACTION_TAKEN_COL].where(
                df[self.LEGACY_ACTION_TAKEN_COL].notna(),
                df.get(self.config.columns.action_type_col)
            )

        maintenance_thresholds = self.config.rules.maintenance_type_thresholds
        action_thresholds = self.config.rules.action_taken_thresholds
        failure_thresholds = self.config.rules.failure_mode_thresholds
        enforce_failure_thresholds = self.config.rules.enforce_failure_mode_thresholds

        def normalize_label(value: Any) -> Optional[str]:
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(value, str):
                cleaned = value.strip()
                return cleaned if cleaned else None
            return str(value).strip()

        for idx, row in df.iterrows():
            legacy_maint = row.get(self.config.columns.maintenance_type_col)
            legacy_action = row.get(self.config.columns.action_type_col)
            op_conf = row.get("P4_Operational_Confidence")
            op_review = self._coerce_review_flag(row.get("P4_Operational_Review_Flag", False))
            sem_conf = row.get("P4_Semantic_Confidence")
            sem_review = self._coerce_review_flag(row.get("P4_Semantic_Review_Flag", False))

            if self._passes_threshold(op_conf, op_review, maintenance_thresholds):
                merged_maint = row.get("P4_Maintenance_Type")
            else:
                merged_maint = legacy_maint
            df.at[idx, self.MERGED_MAINTENANCE_TYPE_COL] = merged_maint
            df.at[idx, self.MAINTENANCE_TYPE_CORRECTED_FLAG_COL] = bool(
                normalize_label(legacy_maint) != normalize_label(merged_maint)
                and normalize_label(merged_maint) is not None
            )

            if self._passes_threshold(op_conf, op_review, action_thresholds):
                merged_action = row.get("P4_Action_Taken")
            else:
                merged_action = legacy_action
            df.at[idx, self.MERGED_ACTION_TAKEN_COL] = merged_action
            df.at[idx, self.ACTION_TAKEN_CORRECTED_FLAG_COL] = bool(
                normalize_label(legacy_action) != normalize_label(merged_action)
                and normalize_label(merged_action) is not None
            )

            if enforce_failure_thresholds:
                if self._passes_threshold(sem_conf, sem_review, failure_thresholds):
                    df.at[idx, self.MERGED_FAILURE_MODE_COL] = row.get("P4_Failure_Mode")
                    df.at[idx, self.MERGED_MECHANISM_COL] = row.get("P4_Mechanism")
                    df.at[idx, self.MERGED_CAUSE_COL] = row.get("P4_Cause")
                    df.at[idx, self.MERGED_SYMPTOM_COL] = row.get("P4_Symptom")
                else:
                    df.at[idx, self.MERGED_FAILURE_MODE_COL] = None
                    df.at[idx, self.MERGED_MECHANISM_COL] = None
                    df.at[idx, self.MERGED_CAUSE_COL] = None
                    df.at[idx, self.MERGED_SYMPTOM_COL] = None
            else:
                df.at[idx, self.MERGED_FAILURE_MODE_COL] = row.get("P4_Failure_Mode")
                df.at[idx, self.MERGED_MECHANISM_COL] = row.get("P4_Mechanism")
                df.at[idx, self.MERGED_CAUSE_COL] = row.get("P4_Cause")
                df.at[idx, self.MERGED_SYMPTOM_COL] = row.get("P4_Symptom")

        return df

    # =========================================================================
    # POST-PROCESSING QA FILTERS
    # =========================================================================

    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Task 4: Evaluate final outputs against configuration thresholds,
        flagging any row that fails the confidence or human review checks.
        """
        logger.info("Applying post-processing Quality Assurance filters...")
        
        req_confidence = self.config.rules.required_confidence_level
        allow_review = self.config.rules.allow_human_review_flags
        
        # Simple integer map to evaluate confidence strictly
        conf_map = {"High": 3, "Medium": 2, "Low": 1, None: 0}
        target_score = conf_map.get(req_confidence, 3)

        def evaluate_row(row: pd.Series) -> str:
            conf_score = conf_map.get(row.get("P4_Confidence"), 0)
            rev_flag = self._coerce_review_flag(row.get("P4_Review_Flag", False))

            if conf_score < target_score:
                return "Requires Human Review (Low Confidence)"
            if not allow_review and rev_flag:
                return "Requires Human Review (Flagged by Model)"
                
            return "Pass"

        df["QA_Status"] = df.apply(evaluate_row, axis=1)
        
        passed = len(df[df['QA_Status'] == 'Pass'])
        logger.info(f"QA Filtering complete: {passed}/{len(df)} records passed automatically.")
        return df

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================

    async def execute_full_pipeline(self, input_path: str, output_path: str) -> None:
        """Execute all four phases and QA filtering sequentially."""
        logger.info("Initiating full labelling pipeline execution.")
        out_dir = os.path.dirname(output_path)
        
        self.initialize_performance_logger(out_dir)
        
        dict_path = os.path.join(out_dir, "fmea_dictionaries.json")
        
        try:
            raw_df = self.data_manager.load_dataset(input_path)
            df_phase_1 = await self.run_phase_1_system_code_corrections(raw_df, os.path.join(out_dir, "cp_p1.csv"))
            df_phase_2 = await self.run_phase_2_system_code_audit(df_phase_1, os.path.join(out_dir, "cp_p2.csv"))
            df_phase_2 = await self.run_phase_2_relabel_failed_audits(
                df_phase_2,
                os.path.join(out_dir, "cp_p2_relabel.csv")
            )
            df_phase_2 = self.apply_system_code_merges(df_phase_2)

            dictionaries = await self.run_phase_3_batch_extraction(df_phase_2, dict_path)
            
            final_df = await self.run_phase_4_granular_labelling(df_phase_2, dictionaries, os.path.join(out_dir, "cp_p4.csv"))
            final_df = self.apply_granular_merges(final_df)

            final_qa_df = self.apply_quality_filters(final_df)
            
            self.data_manager.save_checkpoint(final_qa_df, output_path)
            total_cost = self.api_client.get_total_cost()
            logger.info(f"Pipeline completed successfully. Total API Cost: ${total_cost:.4f}")
            
        except LabellingPipelineError as e:
            logger.error(f"Pipeline execution halted due to an internal error: {e}")
            raise
        except Exception as e:
            logger.critical(f"Pipeline execution failed with an unexpected error: {e}")
            raise

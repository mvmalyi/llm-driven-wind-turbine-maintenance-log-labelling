"""Regression tests for the audited pipeline discrepancies."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config_manager import (
    ColumnConfig,
    DataConfig,
    LLMConfig,
    PipelineRulesConfig,
    ThresholdConfig,
    TaxonomyConfig,
)
from src.data_manager import DataManager
from src.exceptions import ConfigurationError
from src.pipeline_orchestrator import PipelineOrchestrator
from src.taxonomy_handler import TaxonomyHandler


class FakeAsyncClient:
    """Small fake structured-output client for offline orchestration tests."""

    def __init__(self) -> None:
        self.config = LLMConfig()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = []

    async def generate_structured_output(
        self,
        system_prompt: str,
        user_message: str,
        schema: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, int]]:
        self.calls.append((schema["name"], user_message, self.config.reasoning_effort))
        self.total_prompt_tokens += 10
        self.total_completion_tokens += 5

        if schema["name"] == "system_code_classification":
            suggested_code = "=G001 MDA10" if "pitch" in user_message.lower() else "=G001 MCB10"
            return {
                "Suggested_Code": suggested_code,
                "Reasoning": "Synthetic classification.",
                "Confidence_Score": "High",
                "Review_Flag": False,
            }, {"input": 10, "output": 5}

        if schema["name"] == "system_code_audit":
            return {
                "Reasoning": "Synthetic audit.",
                "Confidence_Score": "High",
                "Review_Flag": False,
            }, {"input": 10, "output": 5}

        if schema["name"] == "fmea_dictionary_extraction":
            return {
                "System_Code": "=G001 MCB10",
                "System_Name": "Generator",
                "Topology": "Standard Topology",
                "Total_Logs_Analyzed": 2,
                "Unclassified_Logs_Count": 0,
                "Empirical_Actions_Taken": ["Replace bearing"],
                "Failure_Modes": [
                    {
                        "Failure_Mode": "Bearing fatigue",
                        "Mechanism": "Rolling-contact fatigue",
                        "Cause": "Cyclic loading",
                        "Symptom_Effect": "High vibration",
                        "Frequency_in_Logs": 2,
                    }
                ],
                "Reasoning": "Synthetic extraction.",
                "Confidence_Score": "High",
                "Review_Flag": False,
            }, {"input": 10, "output": 5}

        raise AssertionError(f"Unexpected schema requested: {schema['name']}")

    def get_total_cost(self) -> float:
        return 0.0


def build_taxonomy_handler() -> TaxonomyHandler:
    """Create a compact taxonomy tree without reading a CSV file."""
    handler = TaxonomyHandler("unused.csv")
    handler.taxonomy_tree = {
        "=G001": {
            "_name": "Wind turbine",
            "_full_code": "=G001",
            "_children": {
                "MDA": {
                    "_name": "Pitch System",
                    "_full_code": "=G001 MDA",
                    "_children": {
                        "MDA10": {
                            "_name": "Pitch Battery",
                            "_full_code": "=G001 MDA10",
                            "_children": {},
                        }
                    },
                },
                "MCB": {
                    "_name": "Generator",
                    "_full_code": "=G001 MCB",
                    "_children": {
                        "MCB10": {
                            "_name": "Generator Bearing",
                            "_full_code": "=G001 MCB10",
                            "_children": {},
                        }
                    },
                },
            },
        }
    }
    return handler


def build_orchestrator(config: DataConfig | None = None) -> PipelineOrchestrator:
    """Build an orchestrator with fake dependencies."""
    config = config or DataConfig(
        columns=ColumnConfig(),
        llm=LLMConfig(),
        rules=PipelineRulesConfig(),
        taxonomy=TaxonomyConfig(pitch_system_prefixes=["MDA"]),
    )
    return PipelineOrchestrator(
        config=config,
        data_manager=DataManager(config),
        api_client=FakeAsyncClient(),
        taxonomy_handler=build_taxonomy_handler(),
        chunk_size=10,
    )


class PipelineDiscrepancyTests(unittest.TestCase):
    """Tests for the main discrepancies found in the audit."""

    def test_phase_1_processes_pitch_and_missing_code_rows(self) -> None:
        orchestrator = build_orchestrator()
        df = pd.DataFrame(
            [
                {
                    "WO_ID": 1,
                    "Descriptions": "Replace pitch battery",
                    "System_Code": "=G001 MCB",
                    "System_Name": "Generator",
                },
                {
                    "WO_ID": 2,
                    "Descriptions": "Replace generator bearing",
                    "System_Code": None,
                    "System_Name": "",
                },
                {
                    "WO_ID": 3,
                    "Descriptions": "Routine generator check",
                    "System_Code": "=G001 MCB",
                    "System_Name": "Generator",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = asyncio.run(
                orchestrator.run_phase_1_system_code_corrections(
                    df,
                    str(Path(tmp_dir) / "cp_p1.csv"),
                )
            )

        self.assertEqual(len(orchestrator.api_client.calls), 2)
        self.assertEqual(result.loc[0, "P1_System_Code"], "=G001 MDA10")
        self.assertEqual(result.loc[1, "P1_System_Code"], "=G001 MCB10")
        self.assertTrue(result.loc[2, "P1_Complete"])
        self.assertTrue(pd.isna(result.loc[2, "P1_System_Code"]))
        self.assertTrue(result.loc[0, "P1_Audit_Exempt"])
        self.assertTrue(result.loc[1, "P1_Audit_Exempt"])
        self.assertFalse(result.loc[2, "P1_Audit_Exempt"])

    def test_phase_2_skips_phase_1_targeted_rows(self) -> None:
        orchestrator = build_orchestrator()
        df = pd.DataFrame(
            [
                {
                    "WO_ID": 1,
                    "Descriptions": "Replace pitch battery",
                    "System_Code": "=G001 MCB",
                    "System_Name": "Generator",
                    "P1_Audit_Exempt": True,
                },
                {
                    "WO_ID": 2,
                    "Descriptions": "Routine generator check",
                    "System_Code": "=G001 MCB",
                    "System_Name": "Generator",
                    "P1_Audit_Exempt": False,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = asyncio.run(
                orchestrator.run_phase_2_system_code_audit(
                    df,
                    str(Path(tmp_dir) / "cp_p2.csv"),
                )
            )

        audit_calls = [
            call for call in orchestrator.api_client.calls
            if call[0] == "system_code_audit"
        ]
        self.assertEqual(len(audit_calls), 1)
        self.assertFalse(result.loc[0, "P2_Requires_Relabel"])
        self.assertTrue(result.loc[0, "P2_Audit_Complete"])

    def test_pitch_sensitive_routing_requires_pitch_prefixes(self) -> None:
        config = DataConfig(
            columns=ColumnConfig(),
            taxonomy=TaxonomyConfig(pitch_system_prefixes=[]),
        )
        orchestrator = build_orchestrator(config)
        row = pd.Series({"Descriptions": "Replace pitch battery", "System_Code": "=G001 MCB"})

        with self.assertRaises(ConfigurationError):
            orchestrator._resolve_taxonomy_branch(row, pitch_sensitive=True)

    def test_pitch_detection_handles_configured_spelling_variants(self) -> None:
        orchestrator = build_orchestrator()

        self.assertTrue(orchestrator._has_pitch_mentions("PITCH battery replaced"))
        self.assertTrue(orchestrator._has_pitch_mentions("pich motor alarm"))
        self.assertTrue(orchestrator._has_pitch_mentions("ptch controller reset"))
        self.assertFalse(orchestrator._has_pitch_mentions("generator bearing replaced"))

    def test_csv_loaded_false_completion_flags_are_unprocessed(self) -> None:
        orchestrator = build_orchestrator()

        self.assertTrue(orchestrator._is_unprocessed_value("False"))
        self.assertTrue(orchestrator._is_unprocessed_value("0"))
        self.assertFalse(orchestrator._is_unprocessed_value("True"))
        self.assertFalse(orchestrator._is_unprocessed_value("Corrective"))

    def test_phase_3_rejects_truncation_and_preserves_frequency(self) -> None:
        config = DataConfig(
            columns=ColumnConfig(),
            llm=LLMConfig(),
            rules=PipelineRulesConfig(max_batch_chars=10),
            taxonomy=TaxonomyConfig(pitch_system_prefixes=["MDA"]),
        )
        orchestrator = build_orchestrator(config)

        with self.assertRaises(ConfigurationError):
            orchestrator._build_batched_log_text(["one", "two"], "=G001 MCB")

        config.rules.max_batch_chars = None
        df = pd.DataFrame(
            [
                {
                    "WO_ID": 1,
                    "Descriptions": "Replace generator bearing",
                    "System_Code": "=G001 MCB",
                    "System_Name": "Generator",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dictionaries = asyncio.run(
                orchestrator.run_phase_3_batch_extraction(
                    df,
                    str(Path(tmp_dir) / "fmea.json"),
                )
            )

        failure = dictionaries["=G001 MCB"]["Standard Topology"]["Failure_Modes"][0]
        self.assertEqual(failure["Frequency_in_Logs"], 2)
        self.assertEqual(orchestrator.api_client.calls[0][2], "high")
        self.assertEqual(orchestrator.api_client.config.reasoning_effort, "medium")

    def test_topology_context_is_system_aware(self) -> None:
        orchestrator = build_orchestrator()
        generator_row = pd.Series(
            {
                "Descriptions": "Replace generator bearing",
                "System_Code": "=G001 MCB",
                "System_Name": "Generator",
                "Generator_Type": "DFIG",
                "Pitch_Type": "Electrical",
            }
        )
        pitch_row = pd.Series(
            {
                "Descriptions": "Replace pitch battery",
                "System_Code": "=G001 MDA",
                "System_Name": "Pitch System",
                "Generator_Type": "DFIG",
                "Pitch_Type": "Electrical",
            }
        )
        unrelated_row = pd.Series(
            {
                "Descriptions": "Repair tower door",
                "System_Code": "=G001 UAA",
                "System_Name": "Tower",
                "Generator_Type": "DFIG",
                "Pitch_Type": "Electrical",
            }
        )

        self.assertEqual(
            orchestrator._build_topology_context(generator_row),
            "Generator_Type: DFIG",
        )
        self.assertEqual(
            orchestrator._build_topology_context(pitch_row),
            "Pitch_Type: Electrical",
        )
        self.assertEqual(
            orchestrator._build_topology_context(unrelated_row),
            "Standard Topology",
        )
        self.assertEqual(
            orchestrator._build_topology_prompt_block(unrelated_row),
            "",
        )

    def test_system_code_merge_preserves_legacy_and_handles_nan_review_flags(self) -> None:
        orchestrator = build_orchestrator()
        df = pd.DataFrame(
            [
                {
                    "System_Code": "=G001 MCB",
                    "P1_System_Code": "=G001 MDA10",
                    "P1_Confidence": "High",
                    "P1_Review_Flag": float("nan"),
                },
                {
                    "System_Code": "=G001 MCB",
                    "P2_Relabel_System_Code": "=G001 MDA10",
                    "P2_Relabel_Confidence": "Low",
                    "P2_Relabel_Review_Flag": False,
                },
            ]
        )

        result = orchestrator.apply_system_code_merges(df)

        self.assertEqual(result.loc[0, "Legacy_System_Code"], "=G001 MCB")
        self.assertEqual(result.loc[0, "Merged_System_Code"], "=G001 MDA10")
        self.assertTrue(result.loc[0, "System_Code_Corrected_Flag"])
        self.assertEqual(result.loc[1, "Merged_System_Code"], "=G001 MCB")
        self.assertFalse(result.loc[1, "System_Code_Corrected_Flag"])

    def test_phase_1_targeted_merge_takes_priority_over_stale_audit_relabel(self) -> None:
        config = DataConfig(
            columns=ColumnConfig(),
            rules=PipelineRulesConfig(
                system_code_thresholds=ThresholdConfig(required_confidence_level="High")
            ),
            taxonomy=TaxonomyConfig(pitch_system_prefixes=["MDA"]),
        )
        orchestrator = build_orchestrator(config)
        df = pd.DataFrame(
            [
                {
                    "System_Code": "=G001 MCB",
                    "P1_System_Code": "=G001 MDA10",
                    "P1_Confidence": "High",
                    "P1_Review_Flag": False,
                    "P1_Audit_Exempt": True,
                    "P2_Relabel_System_Code": "=G001 MCB10",
                    "P2_Relabel_Confidence": "High",
                    "P2_Relabel_Review_Flag": False,
                }
            ]
        )

        result = orchestrator.apply_system_code_merges(df)

        self.assertEqual(result.loc[0, "Merged_System_Code"], "=G001 MDA10")

    def test_granular_merge_adds_legacy_columns_and_correction_flags(self) -> None:
        orchestrator = build_orchestrator()
        df = pd.DataFrame(
            [
                {
                    "Maintenance_Type": "Corrective",
                    "Action_Type": "Repair",
                    "P4_Maintenance_Type": "Preventive",
                    "P4_Action_Taken": "Replace bearing",
                    "P4_Failure_Mode": "Not Applicable",
                    "P4_Mechanism": "Not Applicable",
                    "P4_Cause": "Not Applicable",
                    "P4_Symptom": "Not Applicable",
                    "P4_Operational_Confidence": "High",
                    "P4_Operational_Review_Flag": pd.NA,
                    "P4_Semantic_Confidence": "High",
                    "P4_Semantic_Review_Flag": False,
                }
            ]
        )

        result = orchestrator.apply_granular_merges(df)

        self.assertEqual(result.loc[0, "Legacy_Maintenance_Type"], "Corrective")
        self.assertEqual(result.loc[0, "Legacy_Action_Taken"], "Repair")
        self.assertEqual(result.loc[0, "Merged_Maintenance_Type"], "Preventive")
        self.assertEqual(result.loc[0, "Merged_Action_Taken"], "Replace bearing")
        self.assertTrue(result.loc[0, "Maintenance_Type_Corrected_Flag"])
        self.assertTrue(result.loc[0, "Action_Taken_Corrected_Flag"])
        self.assertEqual(result.loc[0, "Merged_Failure_Mode"], "Not Applicable")


if __name__ == "__main__":
    unittest.main()

"""
Configuration management module for the maintenance log labelling pipeline.

This module defines the data structures used to configure dynamic column mapping,
large language model parameters, and deterministic pipeline rules. It ensures the
framework remains dataset-agnostic and easily adjustable for various operational
environments.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ColumnConfig:
    """
    Configuration class for mapping dataset column names.
    
    This allows users to define the names of the columns in their specific dataset
    without modifying the underlying pipeline logic.
    """
    
    # Primary Identifiers and Text
    wo_id_col: str = "WO_ID"
    system_code_col: str = "System_Code"
    system_name_col: str = "System_Name"
    description_col: str = "Descriptions"
    
    # Legacy Categorical Fields
    maintenance_type_col: str = "Maintenance_Type"
    action_type_col: str = "Action_Type"
    
    # Turbine Technical Specifications
    manufacturer_col: str = "Manufacturer"
    turbine_model_col: str = "Turbine_Model"
    drive_type_col: str = "Drive_Type"
    generator_type_col: str = "Generator_Type"
    pitch_type_col: str = "Pitch_Type"
    yaw_brake_type_col: str = "Yaw_Brake_Type"
    
    def get_specification_columns(self) -> List[str]:
        """
        Retrieve a list of all technical specification column names.
        
        Returns:
            List[str]: A list of the mapped specification column strings.
        """
        return [
            self.manufacturer_col,
            self.turbine_model_col,
            self.drive_type_col,
            self.generator_type_col,
            self.pitch_type_col,
            self.yaw_brake_type_col
        ]


@dataclass
class LLMConfig:
    """
    Configuration class for Large Language Model API parameters and pricing.
    """
    
    model: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    phase3_reasoning_effort: Optional[str] = "high"
    max_concurrent_requests: int = 10
    phase3_max_concurrent_requests: Optional[int] = 5
    timeout_seconds: float = 60.0
    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 20.0
    backoff_multiplier: float = 2.0
    jitter_seconds: float = 0.5
    retryable_status_codes: List[int] = field(
        default_factory=lambda: [408, 409, 429, 500, 502, 503, 504]
    )
    
    # Pricing defaults per 1 million tokens (in USD)
    input_cost_per_million: float = 2.50
    output_cost_per_million: float = 15.00
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the financial cost of an API call based on token usage.
        
        Args:
            input_tokens (int): The number of prompt tokens used.
            output_tokens (int): The number of completion tokens generated.
            
        Returns:
            float: The calculated cost in USD.
        """
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_million
        return input_cost + output_cost


@dataclass
class ThresholdConfig:
    """
    Configuration class for confidence/review gating logic.
    """

    required_confidence_level: str = "High"
    allow_human_review_flags: bool = False


@dataclass
class PipelineRulesConfig:
    """
    Configuration class for deterministic pipeline behaviour and validation thresholds.
    """
    
    # Validation Thresholds (global QA)
    required_confidence_level: str = "High"
    allow_human_review_flags: bool = False
    # Merge Thresholds (per task)
    system_code_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    maintenance_type_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    action_taken_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    failure_mode_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    enforce_failure_mode_thresholds: bool = False
    max_row_retries: int = 2
    
    # Deterministic Skip Rules
    skip_preventive_maintenance: bool = True
    skip_missing_descriptions: bool = True

    # Phase 3 prompt sizing controls
    max_batch_descriptions: Optional[int] = None
    max_batch_chars: Optional[int] = None
    
    # Escape Categories
    garbage_data_label: str = "Not Enough Information"
    healthy_turbine_label: str = "Not Applicable"
    action_escape_categories: List[str] = field(
        default_factory=lambda: ["Other", "Not Enough Information"]
    )
    failure_escape_categories: List[str] = field(
        default_factory=lambda: ["Other Novel Failure", "Not Enough Information"]
    )
    
    # Target maintenance types to skip during failure mode extraction
    non_failure_maintenance_types: List[str] = field(
        default_factory=lambda: ["Preventive", "Retrofit"]
    )


@dataclass
class TaxonomyConfig:
    """
    Configuration class for selecting dynamic taxonomy branches.
    """

    facility_prefix: Optional[str] = None
    system_prefix: Optional[str] = None
    pitch_keywords: List[str] = field(default_factory=lambda: ["pitch", "pich", "ptch"])
    pitch_system_prefixes: List[str] = field(default_factory=list)
    yaw_brake_system_prefixes: List[str] = field(default_factory=list)
    generator_system_prefixes: List[str] = field(default_factory=list)
    converter_system_prefixes: List[str] = field(default_factory=list)
    drivetrain_system_prefixes: List[str] = field(default_factory=list)


@dataclass
class RedactionConfig:
    """
    Configuration class for redacting or anonymising sensitive text content.
    """

    enabled: bool = False
    patterns: List[str] = field(default_factory=list)
    replacement: str = "[REDACTED]"
    custom_redactor: Optional[Callable[[str], str]] = None


@dataclass
class DataConfig:
    """
    Master configuration class that unifies all pipeline settings.
    
    This object is intended to be instantiated in the primary Jupyter Notebook
    and passed down through the orchestrator to all constituent modules.
    """
    
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rules: PipelineRulesConfig = field(default_factory=PipelineRulesConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    
    # Global File Paths
    default_input_dir: str = "inputs"
    default_output_dir: str = "outputs"

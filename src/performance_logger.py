"""
Performance and cost logging module.

This module provides the PerformanceLogger class, which tracks input tokens,
output tokens, execution runtime, and financial cost across pipeline phases.
It ensures that all metrics are saved iteratively to a CSV log.
"""

import logging
import os
from typing import Any, Dict, List

import pandas as pd

from src.config_manager import LLMConfig


logger: logging.Logger = logging.getLogger(__name__)


class PerformanceLogger:
    """
    Tracks and logs API token usage, runtime, and financial costs.
    
    Attributes:
        llm_config (LLMConfig): Configuration object for cost calculation.
        log_path (str): The destination path for the CSV log file.
        records (List[Dict[str, Any]]): Internal list of execution records.
    """

    def __init__(self, llm_config: LLMConfig, log_path: str) -> None:
        """
        Initialise the PerformanceLogger.

        Args:
            llm_config (LLMConfig): The configuration containing API pricing.
            log_path (str): Path to save the performance CSV log.
        """
        self.llm_config: LLMConfig = llm_config
        self.log_path: str = log_path
        self.records: List[Dict[str, Any]] = []

        # Attempt to load existing logs to resume tracking seamlessly
        if os.path.exists(log_path):
            try:
                existing_df = pd.read_csv(log_path)
                self.records = existing_df.to_dict('records')
                logger.info(f"Loaded existing performance log with {len(self.records)} records.")
            except Exception as e:
                logger.warning(f"Could not load existing performance log: {e}")

    def log_execution_chunk(
        self, 
        phase_name: str, 
        chunk_index: int, 
        rows_processed: int,
        input_tokens: int, 
        output_tokens: int, 
        runtime_seconds: float
    ) -> None:
        """
        Log the metrics for a single processing chunk and save to disk.

        Args:
            phase_name (str): The identifier for the current pipeline phase.
            chunk_index (int): The sequence number of the processed chunk.
            rows_processed (int): The number of dataset rows processed in this chunk.
            input_tokens (int): Prompt tokens consumed.
            output_tokens (int): Completion tokens generated.
            runtime_seconds (float): Execution time in seconds.
        """
        cost = self.llm_config.calculate_cost(input_tokens, output_tokens)
        
        record = {
            "Phase": phase_name,
            "Chunk_Index": chunk_index,
            "Rows_Processed": rows_processed,
            "Input_Tokens": input_tokens,
            "Output_Tokens": output_tokens,
            "Runtime_Seconds": round(runtime_seconds, 2),
            "Cost_USD": round(cost, 6)
        }
        
        self.records.append(record)
        self._save_log()

    def _save_log(self) -> None:
        """Internally save the accumulated records to the CSV file."""
        try:
            output_dir = os.path.dirname(self.log_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            df = pd.DataFrame(self.records)
            df.to_csv(self.log_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save performance log: {e}")
"""
Data management module for robust dataset ingestion and checkpointing.

This module provides the DataManager class, which handles loading, saving,
validating, and merging datasets, as well as JSON I/O operations. It is entirely 
dataset-agnostic, relying on the injected DataConfig for column mapping and 
raising custom exceptions for graceful error handling.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config_manager import DataConfig
from src.exceptions import DataProcessingError


# Configure basic logging for the module
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger: logging.Logger = logging.getLogger(__name__)


class DataManager:
    """
    A robust I/O manager for datasets, handling validation, checkpointing, and JSON storage.
    
    Attributes:
        config (DataConfig): The master configuration object defining expected columns.
    """

    def __init__(self, config: DataConfig) -> None:
        """
        Initialise the DataManager.

        Args:
            config (DataConfig): Configuration defining dynamic column names.
        """
        self.config: DataConfig = config

    # =========================================================================
    # CSV AND DATAFRAME OPERATIONS
    # =========================================================================

    def load_dataset(
        self, 
        file_path: str, 
        required_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load a CSV dataset into a pandas DataFrame and validate required columns.

        Args:
            file_path (str): The absolute or relative path to the CSV file.
            required_columns (Optional[List[str]]): A list of column names that must
                exist in the loaded dataset.

        Returns:
            pd.DataFrame: The loaded and validated dataframe.

        Raises:
            DataProcessingError: If the file is not found, is empty, or lacks required columns.
        """
        logger.info(f"Attempting to load dataset from {file_path}")
        
        if not os.path.exists(file_path):
            raise DataProcessingError(f"Dataset not found at specified path: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError as e:
            raise DataProcessingError(f"The dataset at {file_path} is completely empty.") from e
        except Exception as e:
            raise DataProcessingError(f"An unexpected error occurred while reading {file_path}: {str(e)}") from e

        if required_columns:
            self._validate_columns(df, required_columns, file_path)
            
        logger.info(f"Successfully loaded {len(df)} records from {file_path}.")
        return df

    def _validate_columns(self, df: pd.DataFrame, expected_columns: List[str], file_name: str) -> None:
        """
        Internal helper to verify the presence of required columns.
        
        Raises:
            DataProcessingError: If any expected columns are missing.
        """
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = (
                f"Dataset '{file_name}' is missing {len(missing_columns)} required columns: "
                f"{', '.join(missing_columns)}"
            )
            logger.error(error_msg)
            raise DataProcessingError(message=error_msg, missing_columns=missing_columns)

    def merge_datasets(
        self, 
        primary_df: pd.DataFrame, 
        secondary_df: pd.DataFrame, 
        join_key: str, 
        how: str = "left"
    ) -> pd.DataFrame:
        """
        Merge two dataframes on a specified key.

        Args:
            primary_df (pd.DataFrame): The base dataframe.
            secondary_df (pd.DataFrame): The enriching dataframe.
            join_key (str): The column name to join on.
            how (str): The type of merge to perform, defaults to 'left'.

        Returns:
            pd.DataFrame: The merged dataframe.

        Raises:
            DataProcessingError: If the join key is missing from either dataset.
        """
        if join_key not in primary_df.columns:
            raise DataProcessingError(f"Join key '{join_key}' is missing from the primary dataset.")
        if join_key not in secondary_df.columns:
            raise DataProcessingError(f"Join key '{join_key}' is missing from the secondary dataset.")

        logger.info(f"Merging datasets on key '{join_key}' using a '{how}' join...")
        try:
            merged_df = primary_df.merge(secondary_df, on=join_key, how=how)
            logger.info("Merge completed successfully.")
            return merged_df
        except Exception as e:
            raise DataProcessingError(f"Failed to merge datasets: {str(e)}") from e

    def load_checkpoint(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Safely attempt to load a progress checkpoint file.

        Args:
            file_path (str): The path to the checkpoint CSV.

        Returns:
            Optional[pd.DataFrame]: The loaded checkpoint dataframe, or None if it does not exist.
        """
        if os.path.exists(file_path):
            logger.info(f"Checkpoint found at {file_path}. Resuming progress...")
            return self.load_dataset(file_path)
        
        logger.info(f"No checkpoint found at {file_path}. Starting fresh.")
        return None

    def save_checkpoint(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Save the current state of the dataset to disk.

        Args:
            df (pd.DataFrame): The dataframe to save.
            file_path (str): The destination path for the CSV file.
            
        Raises:
            DataProcessingError: If the system cannot write to the designated path.
        """
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            df.to_csv(file_path, index=False)
            logger.debug(f"State saved successfully to {file_path}")
        except Exception as e:
            raise DataProcessingError(f"Failed to save checkpoint to {file_path}: {str(e)}") from e

    # =========================================================================
    # JSON OPERATIONS
    # =========================================================================

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load a JSON file from the specified path.
        
        Args:
            file_path (str): The path to the JSON file.
            
        Returns:
            Dict[str, Any]: The parsed JSON data.
            
        Raises:
            DataProcessingError: If the file is not found or contains invalid JSON.
        """
        logger.info(f"Attempting to load JSON from {file_path}")
        
        if not os.path.exists(file_path):
            raise DataProcessingError(f"JSON file not found at specified path: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded JSON from {file_path}")
            return data
        except json.JSONDecodeError as e:
            raise DataProcessingError(f"Failed to parse JSON file at {file_path}: {str(e)}") from e
        except Exception as e:
            raise DataProcessingError(f"An unexpected error occurred while reading JSON from {file_path}: {str(e)}") from e

    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save a dictionary to a JSON file.
        
        Args:
            data (Dict[str, Any]): The data to save.
            file_path (str): The destination path for the JSON file.
            
        Raises:
            DataProcessingError: If the system cannot write to the designated path.
        """
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.debug(f"JSON data saved successfully to {file_path}")
        except Exception as e:
            raise DataProcessingError(f"Failed to save JSON data to {file_path}: {str(e)}") from e

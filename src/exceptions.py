"""
Custom exceptions module for the maintenance log labelling pipeline.

This module defines a hierarchy of custom exception classes to handle errors
gracefully across configuration, data ingestion, taxonomy parsing, and
OpenAI API interactions.
"""

from typing import Optional


class LabellingPipelineError(Exception):
    """
    Base exception class for all errors within the labelling pipeline.
    
    All custom exceptions in this framework inherit from this class,
    allowing users to catch any pipeline-specific error easily.
    """
    pass


class ConfigurationError(LabellingPipelineError):
    """
    Exception raised for errors in the pipeline configuration.
    
    This includes missing environment variables, invalid thresholds,
    or incorrectly mapped column names in the DataConfig.
    """
    pass


class DataProcessingError(LabellingPipelineError):
    """
    Exception raised for errors during data ingestion, merging, or validation.
    
    This includes file not found errors, empty datasets, missing required
    columns, or pandas-related failures.
    """
    
    def __init__(self, message: str, missing_columns: Optional[list[str]] = None) -> None:
        """
        Initialise the DataProcessingError.
        
        Args:
            message (str): The primary error message.
            missing_columns (Optional[list[str]]): A list of columns that were expected
                but not found in the dataset.
        """
        super().__init__(message)
        self.missing_columns = missing_columns or []


class TaxonomyError(LabellingPipelineError):
    """
    Exception raised for errors encountered while parsing or querying taxonomies.
    
    This includes malformed CSV structures, missing fundamental codes,
    or failure to build the hierarchical tree.
    """
    pass


class LLMAPIError(LabellingPipelineError):
    """
    Exception raised for failures during interactions with the Large Language Model API.
    
    This encompasses network timeouts, authentication failures, rate limit
    exceedances, and strict JSON schema rejections.
    """
    
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        """
        Initialise the LLMAPIError.
        
        Args:
            message (str): The explicit error message explaining the API failure.
            status_code (Optional[int]): The HTTP status code returned by the API provider, if available.
        """
        super().__init__(message)
        self.status_code = status_code
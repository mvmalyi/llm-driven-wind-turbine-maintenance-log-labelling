"""
Taxonomy handling module for parsing and managing component codes.

This module provides the TaxonomyHandler class, which loads the taxonomy
from a CSV file and builds a hierarchical tree structure to facilitate
dynamic prompt injection for large language models.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.exceptions import TaxonomyError


# Configure basic logging for the module
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
logger: logging.Logger = logging.getLogger(__name__)


class TaxonomyHandler:
    """
    A class to load, parse, and manage the hierarchical component codes taxonomy.

    Attributes:
        taxonomy_path (str): The file path to the taxonomy CSV.
        raw_taxonomy_df (Optional[pd.DataFrame]): The raw loaded dataframe.
        taxonomy_tree (Dict[str, Any]): The hierarchical dictionary of the taxonomy.
        code_to_name_map (Dict[str, str]): A flat dictionary mapping exact codes to names.
    """

    def __init__(self, taxonomy_path: str) -> None:
        """
        Initialise the TaxonomyHandler with the file path.

        Args:
            taxonomy_path (str): Path to the taxonomy CSV dataset.
        """
        self.taxonomy_path: str = taxonomy_path
        self.raw_taxonomy_df: Optional[pd.DataFrame] = None
        self.taxonomy_tree: Dict[str, Any] = {}
        self.code_to_name_map: Dict[str, str] = {}

    def load_and_parse_taxonomy(self) -> None:
        """
        Load the taxonomy from the CSV and construct the hierarchical tree.

        Raises:
            TaxonomyError: If the file is missing, empty, or lacks required columns.
        """
        logger.info(f"Loading taxonomy from {self.taxonomy_path}")
        try:
            self.raw_taxonomy_df = pd.read_csv(self.taxonomy_path)
        except Exception as e:
            raise TaxonomyError(
                f"Failed to read the taxonomy file at {self.taxonomy_path}: {str(e)}"
            ) from e

        if self.raw_taxonomy_df.empty:
            raise TaxonomyError(f"The taxonomy file at {self.taxonomy_path} is empty.")

        if 'Code' not in self.raw_taxonomy_df.columns or 'Name' not in self.raw_taxonomy_df.columns:
            raise TaxonomyError("The taxonomy CSV must contain both 'Code' and 'Name' columns.")

        logger.info(f"Successfully loaded {len(self.raw_taxonomy_df)} taxonomy records.")
        self._build_tree()

    def _build_tree(self) -> None:
        """
        Internal method to build a nested dictionary tree from the flat codes.

        The codes are split by spaces to determine their hierarchy level.
        For example, '=G001 MDX10 GP001' becomes nested keys.
        """
        logger.info("Building hierarchical taxonomy tree...")

        # Safety check to ensure we have data
        if self.raw_taxonomy_df is None:
            raise TaxonomyError("Cannot build tree, raw taxonomy dataframe is not loaded.")

        for _, row in self.raw_taxonomy_df.iterrows():
            # Clean up any potential leading or trailing whitespace
            raw_code = str(row['Code']).strip()
            name = str(row['Name']).strip()

            # Populate the flat mapping for quick lookups
            self.code_to_name_map[raw_code] = name

            # Split the code into hierarchical parts
            parts = raw_code.split(' ')

            # Traverse and build the nested dictionary
            current_level = self.taxonomy_tree
            for i, part in enumerate(parts):
                if part not in current_level:
                    current_level[part] = {
                        "_name": name if i == len(parts) - 1 else "",
                        "_full_code": " ".join(parts[:i+1]),
                        "_children": {}
                    }
                elif i == len(parts) - 1:
                    # Update the name if we reached the leaf node of this specific code
                    current_level[part]["_name"] = name

                # Move deeper into the tree
                current_level = current_level[part]["_children"]

        logger.info("Taxonomy tree built successfully.")

    def get_system_branch(self, facility_prefix: str, system_prefix: str) -> Dict[str, Any]:
        """
        Retrieve a specific branch of the taxonomy tree for dynamic prompt injection.

        Args:
            facility_prefix (str): The top-level prefix (e.g., '=G001').
            system_prefix (str): The system-level prefix (e.g., 'MDA' for Rotor or Pitch).

        Returns:
            Dict[str, Any]: The subset of the taxonomy tree. Returns an empty dict if not found.
        """
        if not self.taxonomy_tree:
            logger.warning("Taxonomy tree is empty. Ensure load_and_parse_taxonomy() was called.")
            return {}
            
        # 1. Access the specific facility branch and its children
        facility_children = self.taxonomy_tree.get(facility_prefix, {}).get("_children", {})
        
        # 2. Extract the specific system-level node
        system_node = facility_children.get(system_prefix)
        
        # 3. Return it nested under its prefix to maintain the expected dictionary shape, or an empty dict if not found
        if system_node:
            return {system_prefix: system_node}
            
        return {}

    def get_facility_branch(self, facility_prefix: str) -> Dict[str, Any]:
        """
        Retrieve the direct system-level children for a facility prefix.

        Args:
            facility_prefix (str): The top-level prefix (e.g., '=G001').

        Returns:
            Dict[str, Any]: The system-level taxonomy branch for the facility.
        """
        if not self.taxonomy_tree:
            logger.warning("Taxonomy tree is empty. Ensure load_and_parse_taxonomy() was called.")
            return {}
        return self.taxonomy_tree.get(facility_prefix, {}).get("_children", {})

    def filter_system_branch(
        self,
        branch: Dict[str, Any],
        include_prefixes: Optional[List[str]] = None,
        exclude_prefixes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Filter a system-level taxonomy branch by include/exclude prefixes.

        Args:
            branch (Dict[str, Any]): The system-level branch to filter.
            include_prefixes (Optional[List[str]]): System prefixes to include.
            exclude_prefixes (Optional[List[str]]): System prefixes to exclude.

        Returns:
            Dict[str, Any]: Filtered system-level branch.
        """
        include = [p.strip() for p in (include_prefixes or []) if str(p).strip()]
        exclude = [p.strip() for p in (exclude_prefixes or []) if str(p).strip()]
        if not include and not exclude:
            return branch
        filtered: Dict[str, Any] = {}
        for key, value in branch.items():
            if include and not any(key.startswith(prefix) for prefix in include):
                continue
            if exclude and any(key.startswith(prefix) for prefix in exclude):
                continue
            filtered[key] = value
        return filtered

    def filter_taxonomy_tree(
        self,
        include_prefixes: Optional[List[str]] = None,
        exclude_prefixes: Optional[List[str]] = None,
        facility_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Filter the full taxonomy tree by include/exclude prefixes.

        Args:
            include_prefixes (Optional[List[str]]): System prefixes to include.
            exclude_prefixes (Optional[List[str]]): System prefixes to exclude.
            facility_prefix (Optional[str]): Restrict filtering to a single facility.

        Returns:
            Dict[str, Any]: Filtered taxonomy tree or branch.
        """
        if not self.taxonomy_tree:
            logger.warning("Taxonomy tree is empty. Ensure load_and_parse_taxonomy() was called.")
            return {}
        if facility_prefix:
            branch = self.get_facility_branch(facility_prefix)
            return self.filter_system_branch(branch, include_prefixes, exclude_prefixes)

        filtered_tree: Dict[str, Any] = {}
        for facility_key, facility_node in self.taxonomy_tree.items():
            children = self.filter_system_branch(
                facility_node.get("_children", {}),
                include_prefixes,
                exclude_prefixes
            )
            if not children:
                continue
            filtered_tree[facility_key] = {
                "_name": facility_node.get("_name", ""),
                "_full_code": facility_node.get("_full_code", facility_key),
                "_children": children
            }
        return filtered_tree

    def format_branch_for_prompt(self, branch: Dict[str, Any], indent_level: int = 0) -> str:
        """
        Recursively format a taxonomy branch into a readable string for LLM prompts.

        Args:
            branch (Dict[str, Any]): The taxonomy tree subset.
            indent_level (int): The current indentation level for formatting.

        Returns:
            str: A formatted text representation of the taxonomy.
        """
        formatted_text = ""
        indent = "  " * indent_level

        for key, node in branch.items():
            name = node.get("_name", "Unknown")
            full_code = node.get("_full_code", key)

            # Only add lines that have a valid name attached
            if name:
                formatted_text += f"{indent}- {full_code}: {name}\n"

            children = node.get("_children", {})
            if children:
                formatted_text += self.format_branch_for_prompt(children, indent_level + 1)

        return formatted_text

    def get_full_flat_mapping(self) -> Dict[str, str]:
        """
        Retrieve the flat dictionary mapping of all codes to their names.

        Returns:
            Dict[str, str]: A dictionary in the format {code: name}.
        """
        return self.code_to_name_map

## LLM-Driven Wind Turbine Maintenance Log Labelling Framework: <br> Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2605.31281-b31b1b.svg)](https://arxiv.org/abs/2605.31281) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20670957.svg)](https://doi.org/10.5281/zenodo.20670957)
 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mvmalyi/) [![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-1503-9798) [![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Max-Malyi) [![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=FgcRBeUAAAAJ)

A robust, model-agnostic pipeline leveraging Large Language Models (LLMs) to automatically standardise, categorise, and extract reliability intelligence from unstructured wind turbine maintenance work orders. 

This repository contains the underlying Python architecture, dynamic prompts, and data schemas for the following paper. If you utilise this framework in your academic or commercial research, please cite our work accordingly:
> M. Malyi, J. Shek, A. McDonald, and A. Biscaya, 2026. Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence. Preprint. arXiv:2605.31281

**BibTeX:**
```bibtex
@article{malyi2026wind,
  title={Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Evidence-Based Failure Modes},
  author={Malyi, Max and Shek, Jonathan and McDonald, Alasdair and Biscaya, Andr{\'e}},
  journal={arXiv preprint arXiv:2605.31281},
  year={2026}
}
```

---

### About The Project

This framework addresses the critical bottleneck of unstructured historical logs in Computerised Maintenance Management Systems (CMMS). By utilising dynamic Chain-of-Thought prompting and strict JSON schema validation, it autonomously corrects hierarchical system codes (e.g., RDS-PP) and extracts empirical Failure Modes and Effects Analysis (FMEA) dictionaries directly from free-text descriptions.

---

### Key Features

* **Multi-Phase Execution:** Seamlessly transitions from system code auditing to semantic failure mode extraction and granular log-by-log labelling.
* **Robust Idempotency:** Implements asynchronous chunking and row-by-row checkpointing, ensuring no data or API expenditure is lost during execution interruptions.
* **Deterministic Cost Savings:** Applies adjustable programmatic rules (e.g., skipping preventive maintenance or uninformative logs) to minimise unnecessary LLM API calls.
* **Dataset Agnostic:** Features a dynamic `DataConfig` manager, allowing users to map their proprietary column names and technical specifications without altering the core Python logic.
* **Strict Quality Assurance:** Enforces uniform confidence scoring, human review flagging, and reasoning extraction across all generated labels.

---

### Repository Structure

```
llm-driven-wind-turbine-maintenance-log-labelling/
├── prompts/
│   ├── __init__.py
│   ├── schemas.py                  # Unified JSON schemas enforcing rigorous QA metrics
│   └── templates.py                # Decoupled string templates for natural language prompts
├── src/
│   ├── __init__.py
│   ├── config_manager.py           # Handles dynamic column mapping and LLM parameters
│   ├── data_manager.py             # Robust I/O handler for CSV datasets and JSON dictionaries
│   ├── exceptions.py               # Custom hierarchy of pipeline execution errors
│   ├── performance_logger.py       # Tracks API token usage, runtime, and financial cost
│   ├── pipeline_orchestrator.py    # The asynchronous execution engine
│   ├── taxonomy_handler.py         # Parses and formats hierarchical component trees (RDS-PP)
│   └── llm_clients/
│       ├── __init__.py
│       └── openai_client.py        # Schema-agnostic asynchronous API wrapper
├── orchestration_template.ipynb    # Unified Jupyter Notebook for end-user execution
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

```

---

### Required Datasets

To execute this pipeline, users must prepare two distinct datasets in CSV format. Examples of the required columns are mapped in the `DataConfig` class.

**1. Pre-Processed Maintenance Logs (`maintenance_logs.csv`)**
The primary operational dataset extracted from your CMMS, normally requires data cleaning. It must contain, at minimum:

* `WO_ID`: A unique identifier for the work order.
* `Descriptions`: The unstructured free-text entry from the technician.
* Existing `System_Code`, `System_Name`, `Maintenance_Type`.
* Any other sources of information you see relevant, including turbine topology specifications (e.g., `Pitch_Type`, `Generator_Type`) for dynamic context injection.

**2. Taxonomy Dictionary (`rds_pp_components.csv`)**
A static, hierarchical dictionary of system components. It must contain:

* `Code`: The hierarchical identifier (e.g., `=G001 MDX10`).
* `Name`: The human-readable name of the component.

---

### Prerequisites

* Python 3.13 was used in this project.
* An active OpenAI API key.

---

### Installation

1. Clone the repository:

```bash
git clone https://github.com/mvmalyi/llm-driven-wind-turbine-maintenance-log-labelling.git
cd wind-turbine-log-labelling

```

2. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

3. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"

```

---

### Usage

The entire workflow is orchestrated through a single Jupyter Notebook.

1. Launch Jupyter Notebook (`orchestration_template.ipynb`)

2. In **Cell 1**, adjust the file paths to point to your prepared datasets and update the `ColumnConfig` to match your exact CSV headers. For finer adjustments, please refer to the scripts mentioned in the file tree.

3. Execute the cells sequentially to validate your data, run the four-phase asynchronous pipeline, and generate the final QA-filtered dataset alongside a detailed cost performance log.

---

### Output Columns and Merge Behaviour

The pipeline appends phase-prefixed outputs rather than overwriting legacy columns. Key additions include:

* **`P1_ / P2_ columns`:** System code correction and audit outputs (suggested codes, confidence, review flags).
* **`Legacy_System_Code / Merged_System_Code / System_Code_Corrected_Flag`:** Preserved original system code, threshold-gated system code merge result, and correction indicator.
* **`P4_ columns`:** Granular maintenance/action/failure labels, including operational/semantic confidence flags.
* **`Legacy_Maintenance_Type / Legacy_Action_Taken`:** Preserved original granular labels.
* **`Merged_Maintenance_Type / Merged_Action_Taken / Merged_Failure_Mode (+ Mechanism/Cause/Symptom)`:** Corrected merge outputs for downstream analysis.
* **`Maintenance_Type_Corrected_Flag / Action_Taken_Corrected_Flag`:** Boolean indicators marking threshold-approved granular label changes.

Pitch keyword detection and isolated pitch system prefixes are configurable via `TaxonomyConfig`. The `pitch_system_prefixes` setting is required when pitch-sensitive correction is active, because it drives both the pitch-only branch and the restricted non-pitch branch. Per-task confidence thresholds are configured in `PipelineRulesConfig`; failure-mode threshold enforcement exists but is disabled by default via `enforce_failure_mode_thresholds=False`.

---

### Contributing

If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. Contributions to expand the framework to other LLM providers or industrial applications are welcome.

---

### License

Distributed under the MIT License. See `LICENSE` for more information.

---

### Contact

Max Malyi - Max.Malyi@ed.ac.uk

Project Link: [LLM-Driven Wind Turbine Maintenance Log Labelling Framework](https://github.com/mvmalyi/llm-driven-wind-turbine-maintenance-log-labelling)

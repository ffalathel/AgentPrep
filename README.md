# AgentPrep

**AgentPrep** is an intelligent machine learning preprocessing pipeline tool that combines AI agents with deterministic validation to automate data preparation tasks. It provides a guided, interactive experience for cleaning data, engineering features, and ensuring data quality and governance compliance.

## Features

- **AI-Powered Agents**: Leverages LLM agents (OpenAI, Anthropic, Gemini) for intelligent data quality improvements and feature engineering suggestions
- **Deterministic Validation**: All agent proposals are validated by deterministic code before anything is applied; given the same proposals, output is byte-identical
- **Interactive CLI**: User-friendly wizard interface - no configuration files required
- **Constraint Advisor**: Heuristic-based suggestions for pipeline constraints based on your dataset
- **Governance & Policy**: Built-in policy enforcement and data-driven leakage detection on every column (see [Benchmark](#benchmark))
- **Reusable Pipeline**: Every run saves the fitted pipeline; `agentprep apply` transforms test or production data with the training statistics (see [Using the pipeline on new data](#using-the-pipeline-on-new-data))
- **Artifact Management**: Comprehensive artifact tracking, storage, and reporting
- **Metadata Tracking**: Full provenance and metadata generation for auditability

## Architecture

AgentPrep follows a multi-level pipeline architecture:

- **Level 0: Intent Validation** - Validates user configuration and constraints
- **Level 1: Data Ingestion & Schema Normalization** - Loads datasets, infers schemas, normalizes columns
- **Level 2: Data Quality Agent** - Profiles data quality and applies cleaning actions
- **Level 3: Metadata & Profiling Persistence** - Generates and stores comprehensive metadata
- **Level 4: Feature Engineering Agent** - Proposes and generates ML features
- **Level 5: Governance & Policy** - Enforces policies and detects data leakage
- **Level 6: Artifacts, Storage & Reporting** - Captures and exports pipeline artifacts

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

## Installation & Usage

AgentPrep supports two installation methods: **Standard Installation** (for users) and **Development Installation** (for contributors).

### Method 1: Standard Installation (Recommended for Users)

Install directly from PyPI. This is the easiest way to use the tool.

**1. Create a project directory (optional but recommended):**
```bash
mkdir my_ml_project
cd my_ml_project
```

**2. Create and activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install AgentPrep:**
```bash
pip install agentprep
```

**4. Run the tool:**
```bash
# 1. (Optional) Set API Key for AI features
export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY / GEMINI_API_KEY

# 2. Start the interactive wizard
agentprep run
```

---

### Method 2: Development Installation (For Contributors)

Use this method if you want to modify the source code or contribute.

**1. Clone the repository:**
```bash
git clone https://github.com/ffalathel/AgentPrep.git
cd AgentPrep
```

**2. Run the automated setup script:**
This script creates a virtual environment and installs the package in editable mode.
```bash
chmod +x setup_prod.sh
./setup_prod.sh
```

**3. Activate the environment:**
```bash
source venv/bin/activate
```

**4. Run the CLI:**
You can run it using the installed command or directly via python:
```bash
# Option A: Installed command (recommended)
agentprep run

# Option B: Run via python module
python -m cli run
```

The interactive wizard will guide you through:
1. **LLM Provider (optional)**: Choose OpenAI, Anthropic, Gemini, or \"None\" (no LLM usage)
2. **Dataset Selection**: Upload your CSV or Parquet file
3. **Task Configuration**: Select task type (classification, regression, time series, clustering)
4. **Target Column**: Choose your target variable from the dataset columns
5. **Model Family**: Select your intended model type (tree-based, linear, neural)
6. **Constraint Suggestions**: Get intelligent suggestions for pipeline constraints
7. **Output Path**: Specify where to save pipeline outputs

### Example Session

```bash
$ python -m cli run

============================================================
Welcome to AgentPrep!
============================================================

This interactive wizard will guide you through configuring your preprocessing pipeline.

Enter path to your dataset (CSV or Parquet): data/my_dataset.csv
✓ Dataset loaded: 10,000 rows, 15 columns

Available columns:
  1. age
  2. income
  3. education
  ...
  
Select target column (1-15): 3

Select task type:
  1. Classification
  2. Regression
  3. Time Series
  4. Clustering
Select (1-4) [1]: 1

...

✓ Intent validated successfully
Starting preprocessing pipeline...
✓ Pipeline completed successfully
```

### Command-Line Options

```bash
# Run with verbose logging
python -m cli run --verbose

# Specify output directory
python -m cli run --output ./results

# Run with config file (legacy mode)
python -m cli run --config intent.yaml

# Apply a fitted pipeline to new data
agentprep apply --pipeline results/artifacts/<run_id>/pipeline_<run_id>.json \
  --input new_data.csv --output new_data_prepped.csv
```

### Using the pipeline on new data

Run AgentPrep on your **training data only**. Each run saves `pipeline_<run_id>.json` next to the dataset artifact: every cleaning action and engineered feature, with the values it learned (imputation fills, outlier bounds, scaling statistics, bin edges, category lists), plus the columns governance removed. Apply it to your test set and to production data:

```bash
agentprep apply --pipeline pipeline_<run_id>.json --input test.csv --output test_prepped.parquet
```

New data is transformed with the training statistics, never its own, so nothing about the test set leaks into preprocessing. Unseen categories, values outside the training range, and a missing target column are handled. Applying the pipeline to the training file reproduces the run's output exactly. The pipeline is plain JSON, so it is also an audit record of what was done.

## Configuration

### Environment Variables

Set API keys for LLM providers (optional - agents work without them in stub mode). At runtime, the CLI will ask which provider you want to use (or \"None\").

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"
```

**Note**: If no API keys are provided, agents will run in stub mode (no LLM proposals), but the pipeline will still execute deterministic operations. If a key is set but an LLM call fails, the pipeline continues deterministically and records the failure as a warning in the governance report.

Default models are `claude-sonnet-5-5` (Anthropic), `gpt-4o` (OpenAI), and `gemini-2.5-flash-lite` (Gemini). Override with:

```bash
export AGENTPREP_MODEL="your-model-id"
```

### Intent Schema

The pipeline accepts configuration through an `IntentSchema` that includes:

- **Dataset**: Path to your dataset file
- **Task**: Task type and target column
- **Model**: Model family and interpretability requirements
- **Constraints**: Limits on features, interactions, and cardinality
- **Policies**: Outlier handling and other data policies

## Pipeline Levels

### Level 1: Data Ingestion & Schema Normalization

- Loads datasets from CSV or Parquet files
- Normalizes column names and data types (numbers stored as text because of blank cells become numeric)
- Infers schema metadata on the normalized names (data types, nullability, distributions; date strings are detected as datetimes)
- Validates dataset against intent constraints

### Level 2: Data Quality Agent

- Profiles dataset quality (missing values, outliers, duplicate rows, likely placeholder values such as 0, -1, 999 or "?")
- LLM agent proposes data cleaning actions, and nominates columns it suspects are recorded after the outcome
- Deterministic executor validates and applies safe actions (e.g. it rejects imputing a column that is more than 50% missing, which would erase the missingness signal)
- Replaces placeholder values with missing when the LLM judges they mean "not recorded" (e.g. 0 for blood pressure). Code checks that the value exists in the column and, for numbers, is the column's lowest or highest value. Placeholders that mean something real (-1 = never contacted, 0 purchases) are kept
- Tracks applied vs rejected actions

### Level 3: Metadata & Profiling Persistence

- Builds comprehensive pipeline metadata
- Records schema, quality profiles, and applied actions
- Writes metadata to disk for traceability

### Level 4: Feature Engineering Agent

- LLM agent proposes feature transformations
- Validates features for safety and compliance
- Generates features deterministically (missing-value indicators read the data before imputation)
- Tracks feature provenance

### Level 5: Governance & Policy

- Enforces policy rules (feature limits, cardinality, column dropping, data quality)
- Detects data leakage deterministically, on the data before and after cleaning, and drops leaking columns and any features derived from them:
  - engineered features that use the target as input
  - any column that predicts the target almost perfectly on its own (held-out)
  - any column whose missingness determines the target (e.g. a field only filled in for one outcome)
  - two columns that predict the target together but not alone (e.g. a status date that falls before the arrival date only for cancellations; dates split into year/month/day columns are combined). The data can't say which column is at fault: if the LLM also suspected one of them, that one is dropped; otherwise both are named in a warning
- Surfaces LLM-suspected leaks (columns not known at prediction time) as warnings with a measured score; the LLM alone never removes a column
- Warns about class imbalance but does not resample (resampling before the train/test split leaks rows into evaluation)
- Provides governance decisions

### Level 6: Artifacts, Storage & Reporting

- Captures all pipeline artifacts (datasets, schemas, features, metadata)
- Stores artifacts in organized directory structure
- Writes the dataset as CSV and Parquet, the fitted pipeline and metadata as JSON, and reports as Markdown
- Generates human-readable reports

## Benchmark

`bench/run_bench.py` runs the full pipeline on ten public Kaggle datasets with a real LLM, once for a tree model and once for a linear model. Five of them (Titanic, Bank Marketing, Adult with a planted target copy, Telco Churn, Boston Housing) were used to calibrate the leak thresholds. The other five (Hotel Booking Demand, IBM HR Attrition, Pima Diabetes, Stroke Prediction, California Housing) were not, so they test whether the fixes generalize. AgentPrep fits on an 80% split; its saved pipeline transforms the untouched 20% test split, and models are scored on it. Latest results: [bench/RESULTS.md](bench/RESULTS.md).

```bash
pip install scikit-learn
export ANTHROPIC_API_KEY=...   # or put it in .env
python bench/run_bench.py
```

Summary (claude-sonnet-5-5, 20 runs, each in its own process, all exit 0):
- Documented leaks: 5 of 6 removed in every run. Titanic `boat` and `body` (by the leak detector, or by the LLM's own drop action), the planted target copy, and Hotel `reservation_status` and `reservation_status_date`. The last one is caught by the pair check: its gap to the arrival date predicts cancellation. Bank `duration` (known only after the call) can't be detected from data, alone or in a pair, so it is surfaced as a warning and kept. False leak flags: 0. On Hotel the LLM also warns about 2–3 legitimate columns (e.g. `assigned_room_type`); these are warnings only.
- Placeholders: the LLM marked Pima's zero readings (`glucose`, `blood_pressure`, `skin_thickness`, `insulin`, `bmi`) and Adult's `"?"` as missing, and left real values alone (Pima `pregnancies = 0`, Adult `capital_gain = 0`, Bank `pdays = -1`). Debatable calls it made: Stroke `smoking_status = "Unknown"`, Titanic `fare = 0`.
- Deterministic code rejected 27 of 656 LLM proposals: 5 imputations of mostly-missing columns (Titanic `cabin`, `boat`, `body`; Pima `insulin` after its zeros became missing) and 22 features referencing columns that don't exist.
- Reusable pipeline: replaying it on the training file reproduced the output exactly in all 20 runs.
- Held-out score vs. the same data with standard sklearn prep (known leaks removed from both): linear models gain on 5 of 10 datasets (Boston R² +0.098, California R² +0.029, Stroke AUC +0.017, Bank and Telco +0.006) and are within −0.011 on the rest. Gradient boosting ranges from −0.023 (Titanic) to +0.022 (Stroke). Test sets for Titanic (262 rows), Boston (102), Pima (154) and HR (294) are small, so differences there of ±0.02 are within noise; LLM proposals also vary between runs: on Boston and HR Attrition, where neither new check changed anything, the linear lift moved from +0.120 to +0.098 and from +0.006 to −0.007 between the last two full runs.

Each run uses up to about 1 GB of memory (Hotel). On a machine with 8 GB, run the benchmark with nothing else heavy open.

## Project Structure

```
AgentPrep/
├── cli/                    # CLI modules
│   ├── interactive.py     # Interactive prompts
│   └── constraint_advisor.py  # Constraint suggestions
├── core/                   # Core orchestration
│   ├── orchestrator.py    # Pipeline orchestrator
│   └── recipe.py          # Fitted pipeline: fit, apply, save, load
├── intent/                 # Intent validation
│   ├── schema.py          # Intent schema definitions
│   └── validator.py       # Intent validation logic
├── level1_ingestion/       # Data loading & normalization
├── level2_quality/         # Data quality agent
├── level3_metadata/       # Metadata generation
├── level4_feature/         # Feature engineering agent
├── level5_governance/      # Governance & policies
├── level5_policy/          # Policy enforcement
├── level6_artifacts/       # Artifact management
├── utils/                  # Shared utilities
│   ├── logging.py         # Logging setup
│   ├── constants.py       # Application constants
│   ├── file_helpers.py    # File utilities
│   └── llm_client.py      # LLM client wrapper
└── cli/                    # CLI package (use: python -m cli)
```

## Supported Formats

- **Datasets**: CSV, Parquet
- **Configurations**: YAML, JSON (via interactive mode)
- **Output Formats**: JSON, CSV, Parquet, Markdown

## Exit Codes

- `0`: Success
- `1`: Invalid intent configuration
- `2`: Policy violation detected
- `3`: Runtime error

## Development

### Running Tests

Tests are located in the `tests/` directory. To run tests:

```bash
# Install test dependencies (the "all" extra adds the LLM SDKs for the provider tests)
pip install -e ".[dev,all]"

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

We use `black` for formatting and `ruff` for linting:

```bash
# Install package with development dependencies
pip install -e ".[dev]"

# Format code
black .

# Lint code
ruff check .
```

### Code Structure

- **Modular Design**: Each level is self-contained with clear interfaces
- **Type Safety**: Uses Pydantic for schema validation and type hints throughout
- **Logging**: Centralized logging via `utils.logging`
- **Error Handling**: Comprehensive error handling with custom exception types

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key points:
1. Follow the existing code structure and naming conventions
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting
5. Format code with `black` and lint with `ruff`

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md). **Do not** open public issues for security concerns.

## Support

For issues, questions, or contributions, please [open an issue](https://github.com/ffalathel/AgentPrep/issues) or [create a pull request](https://github.com/ffalathel/AgentPrep/pulls).

---

**AgentPrep** - Intelligent ML Preprocessing with AI Agents

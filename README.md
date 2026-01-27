# AgentPrep

**AgentPrep** is an intelligent machine learning preprocessing pipeline tool that combines AI agents with deterministic validation to automate data preparation tasks. It provides a guided, interactive experience for cleaning data, engineering features, and ensuring data quality and governance compliance.

## Features

- **AI-Powered Agents**: Leverages LLM agents (OpenAI, Anthropic, Gemini) for intelligent data quality improvements and feature engineering suggestions
- **Deterministic Validation**: All agent proposals are validated and executed deterministically for reproducibility
- **Interactive CLI**: User-friendly wizard interface - no configuration files required
- **Constraint Advisor**: Heuristic-based suggestions for pipeline constraints based on your dataset
- **Governance & Policy**: Built-in policy enforcement and data leakage detection
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

- Python 3.8+
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
git clone https://github.com/yourusername/AgentPrep.git
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
```

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

**Note**: If no API keys are provided, agents will run in stub mode (no LLM proposals), but the pipeline will still execute deterministic operations.

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
- Infers schema metadata (data types, nullability, distributions)
- Normalizes column names and data types
- Validates dataset against intent constraints

### Level 2: Data Quality Agent

- Profiles dataset quality (missing values, outliers, duplicates)
- LLM agent proposes data cleaning actions
- Deterministic executor validates and applies safe actions
- Tracks applied vs rejected actions

### Level 3: Metadata & Profiling Persistence

- Builds comprehensive pipeline metadata
- Records schema, quality profiles, and applied actions
- Writes metadata to disk for traceability

### Level 4: Feature Engineering Agent

- LLM agent proposes feature transformations
- Validates features for safety and compliance
- Generates features deterministically
- Tracks feature provenance

### Level 5: Governance & Policy

- Enforces policy rules (constraint violations, data leakage)
- Validates feature engineering proposals
- Detects potential data leakage issues
- Provides governance decisions

### Level 6: Artifacts, Storage & Reporting

- Captures all pipeline artifacts (datasets, schemas, features, metadata)
- Stores artifacts in organized directory structure
- Exports artifacts in multiple formats (JSON, CSV, Parquet, Markdown)
- Generates human-readable reports

## Project Structure

```
AgentPrep/
├── cli/                    # CLI modules
│   ├── interactive.py     # Interactive prompts
│   └── constraint_advisor.py  # Constraint suggestions
├── core/                   # Core orchestration
│   └── orchestrator.py    # Pipeline orchestrator
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
# Install test dependencies
pip install -e ".[dev]"

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
```
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

For issues, questions, or contributions, please [open an issue](link-to-issues) or [create a pull request](link-to-prs).

---

**AgentPrep** - Intelligent ML Preprocessing with AI Agents

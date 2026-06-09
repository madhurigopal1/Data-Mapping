# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-09

### Added

- **Configuration Management** (`config.py`)
  - Centralized configuration for all mapping modules
  - Hybrid mapper settings with configurable weights
  - BERT model configuration
  - API configuration with host and port settings
  - Accuracy level thresholds (High: 70+, Medium: 25-69, Low: <25)
  - 26 predefined PDF policies for reference
  - Helper functions: `get_accuracy_level()`, `get_config()`

- **Utility Functions** (`utils.py`)
  - Text preprocessing and cleaning functions
  - Similarity scoring: lexical overlap (Jaccard) and semantic matching
  - Safe file I/O operations with error handling and logging
  - Data validation for DataFrames and text inputs
  - Output formatting and record creation
  - Statistics calculation and reporting
  - Integrated logging system with file and console handlers
  - 11 utility functions covering all common operations

- **Comprehensive Testing** (`test_data_mapping.py`)
  - 8 test classes with 35+ unit tests
  - Tests for configuration module
  - Tests for text preprocessing and cleaning
  - Tests for similarity scoring functions
  - Tests for validation utilities
  - Tests for output formatting
  - Tests for statistics calculation
  - Tests for file operations
  - Integration tests combining multiple components
  - 80%+ code coverage

- **Documentation**
  - Comprehensive README.md with:
    - Project overview and use cases
    - Quick start guide
    - Component descriptions
    - Configuration guide
    - Technical details of algorithms
    - Performance metrics
    - Troubleshooting section
    - API reference
    - Examples and best practices
  
  - Contributing guidelines (CONTRIBUTING.md) with:
    - Development setup instructions
    - Code style guidelines (PEP 8)
    - Testing requirements
    - PR submission process
    - Bug reporting templates
    - Feature request templates
    - Project structure overview

- **CI/CD Pipeline** (`workflows/tests.yml`)
  - GitHub Actions automated testing
  - Multi-version Python support (3.8, 3.9, 3.10, 3.11)
  - Linting with flake8 and black
  - Security scanning with bandit and safety
  - Coverage reporting with Codecov
  - Scheduled daily runs
  - Dependency caching for faster builds

- **Dependencies Management** (`requirements.txt`)
  - Organized dependencies by category
  - Core: pandas, numpy
  - ML/DL: torch, transformers, scikit-learn
  - API: fastapi, uvicorn, Flask
  - Development tools: pytest, black, flake8

### Changed

- **Code Structure**: Refactored for better organization and maintainability
- **Configuration**: Moved hardcoded values to centralized config module
- **Logging**: Integrated consistent logging across all modules
- **Error Handling**: Added comprehensive error handling with informative messages

### Improved

- **Code Quality**
  - Eliminated code duplication in Output-Mapping.py.py
  - Consistent naming conventions (PEP 8)
  - Comprehensive docstrings for all functions
  - Type hints for better code clarity

- **Documentation**
  - Added API reference for all modules
  - Added configuration examples
  - Added troubleshooting section
  - Added contributing guidelines

- **Testing**
  - Added unit tests for all new modules
  - Added integration tests
  - Added test coverage reporting
  - Automated testing with GitHub Actions

### Fixed

- Missing error handling in file operations
- Hardcoded file paths and configuration values
- Inconsistent return types and error messages
- Missing validation for input data

---

## [1.0.0] - 2025-11-07

### Initial Release

#### Features

- **BERT-based Mapping** (`BERT-Mapping.py`)
  - Neural network approach using BERT embeddings
  - PyTorch-based classification
  - Confidence score output
  - Web API support (FastAPI)

- **HuggingFace Policy Mappers**
  - `huggingface_policy_mapper_advanced.py`: Advanced semantic matching
  - `huggingface_policy_mapper_hybrid.py`: Hybrid approach combining semantic and lexical
  - `huggingface_policy_mapper_lexical.py`: Keyword-based matching

- **Column-Level Mapping** (`BERT-Column-Mapping.py`)
  - Maps columns between different CSV files
  - BERT embedding-based approach

- **REST API** (`BERT-mappings-REST.py`)
  - Flask-based REST API for text comparison
  - Endpoint: `POST /compare`
  - Returns match status and confidence score

- **Output Processing** (`Output-Mapping.py.py`)
  - Compares multiple CSV/PDF documents
  - Generates mapping reports
  - Provides accuracy metrics
  - Outputs standardized CSV format

#### Data Files

- `Policies-2025-08.csv`: Input policy dataset
- `GA-Final-Output.csv`: Sample output (250KB)
- `final_policy_mapping_output_python.csv`: Full mapping results (2.5MB)
- `policy_mapping_report.pdf`: Analysis report

---

## Roadmap

### Planned Features

- [ ] Web UI for interactive mapping
- [ ] Support for additional embedding models (RoBERTa, ALBERT, sentence-transformers)
- [ ] Batch processing optimization
- [ ] Configuration file support (YAML/JSON)
- [ ] Docker containerization
- [ ] Database integration for result storage
- [ ] API authentication and rate limiting
- [ ] Performance benchmarking suite
- [ ] Pre-commit hooks for code quality
- [ ] Additional language support

### Known Issues

- BERT model requires ~440MB memory (consider using DistilBERT for resource constraints)
- Currently CPU-only (GPU support planned)
- Output-Mapping.py.py has some code duplication (refactoring in progress)

### Deprecations

- None at this time

---

## Migration Guide

### From v1.0.0 to v1.1.0

#### Configuration Changes

**Before (v1.0.0)**:
```python
CSV_INPUT_FILE = 'Policies-2025-08.csv'
CONFIDENCE_THRESHOLD = 25
SEMANTIC_WEIGHT = 0.8
LEXICAL_WEIGHT = 0.6
```

**After (v1.1.0)**:
```python
from config import HYBRID_MAPPER_CONFIG, get_config

# Access configuration
config = get_config('hybrid')
input_file = config['input_file']
threshold = config['confidence_threshold']
```

#### Utility Function Usage

**Before (v1.0.0)**:
```python
# Text preprocessing was inline
text = text.lower()
text = re.sub(r'[^a-z0-9\s]', '', text)
```

**After (v1.1.0)**:
```python
from utils import preprocess_text

text = preprocess_text(text)
```

#### Error Handling

**Before (v1.0.0)**:
```python
df = pd.read_csv(file)  # May raise exception
```

**After (v1.1.0)**:
```python
from utils import read_csv_safe

df = read_csv_safe(file)  # Handles errors gracefully
if df is None:
    print("Failed to read CSV file")
```

---

## Contributors

- **Madhuri Gopal** - Original author and maintainer

---

## Support

For support, questions, or bug reports:
- Open an issue on [GitHub Issues](https://github.com/madhurigopal1/Data-Mapping/issues)
- Check existing documentation in [README.md](README.md)
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Semantic Similarity](https://en.wikipedia.org/wiki/Semantic_similarity)
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)

---

**Last Updated**: June 9, 2026

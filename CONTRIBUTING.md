# Contributing to Data Mapping Solution

Thank you for your interest in contributing! This document provides guidelines for contributing to the Data Mapping project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv
- Git

### Workflow

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Data-Mapping.git
   cd Data-Mapping
   ```

3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes** and test thoroughly
5. **Push to your fork** and submit a Pull Request

---

## Development Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

### 3. Verify Setup

```bash
python test_data_mapping.py
```

---

## Code Style

### Python Style Guide

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards:

- **Indentation**: 4 spaces
- **Line Length**: Maximum 100 characters
- **Naming**: 
  - `lowercase_with_underscores` for functions and variables
  - `CapitalCase` for classes
  - `UPPER_CASE` for constants

### Code Quality

```bash
# Format code
black *.py

# Check style
flake8 *.py
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> dict:
    """
    Brief description of what the function does.
    
    More detailed explanation if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When parameter is invalid
    """
    pass
```

---

## Testing

### Running Tests

```bash
# Run all tests
python test_data_mapping.py

# Run specific test class
python -m unittest test_data_mapping.TestConfigModule

# Run with pytest
pytest test_data_mapping.py -v

# Run with coverage
pytest test_data_mapping.py --cov=. --cov-report=html
```

### Writing Tests

```python
class TestNewFeature(unittest.TestCase):
    """Test cases for new feature"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.test_data = "sample data"
    
    def test_happy_path(self):
        """Test normal operation"""
        result = function_under_test(self.test_data)
        self.assertEqual(result, expected_value)
    
    def test_error_handling(self):
        """Test error conditions"""
        with self.assertRaises(ValueError):
            function_under_test(None)
```

### Test Coverage Goals

- Aim for >80% code coverage
- Test happy paths, edge cases, and error conditions
- Include integration tests

---

## Submitting Changes

### Before Submitting

1. **Run tests**: `python test_data_mapping.py`
2. **Check style**: `flake8 *.py`
3. **Update documentation**: Modify README.md if needed

### Pull Request Process

1. **Create a clear PR title**: `feat: Add new mapping feature` or `fix: Resolve issue #123`
2. **Write a detailed description**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests added
   - [ ] All tests passing
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Docstrings added/updated
   - [ ] No hardcoded values
   - [ ] Backward compatible
   ```

3. **Link related issues**: "Fixes #123"
4. **Wait for review** and address feedback

---

## Reporting Bugs

### Bug Report Template

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.8.x
- Operating System: Windows/Mac/Linux

## Error Message
```
Full error message and stack trace
```
```

### Bug Report Guidelines

- **Search existing issues** before reporting
- **Be descriptive**: Provide clear, detailed information
- **Include reproducible example**: Minimal code to reproduce
- **Check logs**: Include relevant log messages

---

## Requesting Features

### Feature Request Template

```markdown
## Description
Clear description of the requested feature

## Motivation
Why would this feature be useful?

## Proposed Solution
How should it work?

## Alternatives
Other possible solutions or workarounds
```

---

## Project Structure

```
Data-Mapping/
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── test_data_mapping.py   # Unit tests
├── BERT-Mapping.py        # BERT approach
├── huggingface_policy_mapper_hybrid.py  # Hybrid approach
├── README.md              # Documentation
├── CONTRIBUTING.md        # This file
├── requirements.txt       # Dependencies
├── workflows/tests.yml    # CI/CD workflow
└── data/                  # Data directory
    └── Policies-2025-08.csv
```

---

## Development Tips

### Useful Commands

```bash
# Install in development mode
pip install -e .

# Run tests with verbose output
python -m unittest discover -v

# Format code with black
black .

# Check for style issues
flake8 . --max-line-length=100
```

---

## Questions?

- Check [README.md](README.md) for documentation
- Search [GitHub Issues](https://github.com/madhurigopal1/Data-Mapping/issues)
- Create a new issue with question label

Thank you for contributing! 🎉

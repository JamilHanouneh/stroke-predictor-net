# Contributing to StrokePredictorNet

Thank you for considering contributing to StrokePredictorNet. This document provides guidelines for contributing to the project.

---

## Code of Conduct

Please be respectful and constructive in all interactions.

---

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages and logs

### Suggesting Features

Feature suggestions are welcome. Please open an issue describing:

- The proposed feature
- Use case and benefits
- Potential implementation approach

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`pytest tests/`)
6. Commit with descriptive messages
7. Push to your fork
8. Open a Pull Request

---

## Development Setup

```
# Clone your fork
git clone https://github.com/your-username/StrokePredictorNet.git
cd StrokePredictorNet

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies including dev tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

---

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Formatting

Use `black` for code formatting:

```
black src/ scripts/ tests/
```

### Linting

Use `flake8` for linting:

```
flake8 src/ scripts/ tests/ --max-line-length=100
```

---

## Testing

All contributions should include tests:

```
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update relevant documentation in docs/

---

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add uncertainty quantification module
fix: Correct batch normalization for single samples
docs: Update installation instructions
refactor: Simplify data loading pipeline
test: Add unit tests for attention module
```

---

## Questions

For questions about contributing:
- Open an issue for discussion
- Email: jamil.hanouneh1997@gmail.com

---

Thank you for contributing to StrokePredictorNet!
```
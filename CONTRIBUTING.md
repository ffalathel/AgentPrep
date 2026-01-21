# Contributing to AgentPrep

Thank you for your interest in contributing to AgentPrep! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use `black` for code formatting and `ruff` for linting:

```bash
# Format code
black .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

### Running Tests

Run the full test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=. --cov-report=html
```

Run specific tests:

```bash
pytest tests/test_level1_ingestion/
```

### Code Structure

- Follow the existing module structure
- Each level should be self-contained with clear interfaces
- Use type hints throughout
- Add docstrings for all public functions and classes
- Follow PEP 8 style guidelines (enforced by black/ruff)

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, readable code
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

Write clear commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap to 72 characters.
Explain what and why, not how.
```

### 4. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to any related issues
- Screenshots or examples if applicable

## Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] No linting errors (`ruff check .`)
- [ ] Documentation is updated
- [ ] New features have tests
- [ ] Commit messages are clear
- [ ] No secrets or API keys in code

## Adding New Features

### Adding a New Pipeline Level

1. Create a new `levelN_name/` directory
2. Follow the existing level structure:
   - `__init__.py` - Module exports
   - Main implementation files
3. Integrate into `core/orchestrator.py`
4. Add tests in `tests/test_levelN_name/`
5. Update documentation

### Adding New Dependencies

1. Add to `requirements.txt` with version constraints
2. Add to `pyproject.toml` if it's a core dependency
3. Update `requirements-test.txt` if it's a test dependency
4. Document why the dependency is needed

## Reporting Issues

When reporting bugs or requesting features:

1. Check existing issues first
2. Use the issue templates
3. Provide:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Minimal example if possible

## Code Review Process

1. All PRs require at least one approval
2. Address review comments promptly
3. Keep PRs focused and reasonably sized
4. Respond to feedback constructively

## Questions?

- Open an issue for questions or discussions
- Check existing documentation first
- Be respectful and patient

Thank you for contributing to AgentPrep! 🎉

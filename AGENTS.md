## AGENTS.md

This document provides instructions for agentic coding agents operating in this repository.

### Build, Lint, and Test

- **Dependencies**: Install dependencies from `uv.lock` using `uv sync`.
- **Linting**: Use `ruff` for linting. Run `ruff check .` to check for issues.
- **Testing**: Use `pytest` for testing. Run `pytest` to execute all tests.
  - To run a single test file: `pytest tests/test_your_test_file.py`
  - To run a single test function: `pytest tests/test_your_test_file.py::test_your_function`
- **Pipeline Compilation**: To compile the Kubeflow pipeline, run `uv run pipeline.py`. This will generate `pipeline.yaml`.

### Code Style

- **Imports**: Group imports at the top of the file. Standard library imports first, then third-party imports.
- **Formatting**: Follow PEP 8 guidelines. Use a line length of 100 characters.
- **Types**: Use type hints for function signatures, especially for Kubeflow component inputs and outputs.
- **Naming**: Use `snake_case` for variables and functions. Use `PascalCase` for classes.
- **Error Handling**: Use `try...except` blocks for operations that might fail, such as S3 uploads.
- **Docstrings**: Add docstrings to functions to explain their purpose, arguments, and return values.
- **Secrets**: Access secrets like AWS keys via environment variables, not hardcoded in the source.

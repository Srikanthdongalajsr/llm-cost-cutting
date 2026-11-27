# Contributing Guidelines

Thank you for considering contributing to **LLM Cost Lab**! This document outlines the steps to get you up and running, the coding standards we follow, how to run tests, and the process for submitting pull requests.

---

## Table of Contents

1. [Setting Up the Development Environment](#setting-up-the-development-environment)
2. [Coding Standards](#coding-standards)
3. [Testing Procedures](#testing-procedures)
4. [Submitting Pull Requests](#submitting-pull-requests)
5. [Additional Resources](#additional-resources)

---

## Setting Up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/llm-cost-lab.git
   cd llm-cost-lab
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Copy the example environment file and add your API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add the required keys (e.g., GROQ_API_KEY, OPENAI_API_KEY, etc.)
   ```
5. **Run the test suite** to ensure everything is working:
   ```bash
   pytest
   ```

---

## Coding Standards

- **Python version**: 3.9 or newer.
- **Style guide**: Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- **Formatting**: Use `black` for automatic formatting and `isort` for import ordering.
  ```bash
  black .
  isort .
  ```
- **Linting**: Run `flake8` to catch style issues.
  ```bash
  flake8 .
  ```
- **Type hints**: Add type annotations where appropriate and run `mypy` to check them.
  ```bash
  mypy .
  ```
- **Docstrings**: Use Google style docstrings for all public functions, classes, and modules.

---

## Testing Procedures

- Tests are located in the `tests/` directory and use **pytest**.
- To run the full test suite:
  ```bash
  pytest
  ```
- For a quick check while developing, you can run a single test file:
  ```bash
  pytest tests/test_some_module.py
  ```
- Ensure new code includes appropriate unit tests and that existing tests continue to pass.
- Use **coverage** to verify test coverage (optional but encouraged):
  ```bash
  coverage run -m pytest
  coverage report -m
  ```

---

## Submitting Pull Requests

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your work:
   ```bash
   git checkout -b my-feature-or-bugfix
   ```
3. **Make your changes**, ensuring you adhere to the coding standards and add tests.
4. **Run the test suite** and ensure all checks pass.
5. **Commit your changes** with a clear, concise commit message.
6. **Push the branch** to your fork:
   ```bash
   git push origin my-feature-or-bugfix
   ```
7. **Open a Pull Request** against the `main` branch of the upstream repository.
   - Provide a descriptive title.
   - Include a brief description of what the PR does and why it is needed.
   - Reference any related issues (e.g., `Fixes #12`).
8. **Review process**
   - A maintainer will review your PR and may request changes.
   - Make any required updates and push them to the same branch; the PR will update automatically.
9. Once approved, the maintainer will merge your PR.

---

## Additional Resources

- **Issue Tracker**: Use GitHub Issues to report bugs or request features.
- **Code of Conduct**: Please also review the `CODE_OF_CONDUCT.md` for community guidelines.
- **License**: Contributions are licensed under the same terms as the project (see `LICENSE`).

Thank you for your contributions! 🚀
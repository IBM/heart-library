# Contributing to HEART

Thank you for your interest in contributing to HEART! This guide outlines how you can contribute to the project and the processes in place to ensure smooth collaboration.

## Table of Contents

1. [Overview](#overview)
2. [Reporting Issues](#reporting-issues)
3. [Submitting Feature Requests](#submitting-feature-requests)
4. [Contributing Code](#contributing-code)
   - [Development Workflow](#development-workflow)
   - [Coding Standards](#coding-standards)
   - [Testing](#testing)
5. [Security Issues](#security-issues)
6. [Communication](#communication)
7. [License](#license)

## Overview

This project is developed privately in a GitLab repository and released as a public stub on GitHub. While the full development occurs privately, we encourage the community to report issues, suggest features, and contribute to the project by following the guidelines below.

## Reporting Issues

If you encounter any bugs or issues with the public release, please [open an issue](https://github.com/IBM/heart-library/issues/new) on the GitHub repository. 

**Before submitting an issue, please:**

1. Search existing issues to ensure it hasn't already been reported.
2. Provide as much detail as possible, including steps to reproduce the issue, your environment, and any relevant logs.

## Submitting Feature Requests

We welcome feature requests that align with the project’s goals. To submit a feature request:

1. **Check for Existing Requests:** Review the open issues to see if your feature has already been requested.
2. **Open an Issue:** If your request is unique, open an issue and clearly describe the proposed feature, its use case, and why it would be beneficial.

Please note that feature requests are prioritized based on their alignment with the project’s goals and resource availability.

## Contributing Code

Given the repository’s development model, direct code contributions via pull requests are generally **not accepted** in the public GitHub repository. However, we welcome suggestions for improvements and patches through issue discussions.

### Development Workflow

1. **Fork the Repository:** Start by forking the GitHub repository.
2. **Clone Your Fork:** Clone the repository to your local machine.
3. **Create a Branch:** Create a new branch for your work (`git checkout -b feature-name`).
4. **Code & Commit:** Make your changes, commit them, and push the branch to your fork.
5. **Open an Issue:** Submit a detailed issue with a link to your forked branch or a description of the changes you've made.

### Coding Standards

While direct contributions are generally not merged into the public GitHub repository, we follow strict coding standards internally. If you're submitting code snippets or suggestions, please adhere to the following:

- **PEP 8:** Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- **Ruff for Linting:** Use `ruff` to automatically lint your code. The specific linting configurations can be found in the `pyproject.toml` file within the repository. Ensure your code passes all linting checks before submitting.
- **Docstrings:** Use [PEP 257](https://www.python.org/dev/peps/pep-0257/) for docstrings in functions and classes.
- **Type Hints:** Type hints must be used for both function parameters and return types, in accordance with [PEP 484](https://peps.python.org/pep-0484/)

### Testing

All code suggestions should be accompanied by relevant unit tests. We use `pytest` for testing. Even though direct contributions are not merged, please include test cases when suggesting patches or improvements.

## Security Issues

If you discover a security vulnerability in the project, **please do not report it publicly on GitHub.** Instead, post in our discussions tab to indicate that you have discovered a vulnerability (without providing any specifics). One of our team members will then reach out to gather the specifics and implement a resolution.

## Communication

For general questions, feedback, or discussions about the project, you can:

- **GitHub Discussions:** Start a discussion in the [GitHub Discussions](https://github.com/IBM/heart-library/discussions) tab.

Please note that the development team primarily works in a private GitLab repository, so some decisions and discussions may occur outside of public view.

## License

By contributing to HEART, you agree that your contributions will be licensed under the [LICENSE](LICENSE.txt) in this repository.

---

Thank you for your contributions and support!
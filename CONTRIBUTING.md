# Contributing to Hyperlocal Air Quality Forecasting

Thank you for your interest in contributing to this project! We welcome community contributions, bug fixes, enhancements, and documentation updates.

---

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

---

## How Can I Contribute?

### Reporting Bugs
If you find a bug:
1. Check existing GitHub Issues to see if it has already been reported.
2. If not, open a new issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
3. Include clear steps to reproduce the issue, expected behavior, and system details (OS, Python version).

### Suggesting Enhancements
Feature requests are welcome!
1. Check existing issues or discussions before opening a new one.
2. Open an issue using our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).
3. Describe the proposed feature, rationale, and any alternative solutions considered.

### Pull Requests
1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Set up environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Make your changes**:
   - Write clean, documented code following PEP 8.
   - Maintain reproducible pipeline designs (do not introduce data leakage into time-series features).
4. **Run tests**:
   Ensure all existing and new tests pass before submitting:
   ```bash
   pytest tests/ -v
   ```
5. **Commit and Push**:
   Use descriptive commit messages:
   ```bash
   git commit -m "feat: add lag features for wind speed"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**:
   Fill out the pull request template with details of your changes.

---

## Development Guidelines
- **Time-series integrity**: Never leak future timestamps into past feature windows. All lag and rolling calculations must reference past records strictly ($\le t$).
- **Configuration**: Avoid hardcoding parameters in Python scripts; expose configurable values in `configs/config.yaml`.
- **Reproducibility**: Fix random seeds (`random_state=42`) for any stochastic processes.


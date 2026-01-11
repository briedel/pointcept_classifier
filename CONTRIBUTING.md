# Contributing to Pointcept Classifier

Thank you for your interest in contributing to the Pointcept Classifier for IceCube events!

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/briedel/pointcept_classifier.git
cd pointcept_classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. Install development dependencies:
```bash
pip install pytest pytest-cov black flake8 mypy
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

Format code with Black:
```bash
black pointcept_classifier/ train.py inference.py
```

Check style with flake8:
```bash
flake8 pointcept_classifier/ --max-line-length=100
```

## Testing

Run tests before submitting:
```bash
pytest tests/
```

Add tests for new features in the `tests/` directory.

## Submitting Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/my-new-feature
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add feature: description"
```

3. Push to your fork:
```bash
git push origin feature/my-new-feature
```

4. Open a Pull Request on GitHub

## Areas for Contribution

- **Data loaders**: Support for additional data formats
- **Model architectures**: Integration with more Pointcept models
- **Preprocessing**: Additional augmentation techniques
- **Evaluation**: More comprehensive metrics
- **Documentation**: Tutorials and examples
- **Performance**: Optimization and profiling
- **Testing**: Increase test coverage

## Questions?

Open an issue on GitHub for questions or discussions.

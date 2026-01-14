# Contributing

We welcome contributions! Please follow these guidelines.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Civilization.git
   cd Civilization
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

4. Run code quality checks:
   ```bash
   black civsim
   flake8 civsim
   mypy civsim
   ```

5. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

6. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```

7. Open a Pull Request

## Code Standards

### Python Style
- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings in Google style

### Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- Keep first line under 50 characters
- Reference issues: "Fix #123: issue description"

### Testing
- Write tests for new features
- Maintain test coverage above 80%
- Run tests before submitting PR

## Areas for Contribution

### Code Improvements
- Performance optimizations
- Bug fixes
- New features

### Documentation
- Improve README files
- Add code examples
- Update API documentation

### Testing
- Add unit tests
- Integration tests
- Performance benchmarks

### Research
- New AI algorithms
- Advanced simulation features
- Visualization improvements

## Issue Reporting

When reporting bugs:
- Use the issue template
- Include Python version and OS
- Provide minimal reproduction case
- Include error messages and stack traces

## Code Review Process

All submissions require review. We'll:
- Review code for style and correctness
- Run automated tests
- Check for performance implications
- Verify documentation updates

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
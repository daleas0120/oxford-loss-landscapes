# Tests

This directory contains the test suite for the Oxford Loss Landscapes package.

## Structure

- `conftest.py` - Test configuration and fixtures
- `test_basic.py` - Basic functionality tests
- Future test modules will be added here

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=oxford_loss_landscapes

# Run specific test file
pytest tests/test_basic.py
```

## Test Categories

- **Unit Tests**: Testing individual functions and classes
- **Integration Tests**: Testing module interactions
- **Example Tests**: Ensuring examples run correctly

## Writing Tests

When adding new functionality, please add corresponding tests:

1. Create test files following the `test_*.py` naming convention
2. Use descriptive test function names starting with `test_`
3. Include docstrings explaining what each test validates
4. Use appropriate pytest fixtures for setup/teardown

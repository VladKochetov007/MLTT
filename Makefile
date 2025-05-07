.PHONY: all clean install develop build test

# Default Python virtual environment path
VENV_PATH ?= .venv

all: develop

# Clean all build artifacts
clean:
	rm -rf *.egg-info/
	rm -rf dist/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.so" -exec rm -f {} +

# Install in development mode
develop:
	pip install -e .

# Install in production mode
install:
	pip install .


# Full clean rebuild and install
rebuild: clean install

# Cognitive Dddballs

Final mini-project for the "Models of learning and descision making in changing environments" seminar at the university of Osnabrück in the winter term 2025/26.

# Short Description of the project

TODO

# Development Setup

## Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Code Quality with Ruff

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

### Installation

```bash
pip install -r requirements-dev.txt
```

or directly via pip:

```bash
pip install ruff
```

### Usage

**Linting:**
```bash
ruff check .
```

**Formatting:**
```bash
ruff format .
```

**Auto-fix issues:**
```bash
ruff check --fix .
```

### Configuration

Ruff is configured in `pyproject.toml` or `ruff.toml` at the project root.



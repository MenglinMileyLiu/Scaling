# scaling

## Introduction

The **scaling** package is designed to study scaling and ordinal classification problems in political science. It implements a two-stage workflow: (1) document summarization to extract meaningful insights from political texts, and (2) pairwise comparison scaling to position items on an ordinal spectrum based on comparative judgments.

## File Architecture

- `src/scaling/`  
  Core Python package containing modules for summarization, scaling algorithms, and utility functions.

- `notebooks/`  
  Jupyter notebooks for exploratory data analysis, experiments, and demonstrations.

- `docs/`  
  Project documentation and references.

- `tests/`  
  Unit and integration tests to ensure code quality and correctness.

- `data/` (optional)  
  Directory for storing raw and processed datasets (typically excluded from version control).

## Installation

To set up the project for development and enable collaborative contributions, follow these steps:

0. Assume python3 and pip3 are installed, install hatch
   ```bash
   python3 -m pip install --upgrade hatch hatchling
   ```

1. Clone the repository:
   ```bash
   git clone https://github.com/MenglinMileyLiu/Scaling
   cd Scaling
   ```

2. Create and install the development environment using Hatch:
   ```bash
   hatch env create
   ```

3. Install the package in editable mode along with development dependencies:
   ```bash
   hatch run pip install -e .
   ```

4. (Optional) Manually activate the Hatch-managed environment shell, you can avoid hatch run:
   ```bash
   hatch shell
   ```

5. (Optional) Exit the virtual environment if you manually activated it:
   ```bash
   deactivate
   ```

## Common Hatch Commands

| Command                           | Description                                        |
|----------------------------------|--------------------------------------------------|
| `hatch env create`               | Create the default development environment        |
| `hatch env create <name>`        | Create a named environment `<name>`               |
| `hatch shell`                    | Open a shell with the project environment active  |
| `hatch run <command>`            | Run a command inside the default environment      |
| `hatch run -e <name> <command>` | Run a command inside the named environment `<name>` |
| `hatch run python`               | Run the Python interpreter inside the environment |
| `hatch run pytest`               | Run tests using pytest                             |
| `hatch run flake8 src tests`     | Run the linter on source and test directories     |
| `hatch run black src tests`      | Format code with Black                             |
| `hatch run mypy src`             | Run static type checking                           |


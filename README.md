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

- `requirements.txt`
  Append the use dependencies with versions to it

- `.gitignore`
  Use .gitignore to exclude files like logs, build artifacts, intermediate results, large data.

## Installation

To set up the project for development and enable collaborative contributions, follow these steps:

0. Assume python3 and pip3 are installed, install hatch
```bash
    python3 -m pip install --upgrade hatch hatchling
    python3 -m pip install hatch-requirements-txt
```

1. Clone the repository:
```bash
    git clone https://github.com/MenglinMileyLiu/Scaling
    cd Scaling
```

2. Create and install the development environment using python3.11:
```bash
    python3.11 -m venv .venv
```

3. Activate the vertual environment to create isolated development env:
```bash
    source .venv/bin/activate 
```

4. Install the package in editable mode along with development dependencies:
```bash
    pip install -e .
```

5. (Optional) Exit the current virtual environment:
```bash
    deactivate
```

6. Copy .env.example to .env, enter the personal API keys as needed.

## Git Cheat Sheet (Version Controll)
Developers normally work on some local dev branch. Avoid directly work on main branch. Use "stash" to bring uncommitted changes across branches.

### Setup & Info
```bash
    git config --global user.name "Your Name"
    git config --global user.email "you@example.com"
    git status               # Show current changes, branch status, and conflics
    git log                  # Show commit history
```

### Starting & Cloning
```bash
    git init                # Initialize a new repo locally
    git clone <url>         # Clone remote repo locally
```

### Working with Changes
```bash
    git add <file>          # Stage file for commit
    git add .               # Stage all changed files
    git diff                # Show unstaged changes
    git diff --staged       # Show staged changes
```

### Commit & Amend
```bash
    git commit -m "Your commit message"   # Commit staged changes
    git commit --amend               # Amend last commit (use with caution)
```

### Branching
```bash
    git branch                    # List local branches
    git branch --all              # List both local and remote branches
    git branch <new-branch>       # Create a new branch
    git checkout <branch>         # Switch to branch
    git switch <branch>           # (Alternative to checkout)
    git checkout -b <branch>      # Create & switch to new branch
```

### Push & Pull & Merge
```bash
    git pull                     # Fetch + merge from remote
    git merge <branch>           # Merge branch into current branch
    git merge --squash <branch>  # Merge squashed commits from a branch
    git push                     # Push current branch to remote
    git push -u origin <branch>  # Push and set upstream branch
```

### Undo Changes
```bash
    git restore <file>          # Undo unstaged changes in a file
    git restore --staged <file> # Unstage a staged file
    git checkout -- .           # Discards unstaged changes
```

### Stash Work (Temporary Save)
```bash
    git stash                   # Save current changes and clean working directory
    git stash pop               # Reapply stashed changes and remove stash
    git stash list              # List stashes
```

### Rebase & Conflicts
```bash
    # In some local branch
    git fetch origin            # Fetch the remote origin changes
    git rebase origin/main      # Apply your changes on local branch on the basis of remote origin
    # Resolve conflicts by editing files
    git rebase --continue       # Confinue rebase changes
    # After conflicts are resolved, commit again
```

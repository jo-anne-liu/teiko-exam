# teiko-exam

## Instructions for Running
1. Add files "cell-count.csv," "part1.py," "part2.py," "part3.py," "part4.py," and "part5.py" to GitHub repository.
2. Create a folder called .devcontainer at the root of the repo and add two files:

.devcontainer/devcontainer.json
```
{
  "name": "Loblaw Study Python Env",
  "dockerFile": "Dockerfile",
  "settings": {
    "terminal.integrated.shell.linux": "/bin/bash"
  },
  "extensions": [
    "ms-python.python",
    "ms-toolsai.jupyter"
  ],
  "postCreateCommand": "pip install -r requirements.txt"
}
```

.devcontainer/Dockerfile
```
# Use the official Python image (choose the version you prefer)
FROM python:3.11-slim

# Install OS‑level dependencies needed by scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libatlas-base-dev \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Create a non‑root user (optional but nice)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /workspaces/${PWD##*/}

# Switch to the user for the rest of the steps
USER $USERNAME
```

requirements.txt
```
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
seaborn>=0.13
matplotlib>=3.8
```
3. Launch codespace
4. Verify that the csv file is in the directory: ls -l *.csv
5. Run part 1: python part1.py cell_counts.csv loblaw_study.db
6. Run part 2: python part2.py loblaw_study.db
7. Run part 3: python part3.py frequencies.csv --db loblaw_study.db --out stats_results.csv --plot pbmc_boxplot.png
8. Run part 4: python part4.py loblaw_study.db
9. Run part 5: python part5.py

## Explanation of Relational Schema

## Overview of Code Structure

## Dashboard Link: 
* https://jo-anne-liu.github.io/teiko.html

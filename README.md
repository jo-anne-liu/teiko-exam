# Teiko Exam for Bioinformatics Engineer

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

### Tables and explanations
* patients: The composite key guarantees uniqueness across projects (the same subject ID could appear in different studies, so we prepend the project name).
* samples: One row per biological specimen collected from a patient. A patient can contribute many samples (different time points, treatments, tissue types).
* cell_types:	Master lookup table for every immune‑cell population you ever measure (e.g., b_cell, cd8_t_cell). Adding a new marker only means inserting a new row here.
* measurements:	The numeric observation: how many cells of a given type were counted in a given sample. The (sample_id, cell_type_id) pair is declared UNIQUE so you never store duplicate counts for the same cell type in the same sample.

### Visual relationship diagram

patients 1 ──< samples >───* measurements *───> cell_types
   ^                                   ^
   |                                   |
   +-----------------------------------+
         (patient_id)            (sample_id)

### Rationale
* Separate tables (normalization)	eliminates redundancy (e.g., patient age is stored once, not repeated for every cell‑type measurement) and reduces storage, prevents contradictory records, and makes updates trivial.
* Composite patient_id (project_subject) guarantees global uniqueness across projects. If two studies happen to label a subject “001”, the prefix keeps them distinct without needing an artificial surrogate key.
* SQLite will reject a measurement that refers to a non‑existent sample, and a sample that refers to a non‑existent patient. This protects from accidental typos or incomplete imports.
* `cell_types` lookup table	decouples the list of markers from the measurements. Adding a new marker (e.g., regulatory_t_cell) is a single INSERT into cell_types; the loading loop automatically discovers the new ID via _cell_type_id. No code changes needed.
* UNIQUE(sample_id, cell_type_id) in measurements	guarantees there is at most one count per cell type per sample. If you re‑run the loader on the same CSV, the INSERT OR REPLACE will simply overwrite the old value rather than create duplicates.
* Integer primary keys (AUTOINCREMENT) for surrogate tables	promotes fast indexing and look‑ups. Integer PKs are compact and SQLite can use them as rowids, which speeds up joins.
* Storing time_from_treatment_start as an integer	makes range queries cheap (WHERE time_from_treatment_start BETWEEN 0 AND 30). It also enables easy grouping by time bins in downstream analysis.
* Explicit sample_type column	allows you to filter on tissue source (e.g., PBMC, tumor, blood) without having to parse column names. This is essential for analyses that are tissue‑specific.

## Overview of Code Structure
Each `.py` file completes a part of the exam. I structured the code this way so that we can stop at each step and troubleshoot if necessary. The intermediate results are preserved in files so the entire pipeline doesn't need to be run for later results in the pipeline. Furthermore, dividing the code into functions allows us to optimize the runtime of the function. 

For scalability, SQLite efficiently handles data with millions of rows. Batch inserts inside a single transaction (with conn: in `part1.py`) dramatically speeds up the initial load.  `part2.py` and `part3.py` leverage group‑by operations that are far faster than Python loops.
Scripts open connections only when needed and close them promptly, keeping the memory footprint modest.

Keeping all the code in Python, a freely available coding language, makes the pipeline accessible for all people. The minimal packages also make the code more accessible. 

### part1.py
* Parses the cell-count.csv file and populates a SQLite database

### part2.py
* Reads the database, flattens the relational model into a pandas DataFrame, and computes per‑sample total cell counts and the relative frequency of each cell type.

### part3.py
* Performs statistical comparisons of those frequencies between responders (response=="yes") and non‑responders (response=="no"), restricted to PBMC samples. Produces a results table and a box‑plot.

### part4.py
* Extracts a very specific subset: baseline (time = 0) melanoma PBMC samples treated with miraclib. Summarises sample counts per project, responder counts, and gender distribution.

### part5.py
* Runs a single aggregated query: average B‑cell count for male melanoma responders at baseline (time = 0).

## Dashboard Link: 
* https://jo-anne-liu.github.io/teiko.html

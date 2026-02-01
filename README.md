# Teiko Exam for Bioinformatics Engineer

## Instructions for Running
1. Add files "cell-count.csv" and "main.py" to GitHub repository.
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
5. Run part 1: python main.py path/to/cell-count.csv loblaw_study.db
6. Outputs will be files `melanoma_samples_per_project.csv`, `melanoma_responders_vs_non.csv`, `melanoma_gender_counts.csv`, `stat_comparison.csv`, `pbmc_boxplot.png`, `frequencies.csv`.

## Explanation of Relational Schema

### Tables and explanations
* patients: The composite key guarantees uniqueness across projects (the same subject ID could appear in different studies, so we prepend the project name).
* samples: One row per biological specimen collected from a patient. A patient can contribute many samples (different time points, treatments, tissue types).
* cell_types:	Master lookup table for every immune‑cell population you ever measure (e.g., b_cell, cd8_t_cell). Adding a new marker only means inserting a new row here.
* measurements:	The numeric observation: how many cells of a given type were counted in a given sample. The (sample_id, cell_type_id) pair is declared UNIQUE so you never store duplicate counts for the same cell type in the same sample.

### Rationale
* Separate tables (normalization)	eliminates redundancy (e.g., patient age is stored once, not repeated for every cell‑type measurement) and reduces storage, prevents contradictory records, and makes updates trivial.
* Composite patient_id (project_subject) guarantees global uniqueness across projects. If two studies happen to label a subject “001”, the prefix keeps them distinct without needing an artificial surrogate key.
* SQLite will reject a measurement that refers to a non‑existent sample, and a sample that refers to a non‑existent patient. This protects from accidental typos or incomplete imports.
* `cell_types` lookup table	decouples the list of markers from the measurements. Adding a new marker (e.g., regulatory_t_cell) is a single INSERT into cell_types; the loading loop automatically discovers the new ID via _cell_type_id. No code changes needed.
* UNIQUE(sample_id, cell_type_id) in measurements	guarantees there is at most one count per cell type per sample. If you re‑run the loader on the same CSV, the INSERT OR REPLACE will simply overwrite the old value rather than create duplicates.
* Integer primary keys (AUTOINCREMENT) for surrogate tables	promotes fast indexing and look‑ups. Integer PKs are compact and SQLite can use them as rowids, which speeds up joins.
* Storing time_from_treatment_start as an integer	makes range queries cheap (WHERE time_from_treatment_start BETWEEN 0 AND 30). It also enables easy grouping by time bins in downstream analysis.
* Explicit sample_type column	allows you to filter on tissue source (e.g., PBMC, tumor, blood) without having to parse column names. This is essential for analyses that are tissue‑specific.

### Scaling for hundreds of projects and thousands of samples
* SQLite can comfortably hold tens of millions of rows (especially when indexed). With ~1 000 patients × 10 samples/patient × 10 cell types ≈ 100 000 measurement rows, the size of the data well within limits. To handle large amounts of data, keep the DB on SSD storage (GitHub Codespaces, local SSD, or a mounted volume).
* Primary‑key indexes (patient_id, sample_id, cell_type_id) make look‑ups O(log N). Joins on these keys stay fast even as N grows. To scale up, eriodically run VACUUM; ANALYZE; (or let SQLite auto‑vacuum) to keep the file compact and the query planner optimal.
* To add new projects, just insert new rows into patients (with a new project prefix) and corresponding samples. No schema alteration needed.	Use bulk INSERT statements or executemany for efficiency when loading many rows at once.
* To add new cell‑type markers, insert a new row into cell_types. The loader’s _cell_type_id will fetch the new ID automatically.	Keep a master list of allowed cell types in a separate config file (YAML/JSON) if you want to validate incoming CSVs before loading.

### Example new analytics to perform
* Longitudinal trajectories (e.g., cell‑type counts over time): Query measurements joined to samples and filter on patient_id and time_from_treatment_start. Example: SELECT time_from_treatment_start, ct.cell_type, m.count FROM measurements m JOIN samples s ON m.sample_id=s.sample_id JOIN cell_types ct ON m.cell_type_id=ct.cell_type_id WHERE s.patient_id='projA_001' ORDER BY time_from_treatment_start;
* Cross‑project meta‑analysis:	Group by project in the patients table, then join to measurements. Example: SELECT project, ct.cell_type, AVG(m.count) AS mean_count FROM measurements m JOIN samples s ON m.sample_id=s.sample_id JOIN patients p ON s.patient_id=p.patient_id JOIN cell_types ct ON m.cell_type_id=ct.cell_type_id GROUP BY project, ct.cell_type;
* Multi‑omics integration (e.g., linking gene expression stored elsewhere): Store the external data in a separate SQLite database or a linked table (e.g., gene_expression). Use the same patient_id/sample_id keys to join across databases.
* Sub‑cohort extraction (e.g., “all melanoma PBMC baseline samples”): Exactly the query used in part4.py. The same pattern scales: just add more predicates (WHERE condition='melanoma' AND sample_type='PBMC' AND time_from_treatment_start=0).
* Statistical modelling (mixed‑effects, survival):	Export the joined tidy table (SELECT … FROM …) to a Pandas DataFrame, then feed it to statsmodels or scikit‑learn. Because the DB is already normalized, the exported DataFrame is tidy (one row per measurement) and ready for pivoting or aggregation.
* Feature engineering (e.g., ratios of cell types):	Compute derived columns on the fly in SQL (COUNT(b_cell)/COUNT(cd8_t_cell) AS b_cd8_ratio) or in Pandas after export. The relational layout makes it trivial to pivot the long table into a wide matrix (pivot_table(index='sample', columns='cell_type', values='count')).

## Overview of Code Structure
Each `.py` file completes a part of the exam. I structured the code this way so that we can stop at each step and troubleshoot if necessary. The intermediate results are preserved in files so the entire pipeline doesn't need to be run for later results in the pipeline. Furthermore, dividing the code into functions allows us to optimize the runtime of the function. 

For scalability, SQLite efficiently handles data with millions of rows. Batch inserts inside a single transaction (with conn: in `part1.py`) dramatically speeds up the initial load.  `part2.py` and `part3.py` leverage group‑by operations that are far faster than Python loops.
Scripts open connections only when needed and close them promptly, keeping the memory footprint modest.

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

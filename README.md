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
5. Run: python main.py path/to/cell-count.csv loblaw_study.db
6. Outputs will be files `melanoma_samples_per_project.csv`, `melanoma_responders_vs_non.csv`, `melanoma_gender_counts.csv`, `stat_comparison.csv`, `pbmc_boxplot.png`, `frequencies.csv`.










## Overview of Code Structure
I structured the code so each step is printed and we can troubleshoot. Furthermore, dividing the code into functions allows us to optimize the runtime of the function. We've stored the results in files so they can be viewed later.

The skeleton of the file is as follows
```
main.py
│
├─ Imports & constants
│
├─ ── Part 1 – CSV → SQLite loader (load_csv, _cell_type_id)
│
├─ ── Part 2 – Frequency builder (load_flat_dataframe, compute_frequencies)
│
├─ ── Part 3 – PBMC statistical analysis
│     │   • DB‑to‑DataFrame loader (load_from_db)
│     │   • CSV‑to‑DataFrame loader (load_from_csv)
│     │   • Filtering & cleaning helpers (filter_pbmc, clean_response_column)
│     │   • compare_groups (Mann‑Whitney / t‑test + FDR)
│     │   • plot_boxplot (Seaborn box‑plot)
│
├─ ── Part 4 – Melanoma‑PBMC baseline summary
│     │   • SQL helper (sql_to_df)
│     │   • get_baseline_melanoma_pbmc
│     │   • samples_per_project, responders_vs_non, gender_counts
│
├─ ── Part 5 – One‑line B‑cell query (raw SQL at bottom)
│
└─ main() – orchestrator
      • parses CLI args (csv_path, db_path)
      • runs the five steps in order
      • writes CSV/PNG artefacts
      • prints a final status message
```

### Part 1 – CSV → SQLite (load_csv)
* Reads the raw CSV with csv.DictReader.
* Builds the normalized schema (patients, samples, cell_types, measurements).
* Uses _cell_type_id to guarantee a unique ID for each cell‑type string.
* Inserts rows with INSERT OR IGNORE / INSERT OR REPLACE to keep the operation idempotent.
* Commits once per file (single transaction) → fast bulk load.

### Part 2 – Frequency computation
* load_flat_dataframe pulls the long table (one row per measurement) via a single SQL join.
* compute_frequencies aggregates counts per sample, merges totals back, and calculates % per cell type.

### Part 3 – Statistical comparison
* load_from_db builds the same long table but also adds total_count in‑SQL (window function) and pre‑computes %.
* filter_pbmc restricts analysis to PBMC samples.
* clean_response_column normalises the response field and drops any unexpected values.
* compare_groups iterates over each cell‑type, splits responders vs. non‑responders, runs either Mann‑Whitney U or Welch’s t‑test, computes an effect size, and applies a multiple‑testing correction (multipletests).
* plot_boxplot draws a Seaborn box‑plot of percentages by response and optionally saves it.

### Part 4 – Melanoma baseline summary
* get_baseline_melanoma_pbmc extracts the subset defined by condition = ‘melanoma’, sample_type = ‘PBMC’, time = 0, treatment = ‘miraclib’.
* samples_per_project, responders_vs_non, gender_counts each produce a tidy summary table (project counts, responder counts, gender distribution).

### Part 5 – One‑line B‑cell query
* A raw SQL statement (parameterised by TARGET_CELL_TYPE) computes the average B‑cell count for melanoma‑male responders at baseline.
* Executed after the main pipeline finishes; the result is printed with two‑decimal formatting.

### `main()` – Orchestrator
* Argument parsing (csv_path, db_path).
* Step 1 – call load_csv.
* Step 2 – compute frequencies, write frequencies.csv.
* Step 3 – load full DB, filter/clean, run compare_groups, write stat_comparison.csv, save pbmc_boxplot.png.
* Step 4 – connect to DB, pull melanoma subset, write three CSV summary tables.
* Step 5 – run the B‑cell query (currently commented out).
* Print a final “All steps completed” banner.

### Why is the Code Written this Way?
* Reproducibility – Every run produces the same set of artefacts (*.db, *.csv, *.png). You can version‑control the script and the generated files become a complete audit trail.
* Modularity – Want to swap the statistical test? Just change the test= argument in compare_groups. Need a different filter (e.g., another tissue type)? Modify filter_pbmc or add a new filter function without touching the rest of the code.
* Scalability – The bulk CSV load is wrapped in a single SQLite transaction, which is fast even for millions of rows. Later you could replace the CSV reader with pandas.read_csv for even larger files.
* Extensibility – Because each part returns a pandas.DataFrame, you can pipe the output into any downstream analysis (machine‑learning models, additional visualisations, export to Excel, etc.).
* Transparency – All SQL statements are explicit strings; you can inspect or log them for debugging or for sharing with collaborators who prefer raw SQL.


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
* Sub‑cohort extraction (e.g., “all melanoma PBMC baseline samples”): Exactly the query used in Part 4. The same pattern scales: just add more predicates (WHERE condition='melanoma' AND sample_type='PBMC' AND time_from_treatment_start=0).
* Statistical modelling (mixed‑effects, survival):	Export the joined tidy table (SELECT … FROM …) to a Pandas DataFrame, then feed it to statsmodels or scikit‑learn. Because the DB is already normalized, the exported DataFrame is tidy (one row per measurement) and ready for pivoting or aggregation.
* Feature engineering (e.g., ratios of cell types):	Compute derived columns on the fly in SQL (COUNT(b_cell)/COUNT(cd8_t_cell) AS b_cd8_ratio) or in Pandas after export. The relational layout makes it trivial to pivot the long table into a wide matrix (pivot_table(index='sample', columns='cell_type', values='count')).



## Dashboard Link: 
* https://jo-anne-liu.github.io/teiko.html

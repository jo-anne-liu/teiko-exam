import csv
import pathlib
import sqlite3
import sys
from typing import Tuple

# for generating the SQL schema initially
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    project    TEXT,
    subject    TEXT,
    condition  TEXT,
    age        INTEGER,
    sex        TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id                 TEXT PRIMARY KEY,
    patient_id                TEXT NOT NULL,
    treatment                 TEXT,
    response                  TEXT,
    sample_type               TEXT,
    time_from_treatment_start INTEGER,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS cell_types (
    cell_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_type    TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id      TEXT NOT NULL,
    cell_type_id   INTEGER NOT NULL,
    count          INTEGER NOT NULL,
    FOREIGN KEY (sample_id)    REFERENCES samples(sample_id),
    FOREIGN KEY (cell_type_id) REFERENCES cell_types(cell_type_id),
    UNIQUE(sample_id, cell_type_id)
);
"""

# Helper: get/create a cell_type_id
def _cell_type_id(cur: sqlite3.Cursor, name: str) -> int:
    """Return the existing id for *name* or insert it and return the new id."""
    cur.execute("SELECT cell_type_id FROM cell_types WHERE cell_type = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute("INSERT INTO cell_types (cell_type) VALUES (?)", (name,))
    return cur.lastrowid

# reads CSV and populates the DB
def load_csv(csv_path: pathlib.Path, db_path: pathlib.Path) -> None:
    """Parse the CSV you showed and fill the SQLite DB."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Open connection, create schema (if it doesn't exist yet)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)   # creates all four tables
    conn.commit()

    # Bulk load inside ONE transaction
    with conn:  # starts a transaction automatically
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Patients (project, subject, condition, age, sex)
                patient_id = f"{row['project']}_{row['subject']}"
                cur.execute(
                    """
                    INSERT OR IGNORE INTO patients
                        (patient_id, project, subject, condition, age, sex)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        row["project"].strip(),
                        row["subject"].strip(),
                        row["condition"].strip(),
                        int(row["age"]),
                        row["sex"].strip(),
                    ),
                )

                # Samples (treatment, response, sample_type, time...)
                sample_id = row["sample"].strip()
                cur.execute(
                    """
                    INSERT OR IGNORE INTO samples
                        (sample_id, patient_id, treatment, response,
                         sample_type, time_from_treatment_start)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        patient_id,
                        row["treatment"].strip(),
                        row["response"].strip(),
                        row["sample_type"].strip(),
                        int(row["time_from_treatment_start"]),
                    ),
                )

                # Cell‑type measurements (b_cell, cd8_t_cell, …)
                # The remaining columns are the numeric cell counts.
                # iterate over them dynamically to add/remove
                # cell types without touching the code
                cell_columns = [
                    "b_cell",
                    "cd8_t_cell",
                    "cd4_t_cell",
                    "nk_cell",
                    "monocyte",
                ]

                for col in cell_columns:
                    count_raw = row[col].strip()
                    if count_raw == "":
                        continue  # skip missing values
                    count = int(count_raw)

                    # Resolve (or create) the cell_type_id
                    ct_id = _cell_type_id(cur, col)

                    # Insert measurement – replace if duplicate (shouldn't happen)
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO measurements
                            (sample_id, cell_type_id, count)
                        VALUES (?, ?, ?)
                        """,
                        (sample_id, ct_id, count),
                    )

    # quick sanity‑check
    cur.execute("SELECT COUNT(*) FROM measurements;")
    total_measurements = cur.fetchone()[0]
    print(f"Loaded {total_measurements:,} measurement rows into {db_path.name}")

    # Show the first few measurement rows (joined view) for verification
    cur.execute(
        """
        SELECT m.measurement_id,
               m.sample_id,
               ct.cell_type,
               m.count
        FROM measurements m
        JOIN cell_types ct ON m.cell_type_id = ct.cell_type_id
        ORDER BY m.measurement_id ASC
        LIMIT 5;
        """
    )
    preview = cur.fetchall()
    print("\nFirst 5 measurement rows (measurement_id, sample_id, cell_type, count):")
    for r in preview:
        print(r)

    conn.close()
    
#!/usr/bin/env python
"""
part2_frequency.py
-----------------
Compute, for every sample, the total cell count and the relative
frequency (percentage) of each immune‑cell population.

Usage
-----
    python part2.py /path/to/loblaw_study.db  # prints preview
    python part2.py /path/to/loblaw_study.db --out frequencies.csv
"""

import argparse
import pathlib
import sqlite3
import sys
from typing import Tuple
import pandas as pd


# Load the relational data into a flat DataFrame
def load_flat_dataframe(db_path: pathlib.Path) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per measurement:
        sample, patient_id, treatment, response,
        sample_type, time_from_treatment_start,
        population (cell type), count
    """
    con = sqlite3.connect(db_path)

    query = """
        SELECT
            s.sample_id      AS sample,
            p.patient_id,
            p.project,
            p.subject,
            p.condition,
            p.age,
            p.sex,
            s.treatment,
            s.response,
            s.sample_type,
            s.time_from_treatment_start,
            ct.cell_type     AS population,
            m.count
        FROM measurements m
        JOIN samples s      ON m.sample_id = s.sample_id
        JOIN patients p    ON s.patient_id = p.patient_id
        JOIN cell_types ct ON m.cell_type_id = ct.cell_type_id
        ORDER BY s.sample_id, ct.cell_type;
    """

    df = pd.read_sql_query(query, con)
    con.close()
    return df


# Core calculation – total per sample + percentage per population
def compute_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: flat DataFrame from `load_flat_dataframe`.
    Output: tidy DataFrame with columns:
        sample, total_count, population, count, percentage
    """
    # total cells per sample
    totals = (
        df.groupby("sample")["count"]
        .sum()
        .rename("total_count")
        .reset_index()
    )  # one row per sample

    # merge totals back onto the long table
    merged = df.merge(totals, on="sample", how="left")

    # compute percentage (rounded to two decimals)
    merged["percentage"] = (
        merged["count"] / merged["total_count"] * 100
    ).round(2)

    # keep only the columns Bob asked for, in the requested order
    result = merged[
        ["sample", "total_count", "population", "count", "percentage"]
    ].copy()

    # Sort for readability: sample order then population alphabetically
    result.sort_values(["sample", "population"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


"""
part3.py
-----------------------
Statistical comparison of PBMC cell‑type relative frequencies
between responders (response == "yes") and non‑responders (response == "no").

Key fixes compared to the previous version:
* When a CSV is supplied, the path to the SQLite DB is **required** via --db.
* The script no longer tries to guess a DB name like <csv_stem>.db.
* Clear error messages are raised if the DB cannot be opened.
"""

import argparse
import pathlib
import sqlite3
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests


# Load helpers
def load_from_db(db_path: pathlib.Path) -> pd.DataFrame:
    """
    Build the frequency table directly from the SQLite DB.
    Returns a DataFrame with columns:
        sample, total_count, population, count, percentage,
        treatment, response, sample_type
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    con = sqlite3.connect(db_path)

    query = """
        SELECT
            s.sample_id      AS sample,
            p.patient_id,
            p.project,
            p.subject,
            p.condition,
            p.age,
            p.sex,
            s.treatment,
            s.response,
            s.sample_type,
            ct.cell_type     AS population,
            m.count,
            SUM(m.count) OVER (PARTITION BY s.sample_id) AS total_count
        FROM measurements m
        JOIN samples s      ON m.sample_id = s.sample_id
        JOIN patients p    ON s.patient_id = p.patient_id
        JOIN cell_types ct ON m.cell_type_id = ct.cell_type_id
        ORDER BY s.sample_id, ct.cell_type;
    """

    df = pd.read_sql_query(query, con)
    con.close()

    # Relative frequency (percentage) – round to 2 decimals for readability
    df["percentage"] = (df["count"] / df["total_count"] * 100).round(2)

    # Keep only the columns we need downstream
    df = df[
        [
            "sample",
            "total_count",
            "population",
            "count",
            "percentage",
            "treatment",
            "response",
            "sample_type",
        ]
    ]
    return df


def load_from_csv(csv_path: pathlib.Path, db_path: pathlib.Path) -> pd.DataFrame:
    """
    Load the frequency CSV produced by Part 2 and enrich it with the
    `response` and `sample_type` columns from the SQLite DB.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Frequency CSV not found: {csv_path}")

    # Load the CSV (it already has the percentages)
    freq_df = pd.read_csv(csv_path)

    # Pull the missing metadata from the DB
    meta = pd.read_sql_query(
        """
        SELECT sample_id AS sample,
               response,
               sample_type
        FROM samples
        """,
        sqlite3.connect(db_path),
    )

    # Merge – left join keeps every row from the CSV
    merged = freq_df.merge(meta, on="sample", how="left")

    # Sanity check: we should have a response for every sample
    if merged["response"].isnull().any():
        missing = merged.loc[merged["response"].isnull(), "sample"].unique()
        raise ValueError(
            f"The following samples have no response information in the DB: {missing}"
        )
    return merged


# Filter to PBMC samples only
def filter_pbmc(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where sample_type == 'PBMC' (case‑insensitive)."""
    mask = df["sample_type"].str.upper() == "PBMC"
    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError("No PBMC samples found after filtering.")
    return filtered


# Statistical comparison
def compare_groups(
    df: pd.DataFrame,
    test: str = "mannwhitney",
    adj_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Perform a group‑wise test for every immune‑cell population.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: `population`, `percentage`, `response`
        (`response` should be "yes" or "no").
    test : {"mannwhitney", "ttest"}
        Which statistical test to apply.
    adj_method : {"fdr_bh", "bonferroni", "holm", ...}
        Multiple‑testing correction method (statsmodels.multipletests).

    Returns
    -------
    results : DataFrame
        Columns:
            population
            median_yes
            median_no
            effect_size   (rank‑bisector for MWU, Cohen's d for t‑test)
            p_raw
            p_adj
            reject_null   (True if q < 0.05)
    """
    populations = df["population"].unique()
    rows = []

    for pop in populations:
        sub = df[df["population"] == pop]

        # Split into responder / non‑responder
        yes = sub[sub["response"] == "yes"]["percentage"].values
        no = sub[sub["response"] == "no"]["percentage"].values

        # Skip if one of the groups is empty (should not happen in a well‑formed trial)
        if len(yes) == 0 or len(no) == 0:
            continue

        med_yes = np.median(yes)
        med_no = np.median(no)

        if test == "mannwhitney":
            stat, p_raw = stats.mannwhitneyu(yes, no, alternative="two-sided")
            # Rank‑bisector effect size (r) = U / (n1*n2)
            r = stat / (len(yes) * len(no))
            effect = r
        else:  # t‑test
            stat, p_raw = stats.ttest_ind(yes, no, equal_var=False)
            # Cohen's d
            pooled_sd = np.sqrt(
                ((len(yes) - 1) * np.var(yes, ddof=1)
                + (len(no) - 1) * np.var(no, ddof=1))
                / (len(yes) + len(no) - 2)
            )
            effect = (np.mean(yes) - np.mean(no)) / pooled_sd

        rows.append(
            {
                "population": pop,
                "median_yes": med_yes,
                "median_no": med_no,
                "effect_size": effect,
                "p_raw": p_raw,
            }
        )

    results = pd.DataFrame(rows)

    # Multiple‑testing correction
    if not results.empty:
        reject, p_adj, _, _ = multipletests(results["p_raw"], method=adj_method)
        results["p_adj"] = p_adj
        results["reject_null"] = reject
    else:
        results["p_adj"] = []
        results["reject_null"] = []

    results.sort_values("p_adj", inplace=True)
    return results

def clean_response_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalise to lower‑case.
    - Keep only the two allowed values: 'yes' and 'no'.
    - Drop any rows where response is missing or something else.
    """
    # Normalise case and strip whitespace
    df["response"] = df["response"].astype(str).str.strip().str.lower()

    # Keep only the two valid categories
    valid = {"yes", "no"}
    mask = df["response"].isin(valid)

    # Optional: warn you about rows that are being removed
    if (~mask).any():
        dropped = df.loc[~mask, ["sample", "population", "response"]].drop_duplicates()
        print("Rows with unexpected response values will be excluded:")
        print(dropped.to_string(index=False))

    # Return the cleaned dataframe (only valid rows)
    return df.loc[mask].copy()

# Plotting – boxplot of percentages by response
def plot_boxplot(df: pd.DataFrame, out_path: Optional[pathlib.Path] = None) -> None:
    """
    Draw a Seaborn box‑plot:
        x = cell population
        y = relative frequency (%)
        hue = response ("yes"/"no")
    If `out_path` is supplied the figure is saved, otherwise it is shown
    interactively.
    """
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="population",
        y="percentage",
        hue="response",
        palette="Set2",
        notch=True,
    )
    plt.title(
        "Relative frequencies of immune‑cell populations\n"
        "PBMC samples – responders vs. non‑responders"
    )
    plt.ylabel("Relative frequency (%)")
    plt.xlabel("Cell population")
    plt.legend(title="Response", loc="upper right")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Box‑plot saved to {out_path}")
    else:
        plt.show()
    plt.close()

#!/usr/bin/env python
"""
part4.py
--------------------------------
Early‑treatment melanoma PBMC subset (miraclib, baseline).

Because the original schema does NOT contain a `subject` column,
we use `patient_id` (which uniquely identifies a patient) as the
subject identifier for the responder‑/gender‑counts.
"""

import argparse
import pathlib
import sqlite3
import sys
from typing import Tuple

import pandas as pd

# Helper: run a query and return a pandas DataFrame
def sql_to_df(con: sqlite3.Connection,
              query: str,
              params: tuple = ()) -> pd.DataFrame:
    """Execute *query* with *params* and return the result as a DataFrame."""
    return pd.read_sql_query(query, con, params=params)


# Pull the baseline melanoma PBMC subset
def get_baseline_melanoma_pbmc(con: sqlite3.Connection) -> pd.DataFrame:
    """
    Returns rows that satisfy:
        • condition = 'melanoma'
        • sample_type = 'PBMC'
        • time_from_treatment_start = 0
        • treatment = 'miraclib'

    The column `patient_id` is used as the subject identifier.
    """
    query = """
        SELECT
            s.sample_id,
            s.treatment,
            s.response,
            s.sample_type,
            s.time_from_treatment_start,
            p.project,                -- if you added this column; otherwise it will be NULL
            s.patient_id AS subject,  -- <-- use patient_id as the subject
            p.sex,
            p.age,
            p.condition
        FROM samples   s
        JOIN patients  p ON s.patient_id = p.patient_id
        WHERE
            LOWER(p.condition) = 'melanoma'
            AND LOWER(s.sample_type) = 'pbmc'
            AND s.time_from_treatment_start = 0
            AND LOWER(s.treatment) = 'miraclib';
    """
    return sql_to_df(con, query)


# Summaries
def samples_per_project(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct sample IDs for each project (if the column exists)."""
    if "project" not in df.columns:
        # No project column – return an empty DataFrame with a friendly message
        return pd.DataFrame(columns=["project", "n_samples"])
    return (
        df.groupby("project")["sample_id"]
        .nunique()
        .reset_index(name="n_samples")
        .sort_values("n_samples", ascending=False)
    )


def responders_vs_non(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count *subjects* (patient_id) that are responders (response='yes')
    vs. non‑responders (response='no').
    """
    # One row per subject – response is constant per subject in our schema
    subj = (
        df.groupby("subject")["response"]
        .first()
        .reset_index()
    )
    return (
        subj.groupby("response")["subject"]
        .nunique()
        .reset_index(name="n_subjects")
        .rename(columns={"response": "Responder (yes/no)"})
    )


def gender_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count *subjects* by reported sex (M/F/Other).  Sex lives in patients.sex.
    """
    subj = df[["subject", "sex"]].drop_duplicates()
    return (
        subj.groupby("sex")["subject"]
        .nunique()
        .reset_index(name="n_subjects")
        .rename(columns={"sex": "Sex (M/F/…)"})
    )

#!/usr/bin/env python
import pathlib
import sqlite3

def main(argv: Tuple[str, ...]) -> None:
    """
    Orchestrates the entire Loblaw immunology workflow:

     Load the raw CSV → create the SQLite database.
     Compute per‑sample cell‑type frequencies (writes frequencies.csv).
     Perform the PBMC statistical comparison (writes stat_comparison.csv
       and saves a box‑plot PNG).
     Summarise the baseline melanoma‑PBMC subset (writes three CSV tables).
     Query the average B‑cell count for melanoma‑male responders at baseline
       and print the result.

    Expected usage:
        python loblaw_full_pipeline.py <path/to/cell_counts.csv> <output_db.sqlite>
    """
    # Argument parsing
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Loblaw immunology pipeline (parts 1‑5) in one go. "
            "Outputs: SQLite DB, frequencies.csv, statistical table, box‑plot, "
            "melanoma summary tables, and average B‑cell value."
        )
    )
    parser.add_argument(
        "csv_path",
        type=pathlib.Path,
        help="Path to the raw cell‑counts CSV (e.g. cell_counts.csv)",
    )
    parser.add_argument(
        "db_path",
        type=pathlib.Path,
        help="Desired output SQLite DB file (e.g. loblaw_study.db)",
    )
    args = parser.parse_args(argv[1:])

    # Step 1 – Load CSV → SQLite (Part 1)
    print("\n=== STEP 1 – Load CSV → SQLite ===")
    load_csv(args.csv_path, args.db_path)

    # Step 2 – Compute frequencies (Part 2)
    print("\n=== STEP 2 – Compute per‑sample frequencies ===")
    flat_df = load_flat_dataframe(args.db_path)
    freq_df = compute_frequencies(flat_df)
    freq_csv = pathlib.Path("frequencies.csv")
    freq_df.to_csv(freq_csv, index=False)
    print(f"Frequencies written to {freq_csv}")

    # Step 3 – Statistical comparison (Part 3)
    print("\n=== STEP 3 – Statistical comparison (PBMC) ===")
    full_df = load_from_db(args.db_path)
    pbmc_df = filter_pbmc(full_df)
    pbmc_df = clean_response_column(pbmc_df)
    stats_df = compare_groups(pbmc_df, test="mannwhitney", adj_method="fdr_bh")
    stats_csv = pathlib.Path("stat_comparison.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Stats table written to {stats_csv}")

    # Save the box‑plot
    boxplot_path = pathlib.Path("pbmc_boxplot.png")
    plot_boxplot(pbmc_df, out_path=boxplot_path)

    # Step 4 – Melanoma PBMC baseline summary (Part 4)
    print("\n=== STEP 4 – Melanoma PBMC baseline summary ===")
    con = sqlite3.connect(args.db_path)

    mel_df = get_baseline_melanoma_pbmc(con)

    # 4a – samples per project
    proj_tbl = samples_per_project(mel_df)
    proj_csv = pathlib.Path("melanoma_samples_per_project.csv")
    proj_tbl.to_csv(proj_csv, index=False)
    print(f"Samples‑per‑project table → {proj_csv}")

    # 4b – responders vs. non‑responders (subjects)
    resp_tbl = responders_vs_non(mel_df)
    resp_csv = pathlib.Path("melanoma_responders_vs_non.csv")
    resp_tbl.to_csv(resp_csv, index=False)
    print(f"Responder counts → {resp_csv}")

    # 4c – gender distribution
    gender_tbl = gender_counts(mel_df)
    gender_csv = pathlib.Path("melanoma_gender_counts.csv")
    gender_tbl.to_csv(gender_csv, index=False)
    print(f"Gender counts → {gender_csv}")

    con.close()

    # Step 5 – Average B‑cell query (Part 5)
    # print("\n=== STEP 5 – Average B‑cell count (melanoma‑male responders) ===")
    # avg_val = avg_bcell_for_melanoma_male_responders(args.db_path)
    # if avg_val is None:
    #     print("ℹ️ No rows matched the filter criteria – check your data.")
    # else:
    #     print(f"✅ Average B‑cell count = {avg_val:.2f}")

    # print("\n🚀 All steps completed successfully!")


# Settings – change these paths / values if needed
DB_PATH = pathlib.Path("loblaw_study.db")
# If your B‑cell column is named differently (e.g. "B_cell"), adjust below
TARGET_CELL_TYPE = "b_cell"

# Build and run the query
query = f"""
SELECT AVG(m.count) AS avg_b_cells
FROM measurements   m
JOIN samples       s ON m.sample_id      = s.sample_id
JOIN patients      p ON s.patient_id     = p.patient_id
JOIN cell_types    ct ON m.cell_type_id  = ct.cell_type_id
WHERE
    LOWER(p.condition)               = 'melanoma'          -- disease
    AND LOWER(p.sex)                 = 'm'                 -- male
    AND LOWER(s.response)            = 'yes'               -- responder
    AND s.time_from_treatment_start = 0                   -- baseline
    AND LOWER(ct.cell_type)          = LOWER(?);           -- B‑cell population
"""

if __name__ == "__main__":
    main(sys.argv)

# Open the DB, execute the query, fetch the scalar result
with sqlite3.connect(DB_PATH) as con:
    cur = con.cursor()
    cur.execute(query, (TARGET_CELL_TYPE,))
    result = cur.fetchone()[0]   # AVG() returns a single float (or None)

# Present the answer (two‑decimal formatting)
if result is None:
    print("No rows matched the filter criteria – check your data.")
else:
    # Round to two decimals as requested
    avg_rounded = round(result, 2)
    # Ensure we always show two digits after the decimal point

    print(f"Average B‑cell count for melanoma‑male responders at time = 0: {avg_rounded:.2f}")


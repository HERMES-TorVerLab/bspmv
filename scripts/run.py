import os
import csv
import re
import subprocess
from pathlib import Path
import questionary
import pandas as pd
import glob    
from collections import defaultdict

from matrix_macros import *
from dataset import *

# --- Config ---
BUILD_DIR = Path("../build")
MATRIX_DIR = Path("../matrix")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)  # ensure folder exists
CSV_FILE = RESULT_DIR / "results.csv"

EXEC_MODES = ["GPU"]
MATRIX_FORMATS = ["COO", "CSR", "ELL", "HLL", "BWC_COO", "BWC_CSR", "BWC_ELL", "BWC_HLL"]

THREADS_PER_BLOCK = [512]
WARP_SIZE = [4]



# --- Utilities ---
def grep(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def run_single_experiment(fmt, exec_mode, matrix_name):
    log_file_path = BUILD_DIR / Path("single_experiment.log")
    matrix_input_file = MATRIX_DIR / matrix_name

    if not log_file_path.exists():
        log_file_path.touch()
        print(f"[INFO]\tCreated new log file: {log_file_path}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file = RESULT_DIR / f"{matrix_name}_{fmt}_{exec_mode}.csv"

    # Run command
    cmd = f'./bSpMV --matrix-file="{matrix_input_file}" --matrix-format="{fmt}" --exec-mode="{exec_mode}" > "{log_file_path}"'
    subprocess.run(cmd, shell=True, cwd=BUILD_DIR)

    text = log_file_path.read_text()

    # Extract fields
    matrix_file = os.path.splitext(os.path.basename(
        grep(r"Matrix file:\s*(.*)", text, matrix_name)))[0]
    matrix_format = grep(r"Matrix format:\s*(.*)", text, fmt)
    exec_mode_val = grep(r"Execution mode:\s*(.*)", text, exec_mode)
    threads = grep(r"Threads per block:\s*(.*)", text, str(THREADS_PER_BLOCK[0]))
    warp_size_val = grep(r"Warp size:\s*(.*)", text, str(WARP_SIZE[0]))
    kernel_launches = grep(r"Kernel launches:\s*(.*)", text, "0")
    rows = grep(r"Rows:\s*(.*)", text, "0")
    cols = grep(r"Cols:\s*(.*)", text, "0")
    nonzeros = grep(r"Non-zeros:\s*(.*)", text, "0")
    mem = grep(r"Memory \(MB\):\s*(.*)", text, "0")

    num_word = min_nnz_word = max_nnz_word = avg_nnz_word = "0"
    if matrix_format.startswith("BWC"):
        num_word = grep(r"Num words:\s*(.*)", text, "0")
        min_nnz_word = grep(r"Min NNZ/word:\s*(.*)", text, "0")
        max_nnz_word = grep(r"Max NNZ/word:\s*(.*)", text, "0")
        avg_nnz_word = grep(r"Avg NNZ/word:\s*(.*)", text, "0")

    # Print results instead of writing to CSV
    print("Experiment Results")
    print("------------------------------")
    print(f"Matrix       : {matrix_file}")
    print(f"Format       : {matrix_format}")
    print(f"ExecMode     : {exec_mode_val}")
    print(f"Threads      : {threads}")
    print(f"Warp Size    : {warp_size_val}")
    print(f"Kernel Runs  : {kernel_launches}")
    print(f"Rows x Cols  : {rows} x {cols}")
    print(f"Nonzeros     : {nonzeros}")
    print(f"Memory (MB)  : {mem}")

    if matrix_format in ["CSR", "BWC_CSR"]:
        scalar_time = grep(r"Scalar computation time:\s*(.*)", text, "0.00000")
        vector_time = grep(r"Vector computation time:\s*(.*)", text, "0.00000")
        print(f"Scalar Time  : {scalar_time}")
        print(f"Vector Time  : {vector_time}")
    else:
        scalar_time = grep(r"Computation time:\s*(.*)", text, "0.00000")
        print(f"Time         : {scalar_time}")

    if matrix_format.startswith("BWC"):
        print(f"Num Words    : {num_word}")
        print(f"Min NNZ/word : {min_nnz_word}")
        print(f"Max NNZ/word : {max_nnz_word}")
        print(f"Avg NNZ/word : {avg_nnz_word}")

    print("------------------------------\n Single experiment completed.\n")



def run_experiments(log_file):
    log_file_path = BUILD_DIR / Path(log_file)

    if not log_file_path.exists():
        log_file_path.touch()
        print(f"[INFO]\tCreated new log file: {log_file_path}")

    # Ensure results folder exists
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file_path = CSV_FILE  # results/experiment_results.csv

    # --- Guard for existing CSV ---
    if csv_file_path.exists():
        action = questionary.select(
            f"[WARNING] Result file '{csv_file_path}' already exists. What do you want to do?",
            choices=[
                "Append to existing file",
                "Overwrite file",
                "Stop computation"
            ]
        ).ask()

        if action == "Stop computation":
            print("Computation stopped by user.")
            return
        elif action == "Overwrite file":
            csv_file_path.unlink()  # remove existing file
            print(f"[INFO] Overwriting file: {csv_file_path}")

    # Write CSV header if file doesn't exist (new or overwritten)
    if not csv_file_path.exists():
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "MatrixFile","MatrixFormat","ExecMode","Threads","WarpSize","KernelLaunches",
                "Rows","Cols","NonZeros","NumWord","MinNnzWord","MaxNnzWord","AvgNnzWord","Mem","Time"
            ])


    # Use defaults
    formats = MATRIX_FORMATS
    exec_modes = EXEC_MODES

    for matrix_name in MATRIX_FILES:
        matrix_input_file = MATRIX_DIR / matrix_name
        for exec_mode in exec_modes:
            for fmt in formats:
                print(f"Running: File={matrix_name} Format={fmt}, Exec={exec_mode}")
                cmd = f'./bSpMV --matrix-file="{matrix_input_file}" --matrix-format="{fmt}" --exec-mode="{exec_mode}" > "{log_file_path}"'
                subprocess.run(cmd, shell=True, cwd=BUILD_DIR)

                text = log_file_path.read_text()

                # Extract results
                matrix_file = os.path.splitext(os.path.basename(
                    grep(r"Matrix file:\s*(.*)", text, matrix_name)))[0]
                matrix_format = grep(r"Matrix format:\s*(.*)", text, fmt)
                exec_mode_val = grep(r"Execution mode:\s*(.*)", text, exec_mode)
                threads = grep(r"Threads per block:\s*(.*)", text, str(THREADS_PER_BLOCK[0]))
                warp_size_val = grep(r"Warp size:\s*(.*)", text, str(WARP_SIZE[0]))
                kernel_launches = grep(r"Kernel launches:\s*(.*)", text, "0")
                rows = grep(r"Rows:\s*(.*)", text, "0")
                cols = grep(r"Cols:\s*(.*)", text, "0")
                nonzeros = grep(r"Non-zeros:\s*(.*)", text, "0")
                mem = grep(r"Memory \(MB\):\s*(.*)", text, "0")

                num_word = min_nnz_word = max_nnz_word = avg_nnz_word = "0"
                if matrix_format.startswith("BWC"):
                    num_word = grep(r"Num words:\s*(.*)", text, "0")
                    min_nnz_word = grep(r"Min NNZ/word:\s*(.*)", text, "0")
                    max_nnz_word = grep(r"Max NNZ/word:\s*(.*)", text, "0")
                    avg_nnz_word = grep(r"Avg NNZ/word:\s*(.*)", text, "0")

                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if matrix_format in ["CSR", "BWC_CSR"]:
                        scalar_time = grep(r"Scalar computation time:\s*(.*)", text, "0.00000")
                        vector_time = grep(r"Vector computation time:\s*(.*)", text, "0.00000")
                        writer.writerow([matrix_file,matrix_format,exec_mode_val,threads,warp_size_val,
                                         kernel_launches,rows,cols,nonzeros,num_word,min_nnz_word,
                                         max_nnz_word,avg_nnz_word,mem,scalar_time])
                        writer.writerow([matrix_file,matrix_format,exec_mode_val,threads,warp_size_val,
                                         kernel_launches,rows,cols,nonzeros,num_word,min_nnz_word,
                                         max_nnz_word,avg_nnz_word,mem,vector_time])
                    else:
                        scalar_time = grep(r"Computation time:\s*(.*)", text, "0.00000")
                        writer.writerow([matrix_file,matrix_format,exec_mode_val,threads,warp_size_val,
                                         kernel_launches,rows,cols,nonzeros,num_word,min_nnz_word,
                                         max_nnz_word,avg_nnz_word,mem,scalar_time])

    print(f" Saved data to {CSV_FILE}")




def select_results_csv():
    results_dir = Path("results")
    if not results_dir.exists():
        print("[ERROR] results/ directory does not exist.")
        return None

    # List CSV files that do NOT start with "_"
    csv_files = [
        f.name for f in results_dir.glob("*.csv")
        if not f.name.startswith("_")
    ]

    if not csv_files:
        print("[ERROR] No valid CSV files found in results/ directory.")
        return None

    # Let user select a file
    selected_csv = questionary.select(
        "Select CSV file to analyze:",
        choices=csv_files
    ).ask()

    return results_dir / selected_csv

# -------------------------
# Helpers
# -------------------------
def fill_matrix_format(df):
    """
    Fill missing MatrixFormat entries in a round-robin sequence per matrix.
    """
    sequence = [
        "COO", "CSRs", "CSRv", "ELL", "HLL",
        "BWC_COO", "BWC_CSRs", "BWC_CSRv", "BWC_ELL", "BWC_HLL"
    ]

    df = df.copy()

    for matrix, group in df.groupby("MatrixFile", sort=False):
        idx = 0
        for i in group.index:
            val = df.at[i, "MatrixFormat"]
            if pd.isna(val) or str(val).strip() == "":
                df.at[i, "MatrixFormat"] = sequence[idx % len(sequence)]
                idx += 1
            else:
                if val in sequence:
                    idx = (sequence.index(val) + 1) % len(sequence)
                else:
                    idx += 1  # unknown format, still advance
    return df


def normalize_matrix_name(name: str) -> str:
    """
    Ensure matrix names are normalized (c30.sparse -> c30.sparse.bin, others -> .mtx if missing)
    """
    if pd.isna(name):
        return ""
    if name.startswith("c") and ".sparse" in name and not name.endswith(".bin"):
        return name + ".bin"
    if not name.endswith(".mtx") and not name.endswith(".bin"):
        return name + ".mtx"
    return name


def add_problem_type_to_stats(csv_path: Path):
    """
    Add ProblemType and MainType to results CSV and normalize MatrixFile.
    """
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"[INFO] CSV {csv_path} loaded with {len(df)} rows.")

    # Normalize MatrixFile names
    df['MatrixFile'] = df['MatrixFile'].astype(str).apply(normalize_matrix_name)

    # Assign ProblemType / MainType from dictionaries
    problem_type_list = []
    main_type_list = []
    for norm_name in df['MatrixFile']:
        if norm_name in MATRIX_TYPES:
            problem_type = MATRIX_TYPES[norm_name]
            main_type = MAIN_TYPE.get(problem_type, "Unknown")
        else:
            problem_type = "Unknown"
            main_type = "Unknown"
        problem_type_list.append(problem_type)
        main_type_list.append(main_type)

    df['ProblemType'] = problem_type_list
    df['MainType'] = main_type_list

    # Keep only first part of MatrixFile (before extension)
    df['MatrixFile'] = df['MatrixFile'].str.split('.', n=1).str[0]

    # Save updated CSV
    df.to_csv(RESULT_DIR / "_results_type.csv", index=False)
    print(f"[INFO] Updated results saved to '_results_type.csv'")

    return df


def split_csr_variants(df):
    df = df.copy()

    # Process normal CSR
    for matrix, group in df[df["MatrixFormat"] == "CSR"].groupby("MatrixFile", sort=False):
        group_sorted = group.sort_values(by="Time", ascending=True)
        idxs = group_sorted.index.tolist()
        if len(idxs) >= 1:
            df.at[idxs[0], "MatrixFormat"] = "CSRs"
        if len(idxs) >= 2:
            df.at[idxs[1], "MatrixFormat"] = "CSRv"
        for j, idx in enumerate(idxs[2:], start=2):
            df.at[idx, "MatrixFormat"] = f"CSR_run{j}"

    # Process BWC_CSR
    for matrix, group in df[df["MatrixFormat"] == "BWC_CSR"].groupby("MatrixFile", sort=False):
        group_sorted = group.sort_values(by="Time", ascending=True)
        idxs = group_sorted.index.tolist()
        if len(idxs) >= 1:
            df.at[idxs[0], "MatrixFormat"] = "BWC_CSRs"
        if len(idxs) >= 2:
            df.at[idxs[1], "MatrixFormat"] = "BWC_CSRv"
        for j, idx in enumerate(idxs[2:], start=2):
            df.at[idx, "MatrixFormat"] = f"BWC_CSR_run{j}"

    return df

# -------------------------
# Main Analysis Function
# -------------------------
def analyze_results(csv_path: Path):
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    # Step 0: Add ProblemType / MainType
    df = add_problem_type_to_stats(csv_path)
    if df is None:
        return

    EXPECTED_COLUMNS = [
        "MatrixFile","MatrixFormat","ExecMode","Threads","WarpSize","KernelLaunches",
        "Rows","Cols","NonZeros","NumWord","MinNnzWord","MaxNnzWord","AvgNnzWord","Mem","Time",
        "ProblemType","MainType"
    ]

    # Fill missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col not in ["ProblemType","MainType"] else "Unknown"

    # Ensure numeric columns
    numeric_cols = ["Rows","Cols","NonZeros","NumWord","MinNnzWord","MaxNnzWord","AvgNnzWord","Mem","Time"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fill missing MatrixFormat
    df = fill_matrix_format(df)

    # Split CSR → CSRs / CSRv
    df = split_csr_variants(df)

    # Save full DataFrame for reference
    df.to_csv(RESULT_DIR / "_results_type.csv", index=False)
    print(f"[INFO] Full DataFrame with CSRs/CSRv saved in '_results_type.csv'")

    # -------------------------
    # Step 1: Compute baseline memory and GNNZ
    # -------------------------
    baseline_formats = ["COO", "CSRs", "CSRv", "ELL", "HLL"]
    baseline_mem = defaultdict(dict)
    baseline_gnnz = defaultdict(dict)

    for fmt in baseline_formats:
        df_fmt = df[df['MatrixFormat'] == fmt]
        for matrix, group in df_fmt.groupby("MatrixFile"):
            group_sorted = group.sort_values(by="Time")
            baseline_mem[fmt][matrix] = group_sorted['Mem'].tolist()
            baseline_gnnz[fmt][matrix] = [
                row['NonZeros'] / ((row['Time']/1000)*1e9) if row['Time'] > 0 else 0
                for _, row in group_sorted.iterrows()
            ]

    # -------------------------
    # Step 2: Process BWC formats
    # -------------------------
    bwc_df = df[df['MatrixFormat'].str.startswith("BWC_", na=False)]
    bwc_baseline_formats = ["BWC_COO", "BWC_CSRs", "BWC_CSRv", "BWC_ELL", "BWC_HLL"]

    for fmt in bwc_baseline_formats:
        df_fmt = bwc_df[bwc_df['MatrixFormat'] == fmt]
        base_fmt = fmt.replace("BWC_", "")

        records = []
        for _, row in df_fmt.iterrows():
            matrix = row['MatrixFile']
            mem_bwc = row['Mem']
            avg_nnzw = row.get('AvgNnzWord', 0)

            # Use first run of baseline if multiple CSRs/CSRv
            mem_base_list = baseline_mem.get(base_fmt, {}).get(matrix, [mem_bwc])
            mem_base = mem_base_list[0] if mem_base_list else mem_bwc

            gnnz_base_list = baseline_gnnz.get(base_fmt, {}).get(matrix, [0])
            gnnz_base = gnnz_base_list[0] if gnnz_base_list else 0

            # Memory reduction
            mem_reduction = mem_base / mem_bwc if mem_bwc > 0 else 0

            # Speedup computation
            gnnz_bwc = row['NonZeros'] / ((row['Time']/1000)*1e9) if row['Time'] > 0 else 0
            speedup = gnnz_bwc / gnnz_base if gnnz_base > 0 else 0

            records.append({
                'MatrixFile': matrix,
                'AvgNnzWord': avg_nnzw,
                'MemReduction': mem_reduction,
                'Speedup': speedup,
                'GNNZ_Base': gnnz_base,
                'GNNZ_BWC': gnnz_bwc,
                'ProblemType': row.get('ProblemType','Unknown'),
                'MainType': row.get('MainType','Unknown')
            })

        if records:
            summary_df = pd.DataFrame(records)
            summary_csv_path = RESULT_DIR / f"_stats_{fmt.lower()}.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"[INFO] Summary for {fmt} saved in {summary_csv_path}")

    print("[INFO] Analysis completed for all BWC formats.")





# Define top-N function
def get_top_matrices(stats_df, filter_col, filter_value, top_x=2):
    formats = sorted(stats_df['Format'].unique())  # ensure consistent order
    filtered_stats = stats_df[stats_df[filter_col] == filter_value]
    if filtered_stats.empty:
        print(f"[WARN] No matrices found for {filter_col}={filter_value}")
        return pd.DataFrame()
    
    # Compute max speedup for each matrix
    matrix_max_speedup = filtered_stats.groupby('MatrixFile')['Speedup'].max().reset_index()
    top_matrices_list = matrix_max_speedup.sort_values(by='Speedup', ascending=False).head(top_x)['MatrixFile']
    top_stats = filtered_stats[filtered_stats['MatrixFile'].isin(top_matrices_list)]

    # Pivot each metric
    def pivot_metric(metric, prefix):
        pivot = top_stats.pivot(index='MatrixFile', columns='Format', values=metric)
        pivot = pivot.reindex(columns=formats).add_prefix(f"{prefix}_").fillna(0)
        return pivot
    
    speedup_pivot = pivot_metric("Speedup", "Speedup")
    nnz_pivot = pivot_metric("AvgNnzWord", "AvgNnzWord")
    mem_pivot = pivot_metric("MemReduction", "MemReduction")
    pivot_table = speedup_pivot.join(nnz_pivot).join(mem_pivot).reset_index()
    return pivot_table


# ------------------------- 
# Top matrices summary 
# -------------------------
def generate_top_matrices_summary():
    # Step 1: Load all available stats CSVs
    files = sorted(glob.glob(f"{RESULT_DIR}/_stats_bwc_*.csv"))
    if not files:
        print("[ERROR] No _stats_bwc_*.csv files found in RESULT_DIR")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['Format'] = f.split('_')[-1].split('.')[0]  # extract format name
            dfs.append(df)
        except FileNotFoundError:
            print(f"[WARN] File not found: {f}")
        except pd.errors.EmptyDataError:
            print(f"[WARN] Empty CSV: {f}")

    stats_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined stats dataframe has {len(stats_df)} rows")

    # Step 3: Queries
    queries = [
        ('ProblemType', 'Combinatorics', 1, 'Combinatorial'),
        ('MainType', 'undirected graph', 3, 'Undirected Graph'),
        ('MainType', 'directed graph', 3, 'Directed Graph'),
        ('MainType', 'random graph', 1, 'Random Graph'),
        ('MainType', 'structural', 5, 'Structural'),
        ('MainType', 'integer', 2, 'Integer Factorization')
    ]

    all_results = []
    for filter_col, filter_value, top_x, category_name in queries:
        top_matrices = get_top_matrices(stats_df, filter_col, filter_value, top_x=top_x)
        if not top_matrices.empty:
            top_matrices['Category'] = category_name
            all_results.append(top_matrices)

    if all_results:
        final_table = pd.concat(all_results, ignore_index=True)
        cols = ['Category', 'MatrixFile'] + [c for c in final_table.columns if c not in ['Category','MatrixFile']]
        final_table = final_table[cols]
        final_table.to_csv(f"{RESULT_DIR}/_top_matrices_summary.csv", index=False)
        print("Final table saved as '_top_matrices_summary.csv'")
    else:
        print("[WARN] No top matrices found for the given queries")


def check_missing_matrices(base_dir=MATRIX_DIR, matrix_list=MATRIX_FILES):
    """
    Check which matrices are missing from the local dataset folder.

    Args:
        base_dir (Path or str): directory where matrices are stored
        matrix_list (list): list of matrix filenames

    Prints a summary of missing and present matrices.
    """
    base_dir = Path(base_dir)
    missing_count = 0
    total = len(matrix_list)

    print("Checking for missing matrices...\n")

    for matrix in matrix_list:
        # Convert commas to dots if needed, like your bash script
        file_name = matrix.replace(",", ".")
        file_path = base_dir / file_name
        if not file_path.exists():
            print(f"Missing: {file_name}")
            missing_count += 1

    print("\nSummary:")
    print(f"Total files expected: {total}")
    print(f"Files missing: {missing_count}")
    print(f"Files present: {total - missing_count}")




def best15(result_dir: Path, output_csv="_best15_matrices_all_formats.csv"):
    # -----------------------------
    # 1) Load all stats files
    # -----------------------------
    files = [
        "_stats_bwc_coo.csv",
        "_stats_bwc_csrs.csv",
        "_stats_bwc_csrv.csv",
        "_stats_bwc_ell.csv",
        "_stats_bwc_hll.csv"
    ]

    dfs = []
    for f in files:
        path = result_dir / f
        if not path.exists():
            print(f"[WARN] Stats file {f} not found, skipping.")
            continue

        df = pd.read_csv(path)

        # Correct: extract the kernel format from the filename
        # filename: _stats_bwc_coo.csv
        # split('_') -> ['', 'stats', 'bwc', 'coo.csv']
        kernel = f.split('_')[3].split('.')[0].upper()  # index 3 is the kernel name
        df['Format'] = kernel
        dfs.append(df)

    if not dfs:
        print("[ERROR] No stats files loaded. Exiting.")
        return

    # -----------------------------
    # 2) Merge all stats
    # -----------------------------
    merged = pd.concat(dfs, ignore_index=True)

    # Normalize MatrixFile names (remove extensions)
    merged['MatrixFileClean'] = merged['MatrixFile'].str.replace(
        r'\.mtx$|\.bin$|\.sparse$', '', regex=True
    )

    # -----------------------------
    # 3) Compute max speedup per matrix
    # -----------------------------
    max_speedup = merged.groupby('MatrixFileClean')['Speedup'].max().reset_index()
    max_speedup = max_speedup[max_speedup['Speedup'] > 1]  # keep only matrices with max > 1

    # Top 15 matrices by max speedup
    top15_matrices = max_speedup.sort_values(by='Speedup', ascending=False)['MatrixFileClean'].head(15)

    # -----------------------------
    # 4) Filter stats for top 15
    # -----------------------------
    top15_stats = merged[merged['MatrixFileClean'].isin(top15_matrices)].copy()  # make a copy to avoid warnings

    # Add a column to preserve the order of top matrices
    top15_stats['MatrixOrder'] = top15_stats['MatrixFileClean'].apply(lambda x: list(top15_matrices).index(x))

    # Sort by matrix order, then by Speedup descending
    top15_stats = top15_stats.sort_values(['MatrixOrder', 'Speedup'], ascending=[True, False]).drop(columns=['MatrixOrder'])

    # -----------------------------
    # 5) Save final CSV
    # -----------------------------
    output_path = result_dir / output_csv
    top15_stats.to_csv(output_path, index=False)
    print(f"[INFO] Wrote results for top 15 matrices (by max Speedup > 1) to: {output_path}")


def best15_by_mem(result_dir: Path, output_csv="_best15_matrices_mem.csv"):
    import pandas as pd

    # -----------------------------
    # 1) Load all stats files
    # -----------------------------
    files = [
        "_stats_bwc_coo.csv",
        "_stats_bwc_csrs.csv",
        "_stats_bwc_csrv.csv",
        "_stats_bwc_ell.csv",
        "_stats_bwc_hll.csv"
    ]

    dfs = []
    for f in files:
        path = result_dir / f
        if not path.exists():
            print(f"[WARN] Stats file {f} not found, skipping.")
            continue

        df = pd.read_csv(path)

        # Correctly extract kernel name from filename
        kernel = f.split('_')[3].split('.')[0].upper()  # index 3 is kernel (coo, csrs, csrv, ell, hll)
        df['Format'] = kernel
        dfs.append(df)

    if not dfs:
        print("[ERROR] No stats files loaded. Exiting.")
        return

    # -----------------------------
    # 2) Merge all stats
    # -----------------------------
    merged = pd.concat(dfs, ignore_index=True)
    merged['MatrixFileClean'] = merged['MatrixFile'].str.replace(
        r'\.mtx$|\.bin$|\.sparse$', '', regex=True
    )

    # -----------------------------
    # 3) Compute max memory reduction per matrix
    # -----------------------------
    max_mem = merged.groupby('MatrixFileClean')['MemReduction'].max().reset_index()
    max_mem = max_mem[max_mem['MemReduction'] > 1]  # keep only matrices where memory was actually reduced

    # Top 15 matrices by max memory reduction
    top15_matrices = max_mem.sort_values(by='MemReduction', ascending=False)['MatrixFileClean'].head(15)

    # -----------------------------
    # 4) Filter stats for top 15
    # -----------------------------
    top15_stats = merged[merged['MatrixFileClean'].isin(top15_matrices)].copy()
    top15_stats['MatrixOrder'] = top15_stats['MatrixFileClean'].apply(lambda x: list(top15_matrices).index(x))

    # Sort by matrix order, then by MemReduction descending
    top15_stats = top15_stats.sort_values(['MatrixOrder', 'MemReduction'], ascending=[True, False]).drop(columns=['MatrixOrder'])

    # -----------------------------
    # 5) Save final CSV
    # -----------------------------
    output_path = result_dir / output_csv
    top15_stats.to_csv(output_path, index=False)
    print(f"[INFO] Wrote results for top 15 matrices (by max MemReduction > 1) to: {output_path}")


def print_welcome_message():
    print(r"""=========================================================
         _      _____       ___  ___ __       __
        | |    / ____/     |   \/   |\ \     / /
        | |__ | (___  ____ | |\  /| | \ \   / / 
        | '_ \ \___ \|  _ \| | \/ | |  \ \ / /  
        | |_) |____) | |_| | |    | |   \   /
        |_.__/|_____/|  __/|_|    |_|    \_/ 
                     | |
                     |_|

bSpMV Experiment Program v1.0
Binary Sparse Matrix-Vector Multiplication Experiments
Author: Simone Staccone, University of Rome 'Tor Vergata'

DICII Department of Civil and Computer Engineering © 2025
DAMON Research Group
=========================================================
""")


# -------------------------
# Main CLI
# -------------------------
def main():
    
    os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        if Path("results/_results_type.csv").is_file():
            choices=[
                "Unpack Matrices Dataset",
                "Check Missing Matrices", 
                "Run Single Experiment (custom selection)",
                "Run All Experiments",
                "Analyze Results",
                "Generate Top Matrices Summary",
                "Best 15 Matrices Summary",  
                "Exit"
            ]
        else:
            choices=[
                "Unpack Matrices Dataset",
                "Check Missing Matrices", 
                "Run Single Experiment (custom selection)",
                "Run All Experiments",
                "Analyze Results",
                "Exit"
            ]

        choice = questionary.select("Choose an action:", choices).ask()

        if choice == "Unpack Matrices Dataset":
            unpack_matrices()

        elif choice == "Run Single Experiment (custom selection)":
            formats = questionary.select("Select matrix formats:", choices=MATRIX_FORMATS).ask()
            exec_modes = questionary.select("Select execution modes:", choices=EXEC_MODES+["CPU"]).ask()
            matrix = questionary.select("Select matrix to run:", choices=MATRIX_FILES).ask()
            run_single_experiment(formats, exec_modes, matrix)

        elif choice == "Check Missing Matrices":
            check_missing_matrices()


        elif choice == "Run All Experiments":
            log_file = questionary.text("Enter log file name:", default="experiment.log").ask()
            run_experiments(log_file)

        elif choice == "Analyze Results":
            csv_path = select_results_csv()
            if csv_path:
                analyze_results(csv_path)

        elif choice == "Generate Top Matrices Summary":
            generate_top_matrices_summary()

        elif choice == "Best 15 Matrices Summary":  # <-- new branch
            best15(result_dir=RESULT_DIR, output_csv="_best15_matrices_all_formats.csv")
            best15_by_mem(result_dir=RESULT_DIR, output_csv="_best15_mem_matrices_all_formats.csv")

        elif choice == "Exit":
            print("Goodbye!")
            break

        input("\nPress Enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')



if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print_welcome_message()
    input("\nPress Enter to continue...")
    main()

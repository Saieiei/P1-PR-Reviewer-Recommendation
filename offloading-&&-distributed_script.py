#!/usr/bin/env python3
# pip install "dask[distributed]" --upgrade
# pip install openpyxl

import sqlite3
import pandas as pd
import math
import fnmatch
import yaml
from datetime import datetime

# Distributed computing imports using Dask
from dask.distributed import Client
import dask

# We'll *attempt* to import these:
try:
    import cugraph
    import cudf
    _can_use_cugraph = True
except ImportError:
    _can_use_cugraph = False

try:
    import torch
    _can_use_torch = True
except ImportError:
    _can_use_torch = False

# -------------- Feedback Management (same) --------------

def initialize_feedback_table(db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            reviewer TEXT PRIMARY KEY,
            fav_rev_points INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()

def get_feedback(db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT reviewer, fav_rev_points FROM feedback")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def update_feedback_for_reviewer(reviewer_name, increment=1, db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fav_rev_points FROM feedback WHERE reviewer = ?", (reviewer_name,))
    row = cursor.fetchone()
    if row:
        new_points = row[0] + increment
        cursor.execute("UPDATE feedback SET fav_rev_points = ? WHERE reviewer = ?", (new_points, reviewer_name))
    else:
        cursor.execute("INSERT INTO feedback (reviewer, fav_rev_points) VALUES (?, ?)", (reviewer_name, increment))
    conn.commit()
    conn.close()

# -------------- Label Management (same) --------------

def flatten_patterns(possible_patterns):
    results = []
    if isinstance(possible_patterns, str):
        results.append(possible_patterns)
    elif isinstance(possible_patterns, dict):
        for val in possible_patterns.values():
            results.extend(flatten_patterns(val))
    elif isinstance(possible_patterns, list):
        for item in possible_patterns:
            results.extend(flatten_patterns(item))
    return results

def update_missing_labels(db_path="pr_data.db", yaml_path="new-prs-labeler.yml"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT pr_id, labels FROM pull_requests WHERE labels IS NULL OR labels = ''")
    pr_rows = cursor.fetchall()
    if not pr_rows:
        conn.close()
        return

    with open(yaml_path, 'r') as f:
        label_mapping = yaml.safe_load(f)

    for pr_id, _ in pr_rows:
        cursor.execute("SELECT file_path FROM pr_files WHERE pr_id = ?", (pr_id,))
        file_rows = cursor.fetchall()
        file_paths = [row[0] for row in file_rows if isinstance(row[0], str) and row[0].strip()]

        matched_labels = set()
        for label, patterns in label_mapping.items():
            all_string_patterns = flatten_patterns(patterns)
            for pattern in all_string_patterns:
                for file_path in file_paths:
                    if fnmatch.fnmatch(file_path, pattern):
                        matched_labels.add(label)
                        break
        if matched_labels:
            new_labels = ", ".join(sorted(matched_labels))
            cursor.execute("UPDATE pull_requests SET labels = ? WHERE pr_id = ?", (new_labels, pr_id))

    conn.commit()
    conn.close()

# -------------- Data Loading --------------

def load_data(db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    prs_df = pd.read_sql_query("SELECT pr_id, labels FROM pull_requests", conn)
    files_df = pd.read_sql_query("SELECT pr_id, file_path FROM pr_files", conn)
    reviews_df = pd.read_sql_query("SELECT pr_id, reviewer, review_date, state FROM reviews", conn)
    conn.close()
    return prs_df, files_df, reviews_df

def build_reviewer_pr_data(prs_df, files_df, reviews_df):
    files_grouped = files_df.groupby("pr_id")["file_path"].apply(list).reset_index()
    prs_merged = pd.merge(prs_df, files_grouped, on="pr_id", how="left")
    prs_merged["labels"] = prs_merged["labels"].fillna("")
    prs_merged["file_path"] = prs_merged["file_path"].apply(lambda x: x if isinstance(x, list) else [])

    merged = pd.merge(reviews_df, prs_merged, on="pr_id", how="left")
    reviewer_data = {}

    grouped = merged.groupby(["reviewer", "pr_id"])
    for (reviewer, pr_id), group in grouped:
        tags = set()
        for lbl in group["labels"]:
            for token in lbl.split(","):
                token = token.strip().lower()
                if token:
                    tags.add(token)

        files = set()
        for f in group["file_path"]:
            if isinstance(f, str):
                files.add(f.strip().lower())
            elif isinstance(f, list):
                for item in f:
                    files.add(item.strip().lower())

        if reviewer not in reviewer_data:
            reviewer_data[reviewer] = {}

        reviewer_data[reviewer][pr_id] = {
            "tags": tags,
            "files": files
        }

    return reviewer_data

def get_new_pr_data():
    print("Enter the file paths for the new PR (comma-separated):")
    files_input = input("> ").strip()
    print("Enter the labels for the new PR (comma-separated):")
    labels_input = input("> ").strip()

    new_files = {f.strip().lower() for f in files_input.split(",") if f.strip()}
    new_tags = {l.strip().lower() for l in labels_input.split(",") if l.strip()}
    return new_tags, new_files

# -------------- Activity (CPU-based) --------------

def compute_dynamic_activity_scores(reviews_df, tau=30):
    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.utcnow()

    def points_for_state(state):
        s = state.upper() if isinstance(state, str) else ""
        return 1 if s in ["APPROVED", "CHANGES_REQUESTED", "COMMENTED", "COMMIT"] else 0

    reviewer_scores = {}
    for idx, row in reviews_df.iterrows():
        reviewer = row["reviewer"]
        if pd.isnull(reviewer):
            continue

        base_points = points_for_state(row["state"])
        if base_points == 0:
            continue

        review_date = row["review_date"]
        if pd.isnull(review_date):
            continue

        days_diff = (now_utc - review_date).total_seconds() / 86400.0
        decay = math.exp(-days_diff / float(tau))
        partial_score = base_points * decay
        reviewer_scores[reviewer] = reviewer_scores.get(reviewer, 0.0) + partial_score

    if reviewer_scores:
        m = max(reviewer_scores.values())
        for k in reviewer_scores:
            reviewer_scores[k] /= m
    else:
        reviewer_scores = {}
    return reviewer_scores

# -------------- Similarity & Ranking --------------

def compute_absolute_similarity_reviewer(new_tags, new_files, reviewer_pr_data, w1=1, w2=1):
    total_score = 0
    for pr_id, data in reviewer_pr_data.items():
        pr_tags = data["tags"]
        pr_files = data["files"]
        matching_tags = sum(1 for nt in new_tags if any(nt in rt for rt in pr_tags))
        matching_files = sum(1 for nf in new_files if nf in pr_files)
        total_score += (w1 * matching_tags) + (w2 * matching_files)
    return total_score

# Distributed ranking function using Dask for parallel computation
def rank_reviewers(new_tags, new_files, reviewer_pr_data,
                   pagerank_scores, activity_scores,
                   w1=1, w2=1, weight_sim_act=1.0,
                   gamma=0.2, delta=0.001, db_path="pr_data.db"):
    # Compute raw absolute scores concurrently using dask.delayed
    tasks = {}
    for reviewer, pr_data in reviewer_pr_data.items():
         tasks[reviewer] = dask.delayed(compute_absolute_similarity_reviewer)(new_tags, new_files, pr_data, w1, w2)
    computed_scores = dask.compute(*list(tasks.values()))
    abs_scores = {rev: score for rev, score in zip(tasks.keys(), computed_scores)}
    
    max_abs = max(abs_scores.values()) if abs_scores else 0
    feedback = get_feedback(db_path)
    max_feedback = max(feedback.values()) if feedback else 0

    # Function to compute final score per reviewer
    def compute_final_score(reviewer, raw_abs_score):
         norm_abs_score = (raw_abs_score / max_abs) if max_abs > 0 else 0.0
         pr_score = pagerank_scores.get(reviewer, 0.0)
         act_score = activity_scores.get(reviewer, 0.0)
         fav_points = feedback.get(reviewer, 0)
         norm_fav = (fav_points / max_feedback) if max_feedback > 0 else 0.0
         sim_act_product = norm_abs_score * act_score
         final_score = weight_sim_act * sim_act_product + gamma * pr_score + delta * norm_fav
         return (reviewer, raw_abs_score, norm_abs_score, act_score, pr_score, fav_points, norm_fav, final_score)
    
    final_tasks = [dask.delayed(compute_final_score)(rev, abs_scores[rev]) for rev in abs_scores]
    computed_results = dask.compute(*final_tasks)
    results = list(computed_results)
    
    results.sort(key=lambda x: x[7], reverse=True)
    return results

# -------------- 3 Different PageRank Implementations --------------

def pagerank_nvidia_cugraph(reviews_df):
    """
    GPU PageRank with cuGraph (NVIDIA).  Requires cugraph, cudf, etc.
    """
    import cudf
    import cugraph

    # Build list of edges (co-review)
    all_reviewers = reviews_df["reviewer"].dropna().unique().tolist()
    reviewer_to_id = {r: i for i, r in enumerate(all_reviewers)}

    edges_src = []
    edges_dst = []

    pr_groups = reviews_df.groupby("pr_id")["reviewer"].apply(set)
    for _, reviewer_set in pr_groups.items():
        rlist = list(reviewer_set)
        for i in range(len(rlist)):
            for j in range(i+1, len(rlist)):
                r1 = reviewer_to_id[rlist[i]]
                r2 = reviewer_to_id[rlist[j]]
                edges_src.append(r1)
                edges_dst.append(r2)
                edges_src.append(r2)
                edges_dst.append(r1)

    if not edges_src:
        # no edges => uniform
        n = len(all_reviewers)
        if n == 0:
            return {}
        val = 1.0 / n
        return {r: val for r in all_reviewers}

    gdf_edges = cudf.DataFrame({'src': edges_src, 'dst': edges_dst})

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(gdf_edges, source='src', destination='dst', renumber=False)

    pr_df = cugraph.pagerank(G)
    # pr_df columns typically: ['vertex', 'pagerank']
    rank_map = {}
    for row in pr_df.itertuples():
        # row.vertex is ID, row.pagerank is rank
        rank_map[row.vertex] = row.pagerank

    # convert back
    result = {}
    for r, idx in reviewer_to_id.items():
        result[r] = float(rank_map.get(idx, 0.0))
    return result

def pagerank_amd_torch(reviews_df, alpha=0.85, max_iter=100, tol=1e-6):
    """
    GPU PageRank with PyTorch for ROCm (AMD).
    """
    import torch

    all_reviewers = reviews_df["reviewer"].dropna().unique().tolist()
    n = len(all_reviewers)
    if n == 0:
        return {}
    reviewer_to_id = {r: i for i, r in enumerate(all_reviewers)}

    edges = []
    pr_groups = reviews_df.groupby("pr_id")["reviewer"].apply(set)
    for _, reviewer_set in pr_groups.items():
        rlist = list(reviewer_set)
        for i in range(len(rlist)):
            for j in range(i+1, len(rlist)):
                edges.append((reviewer_to_id[rlist[i]], reviewer_to_id[rlist[j]]))
                edges.append((reviewer_to_id[rlist[j]], reviewer_to_id[rlist[i]]))

    if not edges:
        val = 1.0 / n
        return {r: val for r in all_reviewers}

    device = torch.device("cuda")  # for ROCm build, "cuda" is an alias for AMD GPU
    row_idx = []
    col_idx = []
    for (src, dst) in edges:
        row_idx.append(src)
        col_idx.append(dst)
    row_idx_t = torch.tensor(row_idx, dtype=torch.long, device=device)
    col_idx_t = torch.tensor(col_idx, dtype=torch.long, device=device)
    data_vals = torch.ones(len(row_idx), dtype=torch.float32, device=device)

    A = torch.sparse_coo_tensor(
        indices=torch.stack([row_idx_t, col_idx_t], dim=0),
        values=data_vals,
        size=(n, n),
        device=device
    ).coalesce()

    outdeg = torch.sparse.sum(A, dim=1).to_dense()
    outdeg = torch.where(outdeg > 0, outdeg, torch.ones_like(outdeg))

    rank = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)

    one_minus_alpha = 1.0 - alpha
    for _ in range(max_iter):
        old_rank = rank.clone()
        rank_div = rank / outdeg
        # spmv => A * rank_div
        spmv = torch.sparse.mm(A, rank_div.unsqueeze(1)).squeeze(1)
        rank = alpha * spmv + one_minus_alpha * (1.0 / n)
        diff = torch.norm(rank - old_rank, p=1)
        if diff.item() < tol:
            break

    rank_cpu = rank.cpu().numpy()
    result = {}
    for r, idx in reviewer_to_id.items():
        result[r] = float(rank_cpu[idx])
    return result

def pagerank_cpu_networkx(reviews_df):
    """
    CPU fallback using networkx for PageRank.
    """
    import networkx as nx
    G = nx.Graph()
    # build edges
    pr_groups = reviews_df.groupby("pr_id")["reviewer"].apply(set)
    for _, reviewer_set in pr_groups.items():
        rev_list = list(reviewer_set)
        for i in range(len(rev_list)):
            for j in range(i+1, len(rev_list)):
                G.add_edge(rev_list[i], rev_list[j])
    if len(G.nodes) == 0:
        return {}
    pr = nx.pagerank(G, alpha=0.85)
    return pr

# -------------- Main --------------

def main():
    # Initialize Dask distributed client for parallel processing.
    client = Client()
    print("Dask client created:", client)

    db_path = "pr_data.db"

    # 1) Initialize DB, etc.
    initialize_feedback_table(db_path)
    update_missing_labels(db_path=db_path, yaml_path="new-prs-labeler.yml")

    # 2) Load data
    prs_df, files_df, reviews_df = load_data(db_path)
    print("Data loaded from pr_data.db.")

    # 3) Build reviewer data
    reviewer_pr_data = build_reviewer_pr_data(prs_df, files_df, reviews_df)
    if not reviewer_pr_data:
        print("No reviewer PR data found.")
        return

    # 4) Decide which PageRank to do
    pagerank_scores = {}
    # We'll detect if we have an NVIDIA environment first
    if _can_use_cugraph:
        print("Using cuGraph (NVIDIA CUDA) for PageRank.")
        pagerank_scores = pagerank_nvidia_cugraph(reviews_df)
    elif _can_use_torch:
        # We'll try to see if torch.cuda.is_available() (for AMD, it still says True if ROCm is present)
        if hasattr(torch.version, 'hip') or torch.cuda.is_available():
            print("Using PyTorch (ROCm or CUDA) for PageRank.")
            pagerank_scores = pagerank_amd_torch(reviews_df)  # or rename if needed
        else:
            print("PyTorch is installed but no GPU available. Falling back to CPU.")
            pagerank_scores = pagerank_cpu_networkx(reviews_df)
    else:
        print("No GPU library found. Falling back to CPU (networkx).")
        pagerank_scores = pagerank_cpu_networkx(reviews_df)

    print("PageRank done. Number of ranked reviewers:", len(pagerank_scores))

    # 5) Compute dynamic activity
    activity_scores = compute_dynamic_activity_scores(reviews_df)
    print("Activity scores done.")

    # 6) Prompt new PR info
    print("\n=== Enter new PR details ===")
    new_tags, new_files = get_new_pr_data()
    if not (new_tags or new_files):
        print("No new PR info. Exiting.")
        return

    # 7) Rank reviewers using the distributed ranking function
    ranked_reviewers = rank_reviewers(
        new_tags, new_files,
        reviewer_pr_data,
        pagerank_scores,
        activity_scores,
        w1=1, w2=2,
        weight_sim_act=1.0,
        gamma=0.2,
        delta=0.001,
        db_path=db_path
    )

    # Print top 15
    print("\n=== Top Reviewer Recommendations ===")
    for i, (rev, raw_abs, norm_abs, act_score, pr_score, fav_points, norm_fav, final_score) in enumerate(ranked_reviewers[:15], start=1):
        print(f"{i}. Reviewer: {rev:20s} | abs_match={raw_abs}, norm_abs={norm_abs:.4f}, act={act_score:.4f}, pr={pr_score:.4f}, fav_points={fav_points}, norm_fav={norm_fav:.4f}, final={final_score:.4f}")

    # 8) Feedback loop
    feedback_input = input("\nAre you satisfied with the recommendations? (y/n): ").strip().lower()
    if feedback_input not in ['y', 'yes']:
        reviewer_choice = input("Enter the name of the reviewer you prefer: ").strip()
        if reviewer_choice:
            update_feedback_for_reviewer(reviewer_choice, increment=1, db_path=db_path)
            print(f"Feedback recorded for {reviewer_choice}. Re-ranking...")

            ranked_reviewers = rank_reviewers(
                new_tags, new_files,
                reviewer_pr_data,
                pagerank_scores,
                activity_scores,
                w1=1, w2=2,
                weight_sim_act=1.0,
                gamma=0.2,
                delta=0.001,
                db_path=db_path
            )

            print("\n=== Updated Top Reviewer Recommendations ===")
            for i, (rev, raw_abs, norm_abs, act_score, pr_score, fav_points, norm_fav, final_score) in enumerate(ranked_reviewers[:15], start=1):
                print(f"{i}. Reviewer: {rev:20s} | abs_match={raw_abs}, norm_abs={norm_abs:.4f}, act={act_score:.4f}, pr={pr_score:.4f}, fav_points={fav_points}, norm_fav={norm_fav:.4f}, final={final_score:.4f}")

    # 9) Export to Excel
    import openpyxl
    df = pd.DataFrame(
        ranked_reviewers,
        columns=["reviewer", "raw_abs_match", "normalized_abs_match",
                 "activity_score", "pagerank_score", "fav_rev_points",
                 "norm_fav_rev_points", "final_score"]
    )
    out_xlsx = "reviewer_scores_absolute_normalized.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"\nAll reviewer scores saved to {out_xlsx}.")

    # Shutdown Dask client
    client.close()

if __name__ == "__main__":
    main()


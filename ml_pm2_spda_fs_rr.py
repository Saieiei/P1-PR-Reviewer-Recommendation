#  MISSING LABELS _ PARTIAL MATCH 2 _ SAME POINTS _ DYNAMIC ACTIVITY _ FINAL SCORE _ REVIEWER RECOMMENDATION
import sqlite3
import pandas as pd
import networkx as nx
import math
import fnmatch          # used for glob-style file matching
import yaml             # used to load the YAML file
from datetime import datetime, timedelta

def update_missing_labels(db_path="pr_data.db", yaml_path="new-prs-labeler.yml"):
    """
    For each pull request in the database with missing labels (NULL or empty),
    this function looks at the associated file paths and uses a YAML file mapping 
    (file patterns to label names) to determine which labels should be assigned.
    It then updates the 'pull_requests' table in the SQLite database accordingly.
    """
    # Connect to the SQLite database.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Select PRs with missing labels.
    cursor.execute("SELECT pr_id, labels FROM pull_requests WHERE labels IS NULL OR labels = ''")
    pr_rows = cursor.fetchall()
    
    # If no PRs have missing labels, nothing to do.
    if not pr_rows:
        conn.close()
        return
    
    # Load the YAML mapping file.
    with open(yaml_path, 'r') as f:
        label_mapping = yaml.safe_load(f)
    
    # Process each PR with missing labels.
    for pr_id, _ in pr_rows:
        # Retrieve all file paths associated with the current PR.
        cursor.execute("SELECT file_path FROM pr_files WHERE pr_id = ?", (pr_id,))
        file_rows = cursor.fetchall()
        # Create a list of file paths (ignore any null entries).
        file_paths = [row[0] for row in file_rows if row[0]]
        
        matched_labels = set()
        # Iterate over each label and its associated file patterns.
        for label, patterns in label_mapping.items():
            for pattern in patterns:
                # Check if any file path matches the pattern.
                for file_path in file_paths:
                    if fnmatch.fnmatch(file_path, pattern):
                        matched_labels.add(label)
                        break  # No need to check more patterns for this label.
        
        # If one or more labels matched, update the database.
        if matched_labels:
            # Create a comma-separated string of labels (sorted for consistency).
            new_labels = ", ".join(sorted(matched_labels))
            cursor.execute("UPDATE pull_requests SET labels = ? WHERE pr_id = ?", (new_labels, pr_id))
    
    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

def load_data(db_path="pr_data.db"):
    """
    Loads pull_requests, pr_files, and reviews tables from the SQLite database.
    Returns three DataFrames: prs_df, files_df, reviews_df.
    """
    conn = sqlite3.connect(db_path)
    prs_df = pd.read_sql_query("SELECT pr_id, labels FROM pull_requests", conn)
    files_df = pd.read_sql_query("SELECT pr_id, file_path FROM pr_files", conn)
    reviews_df = pd.read_sql_query("SELECT pr_id, reviewer, review_date, state FROM reviews", conn)
    conn.close()
    return prs_df, files_df, reviews_df

def build_reviewer_pr_data(prs_df, files_df, reviews_df):
    """
    Build a dictionary with the structure:
      reviewer -> { pr_id: { "tags": set(), "files": set() } }
    For each PR the reviewer participated in, we collect unique tags (from labels)
    and unique file paths.
    """
    # Group file paths into a list per PR.
    files_grouped = files_df.groupby("pr_id")["file_path"].apply(list).reset_index()
    # Merge pull request data with the grouped file paths.
    prs_merged = pd.merge(prs_df, files_grouped, on="pr_id", how="left")
    # Replace missing labels with an empty string (now that missing ones have been updated).
    prs_merged["labels"] = prs_merged["labels"].fillna("")
    # Ensure file_path is a list; if missing, use an empty list.
    prs_merged["file_path"] = prs_merged["file_path"].apply(lambda x: x if isinstance(x, list) else [])
    
    # Merge with reviews so that we know which reviewer worked on which PR.
    merged = pd.merge(reviews_df, prs_merged, on="pr_id", how="left")
    
    reviewer_data = {}
    grouped = merged.groupby(["reviewer", "pr_id"])
    for (reviewer, pr_id), group in grouped:
        # For tags: split the labels by comma and accumulate unique tokens.
        tags = set()
        for lbl in group["labels"]:
            for token in lbl.split(","):
                token = token.strip().lower()
                if token:
                    tags.add(token)
        
        # For files: accumulate unique file paths (lowercased and stripped).
        files = set()
        for f in group["file_path"]:
            if isinstance(f, str):
                files.add(f.strip().lower())
            elif isinstance(f, list):
                for item in f:
                    files.add(item.strip().lower())
        
        if reviewer not in reviewer_data:
            reviewer_data[reviewer] = {}
        reviewer_data[reviewer][pr_id] = {"tags": tags, "files": files}
    return reviewer_data

# (Other functions remain unchanged...)
def get_new_pr_data():
    """
    Prompt the user for new PR details (file paths and labels) and return them
    as sets of lowercased strings.
    """
    print("Enter the file paths for the new PR (comma-separated):")
    files_input = input("> ").strip()
    print("Enter the labels for the new PR (comma-separated):")
    labels_input = input("> ").strip()
    
    new_files = set([f.strip().lower() for f in files_input.split(",") if f.strip()])
    new_tags = set([l.strip().lower() for l in labels_input.split(",") if l.strip()])
    return new_tags, new_files

def build_reviewer_graph(reviews_df):
    """
    Build an undirected graph where nodes are reviewers and an edge exists between
    two reviewers if they co-reviewed the same PR. The edge weight is the number
    of co-reviews shared.
    """
    G = nx.Graph()
    reviewers = reviews_df["reviewer"].unique()
    G.add_nodes_from(reviewers)
    
    pr_groups = reviews_df.groupby("pr_id")["reviewer"].apply(set)
    for _, reviewer_set in pr_groups.items():
        reviewer_list = list(reviewer_set)
        for i in range(len(reviewer_list)):
            for j in range(i+1, len(reviewer_list)):
                r1 = reviewer_list[i]
                r2 = reviewer_list[j]
                if G.has_edge(r1, r2):
                    G[r1][r2]["weight"] += 1
                else:
                    G.add_edge(r1, r2, weight=1)
    return G

def compute_dynamic_activity_scores(reviews_df, tau=30):
    """
    Compute a dynamic activity score for each reviewer using exponential decay,
    with every review action weighted equally.
    """
    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.utcnow()
    
    def points_for_state(state):
        s = state.upper() if isinstance(state, str) else ""
        if s in ["APPROVED", "CHANGES_REQUESTED", "COMMENTED", "COMMIT"]:
            return 1
        return 0
    
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
        max_score = max(reviewer_scores.values())
        activity_scores = {r: (score / max_score) for r, score in reviewer_scores.items()}
    else:
        activity_scores = {}
    
    return activity_scores

def compute_absolute_similarity_reviewer(new_tags, new_files, reviewer_pr_data, w1=1, w2=1):
    """
    For a single reviewer, compute the absolute matching score for their PR data.
    """
    total_score = 0
    for pr_id, data in reviewer_pr_data.items():
        pr_tags = data["tags"]
        pr_files = data["files"]
        matching_tags = 0
        for nt in new_tags:
            if any(nt in rt for rt in pr_tags):
                matching_tags += 1
        matching_files = 0
        for nf in new_files:
            if nf in pr_files:
                matching_files += 1
        pr_score = (w1 * matching_tags) + (w2 * matching_files)
        total_score += pr_score
    return total_score

def rank_reviewers(new_tags, new_files, reviewer_pr_data, pagerank_scores, activity_scores,
                   w1=1, w2=1, weight_sim_act=1.0, gamma=0.2):
    """
    Rank reviewers based on matching scores, activity, and PageRank.
    """
    abs_scores = {}
    for reviewer, pr_data in reviewer_pr_data.items():
        score = compute_absolute_similarity_reviewer(new_tags, new_files, pr_data, w1, w2)
        abs_scores[reviewer] = score

    max_abs = max(abs_scores.values()) if abs_scores else 0

    results = []
    for reviewer, raw_abs_score in abs_scores.items():
        norm_abs_score = (raw_abs_score / max_abs) if max_abs > 0 else 0.0
        pr_score = pagerank_scores.get(reviewer, 0.0)
        act_score = activity_scores.get(reviewer, 0.0)
        sim_act_product = norm_abs_score * act_score
        final_score = weight_sim_act * sim_act_product + gamma * pr_score
        results.append((reviewer, raw_abs_score, norm_abs_score, act_score, pr_score, final_score))
    
    results.sort(key=lambda x: x[5], reverse=True)
    return results

def main():
    # First, update missing labels in the database based on file paths and the YAML mapping.
    update_missing_labels(db_path="pr_data.db", yaml_path="new-prs-labeler.yml")
    
    # Now load the data (the pull_requests table now has updated labels for PRs that were missing them).
    prs_df, files_df, reviews_df = load_data("pr_data.db")
    print("Data loaded from pr_data.db.")
    
    # Build reviewer PR data.
    reviewer_pr_data = build_reviewer_pr_data(prs_df, files_df, reviews_df)
    if not reviewer_pr_data:
        print("No reviewer PR data found. Please check your database content.")
        return
    
    # Build the reviewer graph and compute PageRank.
    G = build_reviewer_graph(reviews_df)
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    print("PageRank scores computed.")
    
    # Compute dynamic activity scores.
    activity_scores = compute_dynamic_activity_scores(reviews_df, tau=30)
    print("Dynamic activity scores computed.")
    
    # Get new PR data (tags + files) from user input.
    print("\n=== Enter new PR details ===")
    new_tags, new_files = get_new_pr_data()
    if not (new_tags or new_files):
        print("No new PR information provided. Exiting.")
        return
    
    # Rank reviewers based on the combined criteria.
    ranked_reviewers = rank_reviewers(
        new_tags, 
        new_files, 
        reviewer_pr_data, 
        pagerank_scores, 
        activity_scores,
        w1=1,   # weight for matching tags
        w2=2,   # weight for matching files
        weight_sim_act=1.0, 
        gamma=0.2
    )
    
    # Print top 10 reviewer recommendations.
    print("\n=== Top Reviewer Recommendations ===")
    for i, (rev, raw_abs, norm_abs, act_score, pr_score, final_score) in enumerate(ranked_reviewers[:10], start=1):
        print(f"{i}. Reviewer: {rev:20s} | abs_match={raw_abs}, norm_abs={norm_abs:.4f}, act={act_score:.4f}, pr={pr_score:.4f}, final={final_score:.4f}")
    
    # Save all reviewer scores to an Excel file.
    df = pd.DataFrame(
        ranked_reviewers,
        columns=["reviewer", "raw_abs_match", "normalized_abs_match", "activity_score", "pagerank_score", "final_score"]
    )
    excel_name = "reviewer_scores_absolute_normalized.xlsx"
    df.to_excel(excel_name, index=False)
    print(f"\nAll reviewer scores have been saved to '{excel_name}'.")

if __name__ == "__main__":
    main()

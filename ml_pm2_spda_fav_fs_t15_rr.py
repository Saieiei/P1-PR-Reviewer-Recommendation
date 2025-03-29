#  MISSING LABELS _ PARTIAL MATCH 2 _ SAME POINTS _ DYNAMIC ACTIVITY _ FAV REVIEWER _ FINAL SCORE _ TOTAL 15 _ REVIEWER RECOMMENDATION
import sqlite3
import pandas as pd
import networkx as nx
import math
import fnmatch          # for glob-style file matching
import yaml             # for loading YAML mapping file
from datetime import datetime, timedelta

# --- New Functions for Feedback Management ---

def initialize_feedback_table(db_path="pr_data.db"):
    """
    Create the feedback table if it doesn't exist.
    The table contains columns: reviewer (TEXT PRIMARY KEY) and fav_rev_points (INTEGER).
    """
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
    """
    Retrieve feedback data as a dictionary mapping reviewer to fav_rev_points.
    If a reviewer is not in the table, assume 0 points.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT reviewer, fav_rev_points FROM feedback")
    rows = cursor.fetchall()
    conn.close()
    feedback = {row[0]: row[1] for row in rows}
    return feedback

def update_feedback_for_reviewer(reviewer_name, increment=1, db_path="pr_data.db"):
    """
    Increase the fav_rev_points for the specified reviewer by 'increment'.
    If the reviewer does not exist in the feedback table, insert a new row.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Check if reviewer exists
    cursor.execute("SELECT fav_rev_points FROM feedback WHERE reviewer = ?", (reviewer_name,))
    row = cursor.fetchone()
    if row:
        new_points = row[0] + increment
        cursor.execute("UPDATE feedback SET fav_rev_points = ? WHERE reviewer = ?", (new_points, reviewer_name))
    else:
        cursor.execute("INSERT INTO feedback (reviewer, fav_rev_points) VALUES (?, ?)", (reviewer_name, increment))
    conn.commit()
    conn.close()


# --- Existing Functions (unchanged) ---

def flatten_patterns(possible_patterns):
    """
    Recursively flattens nested patterns from YAML so that only strings are returned.
    e.g. a string stays a string,
         a list becomes multiple strings (or more nested lists/dicts to flatten),
         a dict is traversed until we find strings.
    """
    results = []
    if isinstance(possible_patterns, str):
        # If it's already a string, just add it
        results.append(possible_patterns)
    elif isinstance(possible_patterns, dict):
        # If it's a dict, flatten each value
        for val in possible_patterns.values():
            results.extend(flatten_patterns(val))
    elif isinstance(possible_patterns, list):
        # If it's a list, flatten each element
        for item in possible_patterns:
            results.extend(flatten_patterns(item))
    # Otherwise, ignore (not a str/list/dict)
    return results

def update_missing_labels(db_path="pr_data.db", yaml_path="new-prs-labeler.yml"):
    """
    For each PR with missing labels (NULL or empty), determine the labels to assign
    based on file paths using a YAML mapping file, and update the database.
    This version flattens nested YAML entries so only string patterns get passed to fnmatch.
    """
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
        # Only include non-empty string file paths
        file_paths = [row[0] for row in file_rows if isinstance(row[0], str) and row[0].strip()]
        
        matched_labels = set()
        for label, patterns in label_mapping.items():
            # Flatten any nested dicts/lists to obtain a list of strings
            all_string_patterns = flatten_patterns(patterns)
            
            # Perform fnmatch on each string pattern
            for pattern in all_string_patterns:
                for file_path in file_paths:
                    if fnmatch.fnmatch(file_path, pattern):
                        matched_labels.add(label)
                        break
        
        # Update if we found any matching labels
        if matched_labels:
            new_labels = ", ".join(sorted(matched_labels))
            cursor.execute("UPDATE pull_requests SET labels = ? WHERE pr_id = ?", (new_labels, pr_id))

    conn.commit()
    conn.close()

def load_data(db_path="pr_data.db"):
    """
    Load pull_requests, pr_files, and reviews tables from the database.
    """
    conn = sqlite3.connect(db_path)
    prs_df = pd.read_sql_query("SELECT pr_id, labels FROM pull_requests", conn)
    files_df = pd.read_sql_query("SELECT pr_id, file_path FROM pr_files", conn)
    reviews_df = pd.read_sql_query("SELECT pr_id, reviewer, review_date, state FROM reviews", conn)
    conn.close()
    return prs_df, files_df, reviews_df

def build_reviewer_pr_data(prs_df, files_df, reviews_df):
    """
    Build a dictionary: reviewer -> { pr_id: { "tags": set(), "files": set() } }.
    """
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
        reviewer_data[reviewer][pr_id] = {"tags": tags, "files": files}
    return reviewer_data

def get_new_pr_data():
    """
    Prompt the user for new PR file paths and labels, returning sets of lowercased strings.
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
    Build an undirected graph connecting reviewers who co-reviewed the same PR.
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
    Compute a dynamic activity score for each reviewer using exponential decay.
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
    Compute an absolute matching score for a single reviewer based on their PR data.
    """
    total_score = 0
    for pr_id, data in reviewer_pr_data.items():
        pr_tags = data["tags"]
        pr_files = data["files"]
        matching_tags = sum(1 for nt in new_tags if any(nt in rt for rt in pr_tags))
        matching_files = sum(1 for nf in new_files if nf in pr_files)
        pr_score = (w1 * matching_tags) + (w2 * matching_files)
        total_score += pr_score
    return total_score

def rank_reviewers(new_tags, new_files, reviewer_pr_data, pagerank_scores, activity_scores,
                   w1=1, w2=1, weight_sim_act=1.0, gamma=0.2, delta=0.001, db_path="pr_data.db"):
    """
    Rank reviewers based on:
      - Absolute matching score (from tags and files)
      - Dynamic activity score
      - PageRank score
      - Feedback (favorite reviewer points)
      
    The final score is:
      final_score = weight_sim_act * (normalized abs match * activity) + gamma * pr_score + delta * norm_fav_rev_points
      
    Returns a sorted list of tuples with reviewer details.
    """
    abs_scores = {}
    for reviewer, pr_data in reviewer_pr_data.items():
        score = compute_absolute_similarity_reviewer(new_tags, new_files, pr_data, w1, w2)
        abs_scores[reviewer] = score
    max_abs = max(abs_scores.values()) if abs_scores else 0

    # Get feedback data from the feedback table.
    feedback = get_feedback(db_path)
    max_feedback = max(feedback.values()) if feedback else 0

    results = []
    for reviewer, raw_abs_score in abs_scores.items():
        norm_abs_score = (raw_abs_score / max_abs) if max_abs > 0 else 0.0
        pr_score = pagerank_scores.get(reviewer, 0.0)
        act_score = activity_scores.get(reviewer, 0.0)
        sim_act_product = norm_abs_score * act_score
        # Get fav_rev_points for reviewer; default to 0 if not present.
        fav_points = feedback.get(reviewer, 0)
        norm_fav = (fav_points / max_feedback) if max_feedback > 0 else 0.0
        final_score = weight_sim_act * sim_act_product + gamma * pr_score + delta * norm_fav
        results.append((reviewer, raw_abs_score, norm_abs_score, act_score, pr_score, fav_points, norm_fav, final_score))
    results.sort(key=lambda x: x[7], reverse=True)
    return results

# --- Main function with modifications for printing top 15 and feedback loop ---

def main():
    db_path = "pr_data.db"
    # Initialize feedback table if not exists.
    initialize_feedback_table(db_path)
    
    # Update missing labels.
    update_missing_labels(db_path=db_path, yaml_path="new-prs-labeler.yml")
    
    # Load data from the database.
    prs_df, files_df, reviews_df = load_data(db_path)
    print("Data loaded from pr_data.db.")
    
    # Build reviewer PR data.
    reviewer_pr_data = build_reviewer_pr_data(prs_df, files_df, reviews_df)
    if not reviewer_pr_data:
        print("No reviewer PR data found. Please check your database content.")
        return
    
    # Build reviewer graph and compute PageRank.
    G = build_reviewer_graph(reviews_df)
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    print("PageRank scores computed.")
    
    # Compute dynamic activity scores.
    activity_scores = compute_dynamic_activity_scores(reviews_df, tau=30)
    print("Dynamic activity scores computed.")
    
    # Get new PR details from user.
    print("\n=== Enter new PR details ===")
    new_tags, new_files = get_new_pr_data()
    if not (new_tags or new_files):
        print("No new PR information provided. Exiting.")
        return
    
    # Rank reviewers (incorporating feedback).
    ranked_reviewers = rank_reviewers(
        new_tags, 
        new_files, 
        reviewer_pr_data, 
        pagerank_scores, 
        activity_scores,
        w1=1,   # weight for matching tags
        w2=2,   # weight for matching files
        weight_sim_act=1.0, 
        gamma=0.2,
        delta=0.001,
        db_path=db_path
    )
    
    # Print top 15 reviewer recommendations.
    print("\n=== Top Reviewer Recommendations ===")
    for i, (rev, raw_abs, norm_abs, act_score, pr_score, fav_points, norm_fav, final_score) in enumerate(ranked_reviewers[:15], start=1):
        print(f"{i}. Reviewer: {rev:20s} | abs_match={raw_abs}, norm_abs={norm_abs:.4f}, act={act_score:.4f}, pr={pr_score:.4f}, fav_points={fav_points}, norm_fav={norm_fav:.4f}, final={final_score:.4f}")
    
    # --- Feedback Loop ---
    feedback_input = input("\nAre you satisfied with the recommendations? (y/n): ").strip().lower()
    if feedback_input not in ['y', 'yes']:
        reviewer_choice = input("Enter the name of the reviewer you prefer: ").strip()
        if reviewer_choice:
            update_feedback_for_reviewer(reviewer_choice, increment=1, db_path=db_path)
            print(f"Feedback recorded: {reviewer_choice} has been given additional points.")
            # Re-rank after feedback update.
            ranked_reviewers = rank_reviewers(
                new_tags, 
                new_files, 
                reviewer_pr_data, 
                pagerank_scores, 
                activity_scores,
                w1=1,
                w2=2,
                weight_sim_act=1.0,
                gamma=0.2,
                delta=0.001,
                db_path=db_path
            )
            print("\n=== Updated Top Reviewer Recommendations ===")
            for i, (rev, raw_abs, norm_abs, act_score, pr_score, fav_points, norm_fav, final_score) in enumerate(ranked_reviewers[:15], start=1):
                print(f"{i}. Reviewer: {rev:20s} | abs_match={raw_abs}, norm_abs={norm_abs:.4f}, act={act_score:.4f}, pr={pr_score:.4f}, fav_points={fav_points}, norm_fav={norm_fav:.4f}, final={final_score:.4f}")
        else:
            print("No reviewer name provided. Exiting feedback loop.")
    
    # Save all reviewer scores to an Excel file.
    df = pd.DataFrame(
        ranked_reviewers,
        columns=["reviewer", "raw_abs_match", "normalized_abs_match", "activity_score", "pagerank_score", "fav_rev_points", "norm_fav_rev_points", "final_score"]
    )
    excel_name = "reviewer_scores_absolute_normalized.xlsx"
    df.to_excel(excel_name, index=False)
    print(f"\nAll reviewer scores have been saved to '{excel_name}'.")

if __name__ == "__main__":
    main()

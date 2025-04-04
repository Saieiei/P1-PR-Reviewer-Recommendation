#!/usr/bin/env python3
import argparse
import os
import sqlite3
import pandas as pd
import networkx as nx
import math
import fnmatch
import re
import yaml
import requests
from datetime import datetime

# -------------------------------------------------
# Helper functions: GitHub API calls for PR details
# -------------------------------------------------

def get_pr_labels(pr_number):
    """
    Fetch the PR details from GitHub and return a set of label names (in lowercase).
    """
    token = os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("GITHUB_OWNER")
    repo = os.environ.get("GITHUB_REPO")
    if not (token and owner and repo):
        print("Missing one of GITHUB_TOKEN/GITHUB_OWNER/GITHUB_REPO environment variables.")
        return set()
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("Failed to fetch PR details:", r.text)
        return set()
    pr_data = r.json()
    labels = pr_data.get("labels", [])
    return {label.get("name", "").strip().lower() for label in labels if label.get("name")}

def get_pr_changed_files(pr_number):
    """
    Fetch the list of changed files in the PR from GitHub and return a set of filenames (in lowercase).
    """
    token = os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("GITHUB_OWNER")
    repo = os.environ.get("GITHUB_REPO")
    if not (token and owner and repo):
        print("Missing one of GITHUB_TOKEN/GITHUB_OWNER/GITHUB_REPO environment variables.")
        return set()
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("Failed to fetch PR files:", r.text)
        return set()
    files_data = r.json()
    return {f.get("filename", "").strip().lower() for f in files_data if f.get("filename")}

# -------------------------------------------------
# Auto-assign labels from YAML mapping (case-insensitive using regex)
# -------------------------------------------------

def auto_assign_labels_from_yaml(changed_files, yaml_path="new-prs-labeler.yml"):
    """
    For each file in changed_files, check if it matches any pattern in the YAML mapping.
    This version replaces '**' with '*' in the pattern, converts both the pattern and the file path to lowercase,
    and uses a regex compiled with re.IGNORECASE for case-insensitive matching.
    Returns a set of assigned labels.
    """
    import re  # ensure re is imported
    assigned_labels = set()
    try:
        with open(yaml_path, 'r') as f:
            label_mapping = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"WARNING: {yaml_path} not found. Skipping auto label assignment.")
        return assigned_labels

    for label, patterns in label_mapping.items():
        # Flatten the list of patterns (handle both strings and dicts)
        flat_patterns = []
        for p in patterns:
            if isinstance(p, str):
                flat_patterns.append(p)
            elif isinstance(p, dict):
                for key, value in p.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                flat_patterns.append(item)
        # Process each flattened pattern
        for pattern in flat_patterns:
            # Replace "**" with "*" to avoid fnmatch quirks
            mod_pattern = pattern.replace("**", "*")
            # Lowercase the modified pattern and compile a regex with IGNORECASE
            regex = re.compile(fnmatch.translate(mod_pattern.lower()), re.IGNORECASE)
            # Check each changed file (which is already lowercased when fetched)
            for file_path in changed_files:
                if regex.fullmatch(file_path):
                    assigned_labels.add(label.strip().lower())
                    # Do NOT break, so that multiple labels can be accumulated
    return assigned_labels

# -------------------------------------------------
# Existing Database & Similarity Functions
# -------------------------------------------------

def update_missing_labels(db_path="pr_data.db", yaml_path="new-prs-labeler.yml"):
    """
    For each PR in the DB with missing labels, attempt to fill them by matching
    file paths with patterns from new-prs-labeler.yml.
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
        file_paths = [row[0] for row in file_rows if row[0]]
        matched_labels = set()
        for label, patterns in label_mapping.items():
            flat_patterns = []
            for p in patterns:
                if isinstance(p, str):
                    flat_patterns.append(p)
                elif isinstance(p, dict):
                    for key, value in p.items():
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, str):
                                    flat_patterns.append(item)
            for pattern in flat_patterns:
                pattern_lower = pattern.lower()
                regex = re.compile(fnmatch.translate(pattern_lower), re.IGNORECASE)
                for file_path in file_paths:
                    file_path_lower = file_path.lower()
                    if regex.fullmatch(file_path_lower):
                        matched_labels.add(label)
                        break
        if matched_labels:
            new_labels = ", ".join(sorted(matched_labels))
            cursor.execute("UPDATE pull_requests SET labels = ? WHERE pr_id = ?", (new_labels, pr_id))
    conn.commit()
    conn.close()

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
        reviewer_data[reviewer][pr_id] = {"tags": tags, "files": files}
    return reviewer_data

def build_reviewer_graph(reviews_df):
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
    total_score = 0
    for pr_id, data in reviewer_pr_data.items():
        pr_tags = data["tags"]
        pr_files = data["files"]
        matching_tags = sum(1 for nt in new_tags if any(nt in rt for rt in pr_tags))
        matching_files = sum(1 for nf in new_files if nf in pr_files)
        pr_score = (w1 * matching_tags) + (w2 * matching_files)
        total_score += pr_score
    return total_score

def get_feedback(db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT reviewer, fav_rev_points FROM feedback")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def rank_reviewers(new_tags, new_files, reviewer_pr_data, pagerank_scores, activity_scores,
                   w1=1, w2=1, weight_sim_act=1.0, gamma=0.2, delta=0.001, db_path="pr_data.db"):
    abs_scores = {}
    for reviewer, pr_data in reviewer_pr_data.items():
        score = compute_absolute_similarity_reviewer(new_tags, new_files, pr_data, w1, w2)
        abs_scores[reviewer] = score
    max_abs = max(abs_scores.values()) if abs_scores else 0
    feedback = get_feedback(db_path)
    max_feedback = max(feedback.values()) if feedback else 0
    results = []
    for reviewer, raw_abs_score in abs_scores.items():
        norm_abs_score = (raw_abs_score / max_abs) if max_abs > 0 else 0.0
        pr_score = pagerank_scores.get(reviewer, 0.0)
        act_score = activity_scores.get(reviewer, 0.0)
        sim_act_product = norm_abs_score * act_score
        fav_points = feedback.get(reviewer, 0)
        norm_fav = (fav_points / max_feedback) if max_feedback > 0 else 0.0
        final_score = weight_sim_act * sim_act_product + gamma * pr_score + delta * norm_fav
        results.append((reviewer, raw_abs_score, norm_abs_score, act_score, pr_score, fav_points, norm_fav, final_score))
    results.sort(key=lambda x: x[7], reverse=True)
    return results

# -------------------------------------------------
# Function to post a comment on the PR via GitHub API
# -------------------------------------------------

def post_comment(pr_number, comment_body):
    token = os.environ.get("GITHUB_TOKEN")
    owner = os.environ.get("GITHUB_OWNER")
    repo = os.environ.get("GITHUB_REPO")
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {"body": comment_body}
    print("Posting comment to URL:", url)  # Debug output
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        print("Comment posted successfully.")
    else:
        print("Failed to post comment:", response.text)

# -------------------------------------------------
# Main: Automated GitHub Actions run (non-interactive)
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute and post reviewer recommendations automatically.")
    parser.add_argument("--pr_number", type=int, required=True, help="PR number to post recommendations on")
    parser.add_argument("--db_path", type=str, default="pr_data.db", help="Path to the database file")
    args = parser.parse_args()

    # Fetch PR labels and changed files using GitHub API.
    fetched_labels = get_pr_labels(args.pr_number)
    changed_files = get_pr_changed_files(args.pr_number)
    
    print("Fetched PR labels from GitHub:", fetched_labels)
    print("Fetched PR changed files from GitHub:", changed_files)
    
    # If no labels were found on the PR, auto-assign them using YAML mapping.
    auto_labels = auto_assign_labels_from_yaml(changed_files, yaml_path="new-prs-labeler.yml")
    if not fetched_labels:
        new_tags = auto_labels
        print("No labels found on PR; using auto-assigned labels:", new_tags)
    else:
        new_tags = fetched_labels.union(auto_labels)
        print("Combined PR labels (fetched + auto-assigned):", new_tags)
    
    # For the changed files, use them directly.
    new_files = changed_files

    # Optionally update missing labels in your DB for older PRs.
    update_missing_labels(db_path=args.db_path, yaml_path="new-prs-labeler.yml")
    
    # Load historical data.
    prs_df, files_df, reviews_df = load_data(db_path=args.db_path)
    print("Data loaded from", args.db_path)
    
    # Build reviewer data.
    reviewer_pr_data = build_reviewer_pr_data(prs_df, files_df, reviews_df)
    if not reviewer_pr_data:
        print("No reviewer PR data found. Exiting.")
        return
    
    # Build reviewer graph and compute PageRank.
    G = build_reviewer_graph(reviews_df)
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    print("PageRank scores computed.")
    
    # Compute dynamic activity scores.
    activity_scores = compute_dynamic_activity_scores(reviews_df, tau=30)
    print("Dynamic activity scores computed.")
    
    # Rank reviewers.
    ranked_reviewers = rank_reviewers(new_tags, new_files, reviewer_pr_data, pagerank_scores, activity_scores,
                                      w1=1, w2=2, weight_sim_act=1.0, gamma=0.2, delta=0.001, db_path=args.db_path)
    
    # Build the comment.
    comment_lines = ["**Reviewer Recommendations:**"]
    for i, (rev, raw_abs, norm_abs, act_score, pr_score, fav_points, norm_fav, final_score) in enumerate(ranked_reviewers[:15], start=1):
        line = (f"{i}. **{rev}** | abs_match: {raw_abs}, norm_abs: {norm_abs:.4f}, "
                f"activity: {act_score:.4f}, PageRank: {pr_score:.4f}, fav_points: {fav_points}, "
                f"norm_fav: {norm_fav:.4f}, final: {final_score:.4f}")
        comment_lines.append(line)
    comment_lines.append("\n_To provide feedback, comment with `/feedback reviewer_username`._")
    comment_body = "\n".join(comment_lines)
    
    print("Final comment body:\n", comment_body)  # Debug print
    post_comment(args.pr_number, comment_body)

if __name__ == "__main__":
    main()

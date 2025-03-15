import sqlite3
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta

def load_data(db_path="pr_data.db"):
    """
    Loads pull_requests, pr_files, and reviews tables from the SQLite database.
    Returns them as DataFrames for potential reuse, though in this approach
    we'll do a direct SQL query for (reviewer, labels, file_path).
    """
    conn = sqlite3.connect(db_path)
    prs_df = pd.read_sql_query("SELECT pr_id, labels FROM pull_requests", conn)
    files_df = pd.read_sql_query("SELECT pr_id, file_path FROM pr_files", conn)
    reviews_df = pd.read_sql_query("SELECT pr_id, reviewer, review_date, state FROM reviews", conn)
    conn.close()
    return prs_df, files_df, reviews_df

def get_new_pr_details():
    """
    Prompt user for new PR file paths and labels, returning them as sets for easy membership checks.
    """
    print("Enter the file paths for the new PR (comma-separated):")
    files_input = input("> ").strip()
    print("Enter the labels for the new PR (comma-separated):")
    labels_input = input("> ").strip()
    
    new_files = {f.strip() for f in files_input.split(",") if f.strip()}
    new_labels = {l.strip() for l in labels_input.split(",") if l.strip()}
    
    return new_labels, new_files

def build_reviewer_graph(reviews_df):
    """
    Build an undirected graph where nodes are reviewers and an edge exists between two reviewers
    if they co-reviewed the same PR. Edge weight is the number of co-reviews.
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

def compute_activity_scores(reviews_df, window_days=30):
    """
    Compute an activity score for each reviewer based on review actions in the last 'window_days'.
    Different review states are assigned different point values, then normalized to [0..1].
    """
    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"], errors="coerce", utc=True)
    cutoff_utc = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
    recent_reviews = reviews_df[reviews_df["review_date"] >= cutoff_utc]
    
    def points_for_state(state):
        s = state.upper() if isinstance(state, str) else ""
        if s == "APPROVED":
            return 3
        elif s == "CHANGES_REQUESTED":
            return 2
        elif s == "COMMENTED":
            return 1
        elif s == "COMMIT":
            return 2
        else:
            return 0
    
    recent_reviews["points"] = recent_reviews["state"].apply(points_for_state)
    raw_scores = recent_reviews.groupby("reviewer")["points"].sum().to_dict()
    
    if raw_scores:
        max_score = max(raw_scores.values())
        activity_scores = {r: (score / max_score) for r, score in raw_scores.items()}
    else:
        activity_scores = {}
    return activity_scores

def compute_line_based_matches(db_path, new_labels, new_files):
    """
    For each reviewer, count how many DB rows match ANY of the new_labels or new_files.
    Then return a dict: match_count[reviewer].
    
    'Match' means:
      - At least one label in row_labels is in new_labels, OR
      - The file_path is in new_files.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # This query gets all rows for all reviewers
    # with their labels and file_path
    # We'll parse labels per row, then check overlap.
    query = """
    SELECT r.reviewer, p.labels, f.file_path
    FROM reviews AS r
    JOIN pull_requests AS p ON r.pr_id = p.pr_id
    JOIN pr_files AS f ON p.pr_id = f.pr_id
    """
    rows = cursor.execute(query).fetchall()
    conn.close()
    
    match_count = {}
    
    for (reviewer, labels_str, file_path) in rows:
        if reviewer not in match_count:
            match_count[reviewer] = 0
        
        # Parse the row's labels (split by comma)
        if labels_str is None:
            row_labels = set()
        else:
            row_labels = {lbl.strip() for lbl in labels_str.split(",") if lbl.strip()}
        
        # Condition: if ANY label in row_labels is in new_labels OR file_path is in new_files
        if (row_labels.intersection(new_labels)) or (file_path in new_files):
            match_count[reviewer] += 1
    
    return match_count

def rank_reviewers(match_count, pagerank_scores, activity_scores,
                   alpha=0.5, beta=0.3, gamma=0.2):
    """
    1) Let max_count = max of all match_count values
    2) sim_score(reviewer) = match_count[reviewer] / max_count  (so top absolute matches => 1.0)
    3) final_score = alpha*sim + beta*act + gamma*pr
    Returns a list of (reviewer, sim_score, act_score, pr_score, final_score).
    """
    if not match_count:
        return []
    max_count = max(match_count.values())
    if max_count == 0:
        # if everything is 0, all sim_scores are 0
        max_count = 1
    
    results = []
    for reviewer, mcount in match_count.items():
        sim_score = mcount / float(max_count)
        act_score = activity_scores.get(reviewer, 0.0)
        pr_score  = pagerank_scores.get(reviewer, 0.0)
        
        final_score = alpha*sim_score + beta*act_score + gamma*pr_score
        results.append((reviewer, sim_score, act_score, pr_score, final_score))
    
    # sort by final_score descending
    results.sort(key=lambda x: x[4], reverse=True)
    return results

def main():
    db_path = "pr_data.db"
    
    # 1) Load data (optional for building the graph + activity)
    prs_df, files_df, reviews_df = load_data(db_path)
    print("Data loaded from pr_data.db.")
    
    # 2) Prompt for new PR details (labels, files)
    new_labels, new_files = get_new_pr_details()
    if not new_labels and not new_files:
        print("No new PR details provided. Exiting.")
        return
    
    # 3) Build the reviewer graph & compute PageRank
    G = build_reviewer_graph(reviews_df)
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    print("PageRank scores computed.")
    
    # 4) Compute activity scores
    activity_scores = compute_activity_scores(reviews_df, window_days=30)
    print("Activity scores computed.")
    
    # 5) Count line-based matches for each reviewer
    match_count = compute_line_based_matches(db_path, new_labels, new_files)
    if not match_count:
        print("No matches found for any reviewer.")
        return
    
    # 6) Rank reviewers by absolute match_count (normalized) + activity + pagerank
    ranked_reviewers = rank_reviewers(match_count, pagerank_scores, activity_scores,
                                      alpha=0.5, beta=0.3, gamma=0.2)
    
    # 7) Print top 10
    print("\n=== Top Reviewer Recommendations (Line-based absolute matches) ===")
    for i, (rev, sim_score, act_score, pr_score, final_score) in enumerate(ranked_reviewers[:10], start=1):
        print(f"{i}. Reviewer: {rev:20s} | "
              f"sim={sim_score:.2f}, act={act_score:.2f}, pr={pr_score:.2f}, final={final_score:.2f}")
    
    # 8) Export all to Excel
    df = pd.DataFrame(ranked_reviewers,
                      columns=["reviewer", "similarity_score", "activity_score", "pagerank_score", "final_score"])
    excel_name = "reviewer_scores_overlap_line_based.xlsx"
    df.to_excel(excel_name, index=False)
    print(f"\nAll reviewer scores have been saved to '{excel_name}'.")

if __name__ == "__main__":
    main()

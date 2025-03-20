import sqlite3
import pandas as pd
import networkx as nx
import math
from datetime import datetime, timedelta

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

def build_reviewer_documents(prs_df, files_df, reviews_df):
    """
    Build a text document for each reviewer by combining file paths and labels
    from PRs they reviewed.
    
    Returns a dictionary: { reviewer: "combined text of labels + file paths" }
    """
    # Group file paths by PR and combine them into one string per PR.
    files_grouped = files_df.groupby("pr_id")["file_path"].apply(lambda x: " ".join(x)).reset_index()
    files_grouped.rename(columns={"file_path": "all_files"}, inplace=True)
    
    # Merge file paths with PR data to get labels in the same row.
    prs_merged = pd.merge(prs_df, files_grouped, on="pr_id", how="left")
    prs_merged["all_files"] = prs_merged["all_files"].fillna("")
    
    # Combine labels and file paths into a single text field for each PR.
    prs_merged["pr_text"] = prs_merged["labels"].fillna("") + " " + prs_merged["all_files"]
    
    # Merge with reviews so each review row includes the PR text.
    merged_reviews = pd.merge(reviews_df, prs_merged, on="pr_id", how="left")
    
    # Group by reviewer to combine all reviewed PR texts into one big document per reviewer.
    reviewer_docs = merged_reviews.groupby("reviewer")["pr_text"].apply(lambda texts: " ".join(texts)).to_dict()
    return reviewer_docs

def get_new_pr_doc():
    """
    Prompt the user for new PR details (file paths and labels) and combine them into a single document.
    """
    print("Enter the file paths for the new PR (comma-separated):")
    files_input = input("> ").strip()
    print("Enter the labels for the new PR (comma-separated):")
    labels_input = input("> ").strip()
    
    # Clean and combine user input into a single string.
    files_str = " ".join([f.strip() for f in files_input.split(",") if f.strip()])
    labels_str = " ".join([l.strip() for l in labels_input.split(",") if l.strip()])
    
    new_pr_doc = labels_str + " " + files_str
    return new_pr_doc.strip()

def build_reviewer_graph(reviews_df):
    """
    Build an undirected graph where nodes are reviewers and an edge exists between
    two reviewers if they co-reviewed the same PR. The edge weight is the number
    of co-reviews shared.
    """
    G = nx.Graph()
    reviewers = reviews_df["reviewer"].unique()
    G.add_nodes_from(reviewers)
    
    # Group by pr_id to find sets of reviewers who worked on the same PR.
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
    
    The formula for each review is:
        partial_score = base_points * exp(-days_since_review / tau)
    where base_points is 1 for any valid review action (APPROVED, COMMENTED, etc.).
    
    We then sum partial scores per reviewer, and normalize so the highest sum = 1.
    """
    # Convert review_date to a datetime object.
    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"], errors="coerce", utc=True)
    now_utc = pd.Timestamp.utcnow()
    
    def points_for_state(state):
        s = state.upper() if isinstance(state, str) else ""
        # Assign 1 point for typical review actions
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
        
        # Calculate how many days have passed since the review.
        days_diff = (now_utc - review_date).total_seconds() / 86400.0  # Convert seconds to days
        decay = math.exp(-days_diff / float(tau))  # Exponential decay factor
        
        partial_score = base_points * decay
        reviewer_scores[reviewer] = reviewer_scores.get(reviewer, 0.0) + partial_score

    # Normalize scores so that the max activity score is 1.
    if reviewer_scores:
        max_score = max(reviewer_scores.values())
        activity_scores = {r: (score / max_score) for r, score in reviewer_scores.items()}
    else:
        activity_scores = {}
    
    return activity_scores

def compute_absolute_similarity(new_pr_doc, reviewer_doc):
    """
    Compute the absolute matching score between the new PR document and a reviewer's document.
    
    - Tokenize both documents (lowercase split on whitespace).
    - Count how many tokens in the new PR doc appear in the reviewer’s doc, summing their frequencies.
    """
    # Tokenize to lowercase
    new_tokens = new_pr_doc.lower().split()
    reviewer_tokens = reviewer_doc.lower().split()
    
    # Build a frequency map for tokens in the reviewer's document
    freq_map = {}
    for token in reviewer_tokens:
        freq_map[token] = freq_map.get(token, 0) + 1
    
    # Sum frequencies for all tokens that appear in new_tokens
    abs_score = 0
    for token in new_tokens:
        abs_score += freq_map.get(token, 0)
    
    return abs_score

def rank_reviewers(new_pr_doc, reviewer_docs, pagerank_scores, activity_scores,
                   weight_sim_act=1.0,  # weight for (normalized_abs_match_score * activity_score)
                   gamma=0.2           # weight for PageRank
                  ):
    """
    1) Compute the raw absolute match score for each reviewer.
    2) Normalize those scores between 0 and 1.
    3) Combine with activity and PageRank using:
         final_score = weight_sim_act * (normalized_abs_match_score * act_score) + gamma * pr_score
    
    Returns a sorted list of tuples:
      (reviewer, raw_abs_match, normalized_abs_match, activity_score, pagerank_score, final_score)
    """
    # Step 1: Calculate raw absolute matching for each reviewer
    abs_scores = {}
    for reviewer, doc in reviewer_docs.items():
        abs_match_score = compute_absolute_similarity(new_pr_doc, doc)
        abs_scores[reviewer] = abs_match_score

    # Step 2: Find the maximum raw score to normalize
    max_abs = max(abs_scores.values()) if abs_scores else 0

    results = []
    for reviewer, raw_abs_score in abs_scores.items():
        # Normalize the absolute matching score (handle the case if max_abs is 0)
        norm_abs_score = (raw_abs_score / max_abs) if max_abs > 0 else 0.0
        
        # Retrieve PageRank and activity scores
        pr_score = pagerank_scores.get(reviewer, 0.0)
        act_score = activity_scores.get(reviewer, 0.0)
        
        # Combine normalized_abs_score with activity, then add PageRank
        sim_act_product = norm_abs_score * act_score
        final_score = weight_sim_act * sim_act_product + gamma * pr_score
        
        # Store all data in a tuple
        results.append((reviewer, raw_abs_score, norm_abs_score, act_score, pr_score, final_score))
    
    # Step 3: Sort reviewers by final_score in descending order
    results.sort(key=lambda x: x[5], reverse=True)
    return results

def main():
    # 1) Load data from the database
    prs_df, files_df, reviews_df = load_data("pr_data.db")
    print("Data loaded from pr_data.db.")
    
    # 2) Build reviewer documents (combined text for each reviewer)
    reviewer_docs = build_reviewer_documents(prs_df, files_df, reviews_df)
    if not reviewer_docs:
        print("No reviewer documents found. Please check your database content.")
        return
    
    # 3) Build a reviewer graph and compute PageRank scores
    G = build_reviewer_graph(reviews_df)
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    print("PageRank scores computed.")
    
    # 4) Compute dynamic activity scores (exponential decay)
    activity_scores = compute_dynamic_activity_scores(reviews_df, tau=30)
    print("Dynamic activity scores computed.")
    
    # 5) Prompt for new PR details
    print("\n=== Enter new PR details ===")
    new_pr_doc = get_new_pr_doc()
    if not new_pr_doc:
        print("No new PR information provided. Exiting.")
        return
    
    # 6) Rank reviewers by combining normalized absolute matching, activity, and PageRank
    ranked_reviewers = rank_reviewers(
        new_pr_doc, 
        reviewer_docs, 
        pagerank_scores, 
        activity_scores,
        weight_sim_act=1.0,  # You can adjust this weight
        gamma=0.2            # You can adjust this weight
    )
    
    # 7) Print top 10 recommendations
    print("\n=== Top Reviewer Recommendations ===")
    for i, (rev, raw_abs, norm_abs, act_score, pr_score, final_score) in enumerate(ranked_reviewers[:10], start=1):
        print(
            f"{i}. Reviewer: {rev:20s} | "
            f"abs_match={raw_abs}, norm_abs={norm_abs:.4f}, act={act_score:.4f}, "
            f"pr={pr_score:.4f}, final={final_score:.4f}"
        )
    
    # 8) Save all reviewer scores to an Excel file
    df = pd.DataFrame(
        ranked_reviewers,
        columns=["reviewer", "raw_abs_match", "normalized_abs_match", "activity_score", "pagerank_score", "final_score"]
    )
    excel_name = "reviewer_scores_absolute_normalized.xlsx"
    df.to_excel(excel_name, index=False)
    print(f"\nAll reviewer scores have been saved to '{excel_name}'.")

if __name__ == "__main__":
    main()

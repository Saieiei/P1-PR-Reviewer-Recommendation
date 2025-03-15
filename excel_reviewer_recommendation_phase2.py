import sqlite3
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    Build a text document for each reviewer by combining file paths and labels from PRs they reviewed.
    """
    # Group file paths by PR, concatenating them into one string per PR.
    files_grouped = files_df.groupby("pr_id")["file_path"].apply(lambda x: " ".join(x)).reset_index()
    files_grouped.rename(columns={"file_path": "all_files"}, inplace=True)
    
    # Merge with PR data to get labels.
    prs_merged = pd.merge(prs_df, files_grouped, on="pr_id", how="left")
    prs_merged["all_files"] = prs_merged["all_files"].fillna("")
    
    # Build a combined text for each PR (labels + file paths).
    prs_merged["pr_text"] = prs_merged["labels"].fillna("") + " " + prs_merged["all_files"]
    
    # Merge with reviews so that each review row includes the PR text.
    merged_reviews = pd.merge(reviews_df, prs_merged, on="pr_id", how="left")
    
    # Group by reviewer to combine all PR texts into one document per reviewer.
    reviewer_docs = merged_reviews.groupby("reviewer")["pr_text"].apply(lambda texts: " ".join(texts)).to_dict()
    return reviewer_docs

def fit_tfidf(reviewer_docs):
    """
    Given a dictionary {reviewer: document_text}, fit a TfidfVectorizer.
    Returns (reviewers, vectorizer, tfidf_matrix).
    """
    reviewers = list(reviewer_docs.keys())
    documents = [reviewer_docs[rev] for rev in reviewers]
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"[A-Za-z0-9_\-/\.]+"
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return reviewers, vectorizer, tfidf_matrix

def get_new_pr_doc():
    """
    Prompt user for new PR details (file paths and labels) and combine them into a single document string.
    """
    print("Enter the file paths for the new PR (comma-separated):")
    files_input = input("> ").strip()
    print("Enter the labels for the new PR (comma-separated):")
    labels_input = input("> ").strip()
    
    files_str = " ".join([f.strip() for f in files_input.split(",") if f.strip()])
    labels_str = " ".join([l.strip() for l in labels_input.split(",") if l.strip()])
    
    new_pr_doc = labels_str + " " + files_str
    return new_pr_doc.strip()

def build_reviewer_graph(reviews_df):
    """
    Build an undirected graph where nodes are reviewers and an edge exists between two reviewers
    if they co-reviewed the same PR. Edge weight is the number of co-reviews.
    """
    import networkx as nx
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
    Different review states are assigned different point values.
    """
    # Convert to datetime in UTC
    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"], errors="coerce", utc=True)
    # Make cutoff also UTC
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
    
    # Normalize scores to [0, 1].
    if raw_scores:
        max_score = max(raw_scores.values())
        activity_scores = {r: (score / max_score) for r, score in raw_scores.items()}
    else:
        activity_scores = {}
    return activity_scores

def rank_reviewers(new_pr_doc, reviewers, vectorizer, tfidf_matrix, pagerank_scores, activity_scores,
                   alpha=0.5, beta=0.3, gamma=0.2):
    """
    For the new PR document, compute its cosine similarity to each reviewer's document,
    and then combine the similarity score with PageRank and activity scores using a weighted sum.
    
    Returns a list of tuples:
        (reviewer, sim_score, act_score, pr_score, final_score)
    sorted in descending order by final_score.
    """
    new_vec = vectorizer.transform([new_pr_doc])
    similarities = cosine_similarity(new_vec, tfidf_matrix)[0]
    
    results = []
    for i, reviewer in enumerate(reviewers):
        sim_score = similarities[i]
        pr_score = pagerank_scores.get(reviewer, 0.0)
        act_score = activity_scores.get(reviewer, 0.0)
        final_score = alpha * sim_score + beta * act_score + gamma * pr_score
        results.append((reviewer, sim_score, act_score, pr_score, final_score))
    
    # Sort by final_score descending
    results.sort(key=lambda x: x[4], reverse=True)
    return results

def main():
    # 1) Load data from the database
    prs_df, files_df, reviews_df = load_data("pr_data.db")
    print("Data loaded from pr_data.db.")
    
    # 2) Build reviewer documents
    reviewer_docs = build_reviewer_documents(prs_df, files_df, reviews_df)
    if not reviewer_docs:
        print("No reviewer documents found. Please check your database content.")
        return
    
    # 3) Fit TF-IDF
    reviewers, vectorizer, tfidf_matrix = fit_tfidf(reviewer_docs)
    print(f"TF-IDF matrix built for {len(reviewers)} reviewers.")
    
    # 4) Build graph & compute PageRank
    G = build_reviewer_graph(reviews_df)
    pagerank_scores = nx.pagerank(G, alpha=0.85, weight="weight")
    print("PageRank scores computed.")
    
    # 5) Compute activity scores
    activity_scores = compute_activity_scores(reviews_df, window_days=30)
    print("Activity scores computed.")
    
    # 6) Prompt for new PR data
    print("\n=== Enter new PR details ===")
    new_pr_doc = get_new_pr_doc()
    if not new_pr_doc:
        print("No new PR information provided. Exiting.")
        return
    
    # 7) Rank reviewers
    ranked_reviewers = rank_reviewers(
        new_pr_doc,
        reviewers,
        vectorizer,
        tfidf_matrix,
        pagerank_scores,
        activity_scores,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    )
    
    # ranked_reviewers is a list of tuples:
    # (reviewer, sim_score, act_score, pr_score, final_score)
    
    # 8) Print top 10 in console
    print("\n=== Top Reviewer Recommendations ===")
    for i, (rev, sim_score, act_score, pr_score, final_score) in enumerate(ranked_reviewers[:10], start=1):
        print(f"{i}. Reviewer: {rev:20s} | "
              f"sim={sim_score:.4f}, act={act_score:.4f}, pr={pr_score:.4f}, final={final_score:.4f}")
    
    # 9) Export ALL reviewers to Excel
    df = pd.DataFrame(ranked_reviewers,
                      columns=["reviewer", "similarity_score", "activity_score", "pagerank_score", "final_score"])
    excel_name = "reviewer_scores.xlsx"
    df.to_excel(excel_name, index=False)
    print(f"\nAll reviewer scores have been saved to '{excel_name}'.")

if __name__ == "__main__":
    main()

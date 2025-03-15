import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(db_path="pr_data.db"):
    """
    Loads pull_requests, pr_files, and reviews from the SQLite database into pandas DataFrames.
    Returns (prs_df, files_df, reviews_df).
    """
    conn = sqlite3.connect(db_path)
    prs_df = pd.read_sql_query("SELECT pr_id, labels FROM pull_requests", conn)
    files_df = pd.read_sql_query("SELECT pr_id, file_path FROM pr_files", conn)
    reviews_df = pd.read_sql_query("SELECT pr_id, reviewer, state FROM reviews", conn)
    conn.close()
    return prs_df, files_df, reviews_df

def build_reviewer_documents(prs_df, files_df, reviews_df):
    """
    Build a text "document" for each reviewer by combining:
      - File paths from PRs they reviewed
      - Labels from PRs they reviewed
    Returns a dict: { reviewer: "combined text" }
    """
    # Merge PR data with files on pr_id
    # We'll group the file paths for each PR into one string
    files_grouped = files_df.groupby("pr_id")["file_path"].apply(lambda x: " ".join(x)).reset_index()
    files_grouped.rename(columns={"file_path": "all_files"}, inplace=True)
    
    # Merge with PRs to get labels, too
    prs_merged = pd.merge(prs_df, files_grouped, on="pr_id", how="left")
    prs_merged["all_files"] = prs_merged["all_files"].fillna("")
    
    # For each PR, build a text chunk: (labels + file paths)
    # e.g., "bug clang src/foo.cpp docs/readme.md"
    prs_merged["pr_text"] = prs_merged["labels"].fillna("") + " " + prs_merged["all_files"]
    
    # Now merge with reviews_df to see who reviewed each PR
    merged_reviews = pd.merge(reviews_df, prs_merged, on="pr_id", how="left")
    
    # Group by reviewer, concatenate all "pr_text" for that reviewer
    # This is each reviewer's "document"
    reviewer_docs = merged_reviews.groupby("reviewer")["pr_text"].apply(lambda texts: " ".join(texts)).to_dict()
    
    return reviewer_docs

def fit_tfidf(reviewer_docs):
    """
    Given a dict of {reviewer: doc_text}, returns:
      - A list of reviewers
      - The fitted TfidfVectorizer
      - The TF-IDF matrix (reviewer_count x vocab_size)
    """
    reviewers = list(reviewer_docs.keys())
    documents = [reviewer_docs[rev] for rev in reviewers]
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"[A-Za-z0-9_\-/\.]+"  # allow file paths, e.g. src/foo.cpp
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return reviewers, vectorizer, tfidf_matrix

def get_new_pr_text():
    """
    Prompt user for new PR info (file paths, labels).
    Combine them into one text string for TF-IDF.
    """
    print("Enter the file paths for the new PR (comma-separated):")
    files_input = input("> ").strip()
    print("Enter the labels for the new PR (comma-separated):")
    labels_input = input("> ").strip()
    
    # e.g., "src/foo.cpp, src/bar.cpp" -> "src/foo.cpp src/bar.cpp"
    files_str = " ".join([f.strip() for f in files_input.split(",") if f.strip()])
    labels_str = " ".join([l.strip() for l in labels_input.split(",") if l.strip()])
    
    # Combine them into one text doc
    new_pr_doc = labels_str + " " + files_str
    return new_pr_doc.strip()

def rank_reviewers_by_similarity(
    reviewers,
    vectorizer,
    tfidf_matrix,
    new_pr_doc
):
    """
    Given a new PR doc, compute similarity with each reviewer doc.
    Returns a list of (reviewer, similarity_score) sorted desc by score.
    """
    new_vec = vectorizer.transform([new_pr_doc])
    similarities = cosine_similarity(new_vec, tfidf_matrix)[0]
    
    # Pair each reviewer with the similarity score
    reviewer_scores = list(zip(reviewers, similarities))
    # Sort by score descending
    reviewer_scores.sort(key=lambda x: x[1], reverse=True)
    return reviewer_scores

def main():
    # 1. Load data from DB
    prs_df, files_df, reviews_df = load_data("pr_data.db")
    
    # 2. Build each reviewer's document
    reviewer_docs = build_reviewer_documents(prs_df, files_df, reviews_df)
    if not reviewer_docs:
        print("No reviewer documents found. Please check your database content.")
        return
    
    # 3. Fit TF-IDF on reviewer documents
    reviewers, vectorizer, tfidf_matrix = fit_tfidf(reviewer_docs)
    print(f"Built TF-IDF matrix for {len(reviewers)} reviewers.")
    
    # 4. Prompt user for new PR's file paths and labels
    print("\n=== Enter details for a new PR to get reviewer suggestions ===")
    new_pr_doc = get_new_pr_text()
    if not new_pr_doc:
        print("No input provided. Exiting.")
        return
    
    # 5. Compute similarity
    reviewer_scores = rank_reviewers_by_similarity(reviewers, vectorizer, tfidf_matrix, new_pr_doc)
    
    # 6. Print top reviewers
    print("\n=== Top Reviewers by Similarity ===")
    for i, (rev, score) in enumerate(reviewer_scores[:10], start=1):
        print(f"{i}. Reviewer: {rev:20s} | Similarity: {score:.4f}")

if __name__ == "__main__":
    main()

import os
import requests
import sqlite3
from urllib.parse import quote

###############################################################################
# 1) DATABASE SETUP
###############################################################################
def get_db_connection(db_path="pr_data.db"):
    """
    Opens (or creates) a SQLite database named 'pr_data.db'.
    Returns the connection object.
    """
    conn = sqlite3.connect(db_path)
    return conn

def create_tables(conn):
    """
    Creates the following tables if they do not exist:
      - pull_requests
      - pr_files
      - reviews

    Includes UNIQUE constraints to avoid duplicates.
    """
    create_pull_requests_table = """
    CREATE TABLE IF NOT EXISTS pull_requests (
        pr_id INTEGER PRIMARY KEY,
        title TEXT,
        user_login TEXT,
        labels TEXT,
        created_at TEXT,
        updated_at TEXT
    );
    """
    create_pr_files_table = """
    CREATE TABLE IF NOT EXISTS pr_files (
        pr_id INTEGER,
        file_path TEXT,
        UNIQUE(pr_id, file_path)
    );
    """
    create_reviews_table = """
    CREATE TABLE IF NOT EXISTS reviews (
        pr_id INTEGER,
        reviewer TEXT,
        review_date TEXT,
        UNIQUE(pr_id, reviewer, review_date)
    );
    """
    with conn:
        conn.execute(create_pull_requests_table)
        conn.execute(create_pr_files_table)
        conn.execute(create_reviews_table)

###############################################################################
# 2) SEARCH API FOR PULL REQUESTS WITH DATE RANGE
###############################################################################
def search_pull_requests(owner, repo, start_date, end_date):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("No GitHub token found. Please set GITHUB_TOKEN environment variable.")

    query = f"repo:{owner}/{repo} is:pr created:{start_date}..{end_date} state:all"
    per_page = 100
    page = 1
    all_items = []

    while True:
        url = f"https://api.github.com/search/issues?q={quote(query)}&per_page={per_page}&page={page}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }

        # Make the request
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Search request failed: {response.status_code} {response.text}")

        # Parse JSON
        data = response.json()

        # DEBUG PRINTS: See exactly what we got back
        print("Search URL:", url)
        print("Response JSON:", data)

        items = data.get("items", [])
        all_items.extend(items)

        # If fewer than 'per_page' results, we're done
        if len(items) < per_page:
            break

        page += 1
        if page > 50:
            print("Reached 50 pages of results, stopping.")
            break

    return all_items

###############################################################################
# 3) API FETCHING FOR FILES, REVIEWS, AND FULL PR DATA
###############################################################################
def fetch_pr_data(owner, repo, pr_number):
    """
    Fetch the full pull request data from the standard endpoint,
    needed for detailed fields (labels, created_at, etc.).
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("No GitHub token found. Please set GITHUB_TOKEN environment variable.")

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PR #{pr_number} details: {response.status_code} {response.text}")
    return response.json()

def fetch_pr_files(owner, repo, pr_number):
    """
    Fetch the list of changed files for a given PR (by number).
    Returns a list of file objects.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("No GitHub token found. Please set GITHUB_TOKEN environment variable.")

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request for PR #{pr_number} files failed: {response.status_code} {response.text}")
    return response.json()

def fetch_pr_reviews(owner, repo, pr_number):
    """
    Fetch the list of reviews for a given PR (by number).
    Returns a list of review objects.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("No GitHub token found. Please set GITHUB_TOKEN environment variable.")

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request for PR #{pr_number} reviews failed: {response.status_code} {response.text}")
    return response.json()

###############################################################################
# 4) DB INSERT FUNCTIONS
###############################################################################
def insert_pull_request(conn, pr):
    """
    Inserts or replaces a pull request record into the pull_requests table.
    pr is the full PR object from the standard endpoint.
    """
    pr_id = pr["number"]
    title = pr.get("title", "")
    user_login = pr.get("user", {}).get("login", "")
    labels = pr.get("labels", [])
    labels_str = ",".join([label.get("name", "") for label in labels])
    created_at = pr.get("created_at", "")
    updated_at = pr.get("updated_at", "")

    sql = """
    INSERT OR REPLACE INTO pull_requests
    (pr_id, title, user_login, labels, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    with conn:
        conn.execute(sql, (pr_id, title, user_login, labels_str, created_at, updated_at))

def insert_pr_file(conn, pr_id, file_path):
    """
    Inserts a single file record into the pr_files table.
    Uses 'INSERT OR IGNORE' to avoid duplicates.
    """
    sql = "INSERT OR IGNORE INTO pr_files (pr_id, file_path) VALUES (?, ?);"
    with conn:
        conn.execute(sql, (pr_id, file_path))

def insert_review(conn, pr_id, reviewer, review_date):
    """
    Inserts a single review record into the reviews table.
    Uses 'INSERT OR IGNORE' to avoid duplicates.
    """
    sql = "INSERT OR IGNORE INTO reviews (pr_id, reviewer, review_date) VALUES (?, ?, ?);"
    with conn:
        conn.execute(sql, (pr_id, reviewer, review_date))

###############################################################################
# 5) MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    # 1) Prompt for GitHub details and date range
    owner = "llvm"  # your username
    repo = "llvm-project"
    print(f"Using repository: {owner}/{repo}")
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    # 2) Connect to pr_data.db and create tables
    conn = get_db_connection("pr_data.db")
    create_tables(conn)

    # 3) Search for PRs in the specified date range using the Search API
    print(f"Searching for PRs created between {start_date} and {end_date}...")
    pr_items = search_pull_requests(owner, repo, start_date, end_date)
    print(f"Found {len(pr_items)} PRs in that range.")

    # 4) For each PR, fetch full data, then fetch files and reviews, and insert into DB
    for item in pr_items:
        pr_number = item["number"]
        pr_title = item.get("title", "")
        print(f"\nProcessing PR #{pr_number}: {pr_title}")

        # (A) Fetch full PR data and insert
        pr_data = fetch_pr_data(owner, repo, pr_number)
        insert_pull_request(conn, pr_data)

        # (B) Fetch changed files and insert
        files_list = fetch_pr_files(owner, repo, pr_number)
        for file_item in files_list:
            file_path = file_item.get("filename", "")
            insert_pr_file(conn, pr_number, file_path)

        # (C) Fetch reviews and insert
        reviews_list = fetch_pr_reviews(owner, repo, pr_number)
        for review_item in reviews_list:
            user_info = review_item.get("user", {})
            reviewer_name = user_info.get("login", "")
            review_date = review_item.get("submitted_at", "")
            insert_review(conn, pr_number, reviewer_name, review_date)

    # 5) Close the database connection
    conn.close()
    print("\nData collection complete!")

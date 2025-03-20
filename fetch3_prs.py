import requests
import sqlite3
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3
import sys

# If you have SSL issues behind a corporate firewall, you can disable warnings:
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
#  CONFIGURE THESE VALUES
# ==============================
GITHUB_TOKEN = "ghp"     # Replace with your actual GitHub token
OWNER = "llvm"                       # Repository owner
REPO = "llvm-project"                # Repository name
DATABASE_FILE = "pr_data.db"         # SQLite database file

def create_tables_if_needed(conn):
    """
    Optional helper to create the necessary tables if they don't exist.
    Adjust the schema if your columns differ.
    """
    cursor = conn.cursor()
    
    # Create pull_requests table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pull_requests (
            pr_id INTEGER PRIMARY KEY,
            title TEXT,
            user_login TEXT,
            labels TEXT,
            created_at TEXT,
            updated_at TEXT
        );
    """)
    
    # Create pr_files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pr_files (
            pr_id INTEGER,
            file_path TEXT
        );
    """)
    
    # Create reviews table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            pr_id INTEGER,
            reviewer TEXT,
            review_date TEXT
        );
    """)
    
    conn.commit()

def get_prs_in_date_range(start_date, end_date, verify_ssl=True):
    """
    Fetch pull requests from the GitHub Search API in the specified date range.
    Returns a list of PR "issue" objects.
    
    Note: The GitHub Search API is limited to 1000 results per query.
    If your date range yields more than 1000 PRs, you'll need to split it.
    """
    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    })
    session.verify = verify_ssl
    
    # Set up retries for transient errors
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    
    all_prs = []
    page = 1
    per_page = 100
    search_url = "https://api.github.com/search/issues"
    
    while True:
        # Search for PRs in the date range, sorted by creation date
        # For example: created:2023-01-01..2023-12-31
        query = f"repo:{OWNER}/{REPO} is:pr created:{start_date}..{end_date}"
        params = {
            "q": query,
            "sort": "created",
            "order": "asc",
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PRs: {e}")
            break
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            break
        
        all_prs.extend(items)
        
        # If fewer than per_page returned, no more pages
        if len(items) < per_page:
            break
        
        # Watch out for the 1000 results limit.
        if page * per_page >= 1000:
            print("Warning: You have likely hit the 1000-result limit of the GitHub Search API.")
            break
        
        page += 1
    
    return all_prs

def fetch_pr_details(pr_number, verify_ssl=True):
    """
    Fetch additional details for a single pull request:
      1. Files changed in the PR
      2. Reviews on the PR
    
    Returns:
      (list_of_files, list_of_reviews)
    """
    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    })
    session.verify = verify_ssl
    
    # Set up retries
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    
    # 1. Get PR files
    files_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}/files"
    try:
        files_response = session.get(files_url, timeout=10)
        files_response.raise_for_status()
        files_data = files_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching files for PR #{pr_number}: {e}")
        files_data = []
    
    # 2. Get PR reviews
    reviews_url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews"
    try:
        reviews_response = session.get(reviews_url, timeout=10)
        reviews_response.raise_for_status()
        reviews_data = reviews_response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews for PR #{pr_number}: {e}")
        reviews_data = []
    
    return files_data, reviews_data

def insert_data_into_db(conn, pr_list):
    """
    Given a list of PRs (from the Search API),
    insert into pull_requests, pr_files, and reviews.
    """
    cursor = conn.cursor()
    
    for pr_item in pr_list:
        # Each item is a "search result" issue object
        pr_number = pr_item["number"]
        pr_title = pr_item["title"]
        
        # "user" field might be missing if the PR is from an org, so handle carefully
        user_login = pr_item["user"]["login"] if "user" in pr_item and pr_item["user"] else "unknown"
        
        # Convert labels to a comma-separated string
        # The 'labels' array might be missing if no labels
        label_objs = pr_item.get("labels", [])
        # For the Search API, labels might be under pr_item["labels"] or pr_item["issue"]["labels"] 
        # but usually "labels" is correct. Check your data if needed.
        label_names = [label["name"] for label in label_objs] if label_objs else []
        labels_str = ",".join(label_names)
        
        created_at = pr_item.get("created_at", "")
        updated_at = pr_item.get("updated_at", "")
        
        # Insert/replace into pull_requests table
        cursor.execute("""
            INSERT OR IGNORE INTO pull_requests (pr_id, title, user_login, labels, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pr_number, pr_title, user_login, labels_str, created_at, updated_at))
        
        # Now fetch additional details: files and reviews
        files_data, reviews_data = fetch_pr_details(pr_number)
        
        # Insert each file into pr_files
        for f in files_data:
            file_path = f.get("filename", "")
            cursor.execute("""
                INSERT INTO pr_files (pr_id, file_path) VALUES (?, ?)
            """, (pr_number, file_path))
        
        # Insert each review into reviews
        for r in reviews_data:
            reviewer = r["user"]["login"] if "user" in r and r["user"] else "unknown"
            review_date = r.get("submitted_at", "")
            cursor.execute("""
                INSERT INTO reviews (pr_id, reviewer, review_date)
                VALUES (?, ?, ?)
            """, (pr_number, reviewer, review_date))
    
    conn.commit()

def main():
    # Prompt user for date range
    print("Enter date range in YYYY-MM-DD format.")
    start_date = input("Start date (e.g. 2023-01-01): ").strip()
    end_date = input("End date   (e.g. 2023-01-31): ").strip()
    
    # Connect to (or create) the database
    conn = sqlite3.connect(DATABASE_FILE)
    
    # OPTIONAL: Create tables if they don't exist. 
    # If you already have them, you can skip this call.
    create_tables_if_needed(conn)
    
    # Fetch PRs within the user-provided date range
    print(f"\nFetching PRs created from {start_date} to {end_date} ...")
    pr_list = get_prs_in_date_range(start_date, end_date, verify_ssl=True)
    print(f"Found {len(pr_list)} PRs in that range.")
    
    # Insert them into the DB
    print("Storing data in the database...")
    insert_data_into_db(conn, pr_list)
    
    print("Done!")
    conn.close()

if __name__ == "__main__":
    main()

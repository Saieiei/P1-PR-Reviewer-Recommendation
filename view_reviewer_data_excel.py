import sqlite3
import pandas as pd

def get_reviewer_data(reviewer_name, db_file="pr_data.db"):
    """
    Query the database to retrieve PRs, labels, and file paths for a given reviewer.
    
    This joins the 'reviews', 'pull_requests', and 'pr_files' tables.
    
    Parameters:
      reviewer_name (str): The reviewer's username (e.g., "erichkeane").
      db_file (str): Path to the SQLite database file.
    
    Returns:
      List of tuples: Each tuple contains (pr_id, labels, file_path).
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    query = """
    SELECT p.pr_id, p.labels, f.file_path
    FROM reviews AS r
    JOIN pull_requests AS p ON r.pr_id = p.pr_id
    JOIN pr_files AS f ON p.pr_id = f.pr_id
    WHERE r.reviewer = ?
    ORDER BY p.pr_id;
    """
    
    cursor.execute(query, (reviewer_name,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def main():
    reviewer = "nikic"  # Change to the reviewer you want to query
    data = get_reviewer_data(reviewer)
    
    if not data:
        print(f"No data found for reviewer: {reviewer}")
        return

    # Convert the data into a pandas DataFrame for easy Excel export
    df = pd.DataFrame(data, columns=["pr_id", "labels", "file_path"])

    # Write to an Excel file
    output_filename = f"{reviewer}_data.xlsx"
    df.to_excel(output_filename, index=False)
    
    print(f"Data for reviewer: {reviewer} has been saved to {output_filename}.")

if __name__ == "__main__":
    main()

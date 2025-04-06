import sqlite3

def clear_tables(db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Delete all rows from each table.
    cursor.execute("DELETE FROM pr_files")
    cursor.execute("DELETE FROM reviews")
    cursor.execute("DELETE FROM pull_requests")
    cursor.execute("DELETE FROM feedback")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    clear_tables()
    print("All data cleared from pr_files, reviews, and pull_requests tables.")

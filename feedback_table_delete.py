import sqlite3

def clear_feedback_table(db_path="pr_data.db"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Delete all rows from the 'feedback' table
        cursor.execute("DELETE FROM feedback")
        
        # Optionally reclaim space (uncomment if desired)
        # cursor.execute("VACUUM")

        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_feedback_table()
    print("All data cleared from feedback table.")

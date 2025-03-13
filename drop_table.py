import sqlite3

def drop_reviews_table(db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop the existing table
    cursor.execute("DROP TABLE IF EXISTS reviews")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    drop_reviews_table()
    print("Dropped the 'reviews' table. Now rerun your script to create it anew.")

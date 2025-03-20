#!/usr/bin/env python3
import os
import json
import sqlite3
import re

def update_feedback_for_reviewer(reviewer_name, increment=1, db_path="pr_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT fav_rev_points FROM feedback WHERE reviewer = ?", (reviewer_name,))
    row = cursor.fetchone()
    if row:
        new_points = row[0] + increment
        cursor.execute("UPDATE feedback SET fav_rev_points = ? WHERE reviewer = ?", (new_points, reviewer_name))
    else:
        cursor.execute("INSERT INTO feedback (reviewer, fav_rev_points) VALUES (?, ?)", (reviewer_name, increment))
    conn.commit()
    conn.close()

def main():
    # GitHub Actions stores the event payload in the file specified by GITHUB_EVENT_PATH.
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("GITHUB_EVENT_PATH not set.")
        return
    with open(event_path, "r") as f:
        event_data = json.load(f)
    
    comment_body = event_data.get("comment", {}).get("body", "")
    match = re.match(r"\/feedback\s+(\S+)", comment_body)
    if match:
        reviewer_name = match.group(1).strip()
        print(f"Processing feedback for reviewer: {reviewer_name}")
        update_feedback_for_reviewer(reviewer_name, increment=1, db_path="pr_data.db")
    else:
        print("No valid feedback command found in the comment.")

if __name__ == "__main__":
    main()

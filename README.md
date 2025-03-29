# GitHub Pull Request Analytics

This repository provides tools and workflows to help analyze and manage pull requests (PRs), reviewers, and feedback. Below you will find a description of each file, along with instructions on how to use them.

---

## Table of Contents
1. [Overview](#overview)
2. [File Descriptions](#file-descriptions)
3. [Usage](#usage)
4. [Workflows](#workflows)
5. [Database Schema](#database-schema)
6. [How to Use](#How-to-Use)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This project focuses on:
- Storing PR data from a given GitHub repository using `store_prs2.py`.
- Resetting and clearing data with `delete_tables_restart.py`.
- Generating recommendations for top reviewers using `ml_pm2_spda_fav_fs_t15_rr.py` and the `post_recommendations.yml` GitHub Actions workflow.
- Handling feedback to update reviewer scores with `process_feedback.yml` and `process_feedback.py`.
- Providing scripts like `view_reviewer_data_excel.py` to visualize or export reviewer data to Excel.

---

## File Descriptions

### 1. `config.ini`
- **Description:** Configuration file that stores GitHub API credentials, time-frame filters, database file location, and other parameters. 
- **Usage:** 
  - The `[github]` section includes your GitHub token, repository owner, and repository name.
  - The `[filters]` section sets the date range, merge/closed status, and required labels for fetching PR data.
  - The `[database]` section specifies the file name for the SQLite database (default: `pr_data.db`).

### 2. `delete_tables_restart.py`
- **Description:** A script that resets the database by clearing out all records from the `pr_files`, `reviews`, `pull_requests`, and `feedback` tables.
- **Usage:** 
  - Run `python delete_tables_restart.py` in the terminal.
  - All existing data in these tables will be permanently removed.

### 3. `ml_pm2_spda_fav_fs_t15_rr.py`
- **Description:** A standalone script that accesses `pr_data.db` and generates the top 15 reviewers based on certain metrics (feedback scores, number of PR reviews, etc.).
- **Usage:** 
  - Run `python ml_pm2_spda_fav_fs_t15_rr.py` in your terminal.
  - It will produce a list of top 15 reviewers according to the logic defined in the script.
  - **Note:** This script **does not** run automatically via GitHub workflows. It is intended for manual execution.

### 4. `new-prs-labeler.yml`
- **Description:** A GitHub workflow file that automates the labeling of new PRs if they do not have labels when they are opened.
- **Usage:** 
  - Placed in `.github/workflows/new-prs-labeler.yml`.
  - When a new PR is raised without labels, this workflow suggests or applies labels automatically.

### 5. `post_recommendations.yml` and `recommendation.py`
- **Description:** 
  - **`post_recommendations.yml`** is a GitHub Actions workflow file that runs after certain triggers (e.g., on a schedule or when a PR is opened/updated). It calls `recommendation.py`.
  - **`recommendation.py`** contains logic similar to `ml_pm2_spda_fav_fs_t15_rr.py` (without the feedback component) to suggest or display top reviewers for a PR.
- **Usage:** 
  - These files integrate with GitHub Actions to automatically provide reviewer recommendations. 
  - Check the `.github/workflows/post_recommendations.yml` file for triggers and job steps.

### 6. `pr_data.db`
- **Description:** SQLite database file where all PR data is stored.
- **Contains Tables:**
  - **`pull_requests`**: Holds basic PR information (ID, title, author, labels, timestamps).
  - **`reviews`**: Stores review entries (PR ID, reviewer, date, state).
  - **`pr_files`**: Tracks files associated with each PR (PR ID, file paths).
  - **`feedback`**: Records reviewer feedback (reviewer name, favorite reviewer points, etc.).
- **Usage:** 
  - Automatically created and updated by scripts like `store_prs2.py`, or any of the GitHub workflows that run database operations.

### 7. `process_feedback.py` and `process_feedback.yml`
- **Description:** 
  - **`process_feedback.py`** is a script that processes user feedback to update reviewer scores in the `feedback` table.
  - **`process_feedback.yml`** is the corresponding GitHub Actions workflow that can request and handle feedback (e.g., triggered when a user wants to update reviewer points).
- **Usage:** 
  - Check the `.github/workflows/process_feedback.yml` file for triggers and job configuration.
  - When feedback is provided (via GitHub issue comments or another method defined in the workflow), it calls `process_feedback.py` to adjust the database records.

### 8. `store_prs2.py`
- **Description:** 
  - Fetches PR data from GitHub using parameters in `config.ini` and stores them in `pr_data.db`.
- **Usage:** 
  - Run `python store_prs2.py` to retrieve PRs (according to the date range, status, labels, etc. in `config.ini`) and store them in the local database.

### 9. `view_reviewer_data_excel.py`
- **Description:** 
  - Exports reviewer data from the `pr_data.db` into an Excel file for easy viewing or further analysis.
- **Usage:** 
  - Run `python view_reviewer_data_excel.py` to generate an Excel file (e.g., `reviewer_data.xlsx`) containing reviewer stats.

---

## Usage

### Install Dependencies
- Ensure you have **Python 3** and **pip** installed.
- Install any required libraries (if listed in a `requirements.txt` file or directly in the scripts).

### Setup `config.ini`
- Update `[github]` values (e.g., `token`, `owner`, `repo`).
- Adjust the `[filters]` for your desired date range and whether to include only closed/merged PRs.
- Confirm `[database]` points to your local database file (default: `pr_data.db`).

### Initialize/Update the Database
- Run `python store_prs2.py` to fetch and store PR data from GitHub into `pr_data.db`.

### Run Analytics Script
- Execute `python ml_pm2_spda_fav_fs_t15_rr.py` to get top 15 reviewers (manual script).
- Or rely on GitHub Actions if you want automatic recommendations (`post_recommendations.yml` + `recommendation.py`).

### Process Feedback
- Provide feedback via the method set up in `process_feedback.yml`.
- The workflow calls `process_feedback.py` to update reviewer scores in the `feedback` table.

### Clear/Reset Data (Optional)
- If you need to completely reset the database tables, run `python delete_tables_restart.py`.
- **Warning**: This action is irreversible and will remove **all** records from the relevant tables.

### Export Data to Excel (Optional)
- Run `python view_reviewer_data_excel.py` to generate an Excel file summarizing reviewer stats.

---

## Workflows

- **`.github/workflows/post_recommendations.yml`**  
  *Action Purpose:* Automatically post reviewer recommendations when PRs are opened or updated.

- **`.github/workflows/process_feedback.yml`**  
  *Action Purpose:* Listen for feedback events (e.g., comments) and update reviewer scores in the database.

- **`.github/workflows/new-prs-labeler.yml`**  
  *Action Purpose:* Automatically add labels to new PRs if they lack labels.

---

## Database Schema

1. **pull_requests**  
   - `pr_id` (PRIMARY KEY), `title`, `user_login`, `labels`, `created_at`, `updated_at`.

2. **reviews**  
   - `pr_id`, `reviewer`, `review_date`, `state`.

3. **pr_files**  
   - `pr_id`, `file_path`.

4. **feedback**  
   - `reviewer`, `fav_rev_point`.

---

## How to Use

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Saieie/PR-Reviewer-Recommendation.git
   cd PR Reviewer Recommendation
   ````
   
 ### Install Dependencies

- Make sure you have **Python 3** and **pip** installed.  
- Install any required libraries (if listed in a `requirements.txt` file or mentioned in the scripts).

---

### Set Up `config.ini`

- Update the `[github]` section with your GitHub token, repository owner, and repository name.  
- In the `[filters]` section, set the date range and specify whether to include only closed or merged PRs.  
- Ensure the `[database]` section points to the correct database file (default is `pr_data.db`).

---

### Initialize/Update the Database

Run the following command to fetch and store PR data (based on your `config.ini` settings) into `pr_data.db`:
```bash
python store_prs2.py
````
### Generate Analytics (Manual Script)

Run the following command to output the top 15 reviewers based on stored data and feedback metrics:
```bash
python ml_pm2_spda_fav_fs_t15_rr.py
````
### Process Feedback (If Needed)
Provide feedback via the mechanism set up in process_feedback.yml.
The workflow will call process_feedback.py to update reviewer scores in the feedback table.

### Reset the Database (Optional)
If you need to clear all existing data, run:
````bash
python delete_tables_restart.py
````
**Note**: This is irreversible and removes all records from the pr_files, reviews, pull_requests, and feedback tables.

### Export Data to Excel (Optional)
Generate an Excel file with reviewer stats:
```bash
python view_reviewer_data_excel.py
```` 
By following the steps above, you can set up your environment, initialize the database, fetch PR data, analyze reviewers, and optionally reset or export data.

---

## Contributing
Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request if you find any problems or want to add new features.

---

## License
NA


> **Note**: For security reasons, never commit real API tokens or other sensitive credentials to a public repository. Make sure to remove or invalidate any shared tokens (like `ghp_xxx`) before pushing your work to GitHub.


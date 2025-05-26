# PR Reviewer Recommendation System

This repository offers an automated approach to managing and analyzing GitHub pull requests to suggest the top 15 reviewers: it collects data (files, reviews, labels) in an SQLite database, assigns missing labels via file path patterns, ranks possible reviewers using historical data and feedback points, and enables users to update reviewer “favorite” points—thus streamlining PR management and encouraging quality reviews. The main purpose of this repository is to make PR handling more efficient and data-driven, reducing manual labeling effort and helping identify the best reviewers for each pull request.

This repository provides a set of scripts and GitHub Actions workflows to:
1. Store pull request data (PRs, files, and reviews) in a local SQLite database.
2. Automatically label new PRs if they are missing labels.
3. Generate reviewer recommendations based on multiple signals (e.g. file paths, tags, dynamic activity, and feedback).
4. Let users provide feedback to update reviewer “favorite” points.
5. View reviewer data in an Excel export.

Below you’ll find an overview of each file’s purpose, as well as how to set up and use this system.

---

## Table of Contents
- [Installation](#installation)
- [Configuration File](#configuration-file)
  - [config.ini](#configini)
- [Database Reset Script](#database-reset-script)
  - [delete_tables_restart.py](#delete_tables_restartpy)
- [Database File](#database-file)
  - [pr_data.db](#pr_datadb)
- [Scripts](#scripts)
  - [store_prs2.py](#store_prs2py)
  - [view_reviewer_data_excel.py](#view_reviewer_data_excelpy)
  - [ml_pm2_spda_fav_fs_t15_rr.py](#ml_pm2_spda_fav_fs_t15_rrpy)
  - [recommendation.py](#recommendationpy)
  - [process_feedback.py](#process_feedbackpy)
- [GitHub Workflows](#github-workflows)
  - [.github/workflows/post_recommendations.yml](#githubworkflowspost_recommendationsyml)
  - [.github/workflows/process_feedback.yml](#githubworkflowsprocess_feedbackyml)
  - [new-prs-labeler.yml](#new-prs-labeleryml)
    - [Alternate Explanation](#new-prs-labeleryml-alternate-explanation)
- [Usage Workflow](#usage-workflow)
  1. [Update the config.ini](#1-update-the-configini)
  2. [Initialize or Update the Database](#2-initialize-or-update-the-database)
  3. [Optional: Reset the Database](#3-optional-reset-the-database)
  4. [Run Recommendation or Other Scripts](#4-run-recommendation-or-other-scripts)
- [Feedback and Points](#feedback-and-points)
- [Excel Export](#excel-export)
- [Additional Notes](#additional-notes)
- [Demo](#Demo)

---

## Installation

Before running any scripts, ensure you have Python 3.7+ installed. We recommend creating a virtual environment:

```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment on Unix or macOS:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```
Then, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
An example `requirements.txt` might look like this:

```txt
PyGithub>=1.55
pandas>=1.3
openpyxl>=3.0
requests>=2.25
```
---

## Configuration File

### `config.ini`
This file stores all configuration details for the scripts:
- **[github]**:  
  - `token`: Your personal access token (PAT) for GitHub (do **not** commit it to a public repo).
  - `owner`: The GitHub organization or user that owns the repo.
  - `repo`: The repository name.

- **[filters]**:  
  - `start_date` and `end_date`: Used to limit which pull requests get retrieved.
  - `only_closed_prs`, `only_merged_prs`: If set to `true`, only those PRs will be fetched.
  - `required_labels`: If set, only PRs that have these labels are included.

- **[database]**:  
  - `file`: The path to the local SQLite database, e.g. `pr_data.db`.

**Example**:
```ini
[github]
token = ghp_xxxYOURTOKENHERExxx
owner = YourGitHubUsernameOrOrg
repo = YourRepositoryName

[filters]
start_date = 2025-01-01
end_date   = 2025-12-31
only_closed_prs = true
only_merged_prs = true
required_labels = 

[database]
file = pr_data.db
```
---

## Database Reset Script

### `delete_tables_restart.py`
- **Purpose:**  Resets the local database tables to empty by deleting all entries in:
  - `pr_files`
  - `pull_requests`
  - `feedback`	
- **Usage:**
   ```bash
   python delete_tables_restart.py
   ```
   After this, you can run the data-collection script again (e.g., store_prs2.py) to rebuild the database with fresh info.

---

## Database File

### `pr_data.db`
- **Purpose:** SQLite database storing all PR-related data. Includes:
  - **pull_requests:** (`pr_id`, `title`, `user_login`, `labels`, `created_at`, `updated_at`)
  - **pr_files:** (`pr_id`, `file_path`)
  - **reviews:** (`pr_id`, `reviewe`, `review_date`, `state`)
  - **feedback:** (`reviewer`, `fav_rev_points`) — tracks user feedback points.

  This file is generated and updated by the scripts. If it doesn’t exist, it will be created automatically.

---

## Scripts

### store_prs2.py
- **Purpose:** Reads `config.ini`, authenticates with GitHub, fetches PRs within the specified date range, and inserts them (along with their files and reviews) into `pr_data.db`.
- **Usage:**
  ```bash
  python store_prs2.py
  ```
  Ensures the database is populated with up-to-date pull request information.

---

### view_reviewer_data_excel.py
- **Purpose:** Queries the database for a specific reviewer’s data (PRs they reviewed, labels, file paths, etc.), then exports those results into an Excel file.
- **Usage:**
	1. (Optional) Edit the reviewer variable in the script, or pass arguments if you adapt it.
	2. Run:
	   ```bash
	   python view_reviewer_data_excel.py
	   ```
	3. It generates an Excel file like <reviewer>_data.xlsx.

---

### ml_pm2_spda_fav_fs_t15_rr.py
- **Purpose:** A script that queries `pr_data.db` and ranks the top 15 reviewers based on:
  - Matching file paths/tags,
  - “Favorite” reviewer points (feedback),
  - Reviewer dynamic activity,
  - Other weighting factors.

- **Usage**:
  1. Make sure the database is updated (via store_prs2.py).
  2. Run:
     ```bash
     python ml_pm2_spda_fav_fs_t15_rr.py
     ```
  3. It outputs the top 15 reviewers in your terminal.

---

### recommendation.py
- **Purpose:** Similar to `ml_pm2_spda_fav_fs_t15_rr.py` but designed to run within GitHub Actions. It:
  1. Fetches changed files and labels from GitHub for a specific PR,
  2. Ranks potential reviewers,
  3. (Optionally) posts a comment with those recommendations on the PR.
- **Usage**: Usually triggered by `.github/workflows/post_recommendations.yml`, but can be run locally if the environment variables (`GITHUB_TOKEN`, `GITHUB_OWNER`, and `GITHUB_REPO`) are set.

---

### process_feedback.py
- **Purpose:** Listens for user comments like `/feedback <reviewer>` on a pull request. When it sees such a comment:
  1. Increments the `<reviewer>`’s `fav_rev_points` in the `feedback` table of `pr_data.db`.
- **Usage**: Triggered by `.github/workflows/process_feedback.yml` whenever a PR comment matches the `/feedback` pattern.

---

## GitHub Workflows

### `.github/workflows/post_recommendations.yml`
- **Purpose:** Runs `recommendation.py` automatically whenever a pull request is opened, reopened, or updated (synchronized). It:
  1. Checks out the repo.
  2. Installs Python and necessary dependencies.
  3. Runs the recommendation script.
  4. Posts a comment on the PR with recommended reviewers.

---

### `.github/workflows/process_feedback.yml`
- **Purpose:** Waits for new PR comments. If it finds `/feedback <reviewer>`, it calls `process_feedback.py, which updates that reviewer’s `fav_rev_points` in the database.

---

### `new-prs-labeler.yml`
- **Purpose:** Provides YAML-based mappings from label names to file patterns. If a PR lacks labels, the scripts (or a workflow) can assign labels by matching changed file paths against these patterns.

#### `new-prs-labeler.yml (Alternate Explanation)`
- **Logic:**
  1. Check changed files in a PR,
  2. Match them against known wildcard patterns (e.g., clang/**, llvm/**),
  3. Assign the corresponding labels (like clang:frontend).

This helps to categorize PRs automatically.

---

## Usage Workflow
### 1. Update the `config.ini`
Adjust your token, owner, repo, and filter settings. Example:
```ini
[github]
token = ghp_xxxYOURTOKENHERExxx
owner = MyOrgOrUser
repo = MyRepoName

[filters]
start_date = 2025-01-01
end_date = 2025-12-31
only_closed_prs = true
only_merged_prs = true
required_labels = 

[database]
file = pr_data.db
```
### 2. Initialize or Update the Database
Run:
```bash
python store_prs2.py
```
This fetches the relevant PR data from GitHub and stores everything in `pr_data.db`.

### 3. Optional: Reset the Database
If you ever need to clear existing data:
```bash
python delete_tables_restart.py
```
Then re-run `store_prs2.py` to rebuild the database from scratch.

### 4. Run Recommendation or Other Scripts
- Local top 15:
  ```bash
  python ml_pm2_spda_fav_fs_t15_rr.py
  ```
  Displays the top 15 recommended reviewers in the terminal.

- Reviewer’s data to Excel:
  ```bash
  python view_reviewer_data_excel.py
  ```
  Exports an Excel spreadsheet for the specified reviewer.

---

## Feedback and Points
1. Comment on a pull request with:
   ```bash
   /feedback reviwer_names
   ```
2. The process_feedback.yml workflow invokes process_feedback.py, which increments someReviewer’s fav_rev_points in the database.
3. On subsequent recommendations, that reviewer now ranks higher.

---

## Excel Export
If you want to analyze or share a particular reviewer’s activity, run:
```bash 
python view_reviewer_data_excel.py
```
This queries the DB for the reviewer’s PRs, merges label/file data, and writes it out in an Excel file for easy review.

---

## Additional Notes
- Keep your GitHub token private.
- To update label → file path mappings, edit `new-prs-labeler.yml`.
- You can manually run `recommendation.py` if you supply environment variables for GitHub (e.g., `GITHUB_TOKEN`). Typically, though, it’s triggered by `.github/workflows/post_recommendations.yml`.

---
## Demo
https://github.com/user-attachments/assets/f90b9473-ffb4-4d7a-a313-174ed3a570fb
---

> **Note**: For security reasons, never commit real API tokens or other sensitive credentials to a public repository. Make sure to remove or invalidate any shared tokens (like `ghp_xxx`) before pushing your work to GitHub.

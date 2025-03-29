# PR Reviewer Recommendation System

This repository provides a set of scripts and GitHub Actions workflows to:
1. Store pull request data (PRs, files, and reviews) in a local SQLite database.
2. Automatically label new PRs if they are missing labels.
3. Generate reviewer recommendations based on multiple signals (e.g. file paths, tags, dynamic activity, and feedback).
4. Let users provide feedback to update reviewer “favorite” points.
5. View reviewer data in an Excel export.

Below you’ll find an overview of each file’s purpose, as well as how to set up and use this system.

---

## Table of Contents
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

## `pr_data.db`
-**Purpose:** SQLite database storing all PR-related data. Includes:
 -**pull_requests:** (`pr_id`, `title`, `user_login`, `labels`, `created_at`, `updated_at`)
 -**pr_files:** (`pr_id`, `file_path`)
 -**reviews:** (`pr_id`, `reviewe`, `review_date`, `state`)
 -**feedback:** (`reviewer`, `fav_rev_points`) — tracks user feedback points.

This file is generated and updated by the scripts. If it doesn’t exist, it will be created automatically.

---



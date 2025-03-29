# PR Reviewer Recommendation System

This repository implements a system that fetches pull request (PR) data from GitHub, stores it in a SQLite database, and then processes the data to recommend reviewers based on various criteria. The system also automates labeling of PRs and collects user feedback to adjust reviewer scores.

## File Descriptions

### `config.ini`
- **Purpose:** Stores configuration settings such as GitHub API credentials, repository details, and filter criteria.
- **Details:**  
  - **[github]**: Contains the GitHub token, repository owner, and repository name.  
  - **[filters]**: Defines constraints (e.g., start date, end date, and PR status filters) for fetching PRs.  
  - **[database]**: Specifies the SQLite database file (`pr_data.db`) where PR data is stored.

### `delete_tables_restart.py`
- **Purpose:** Resets the database by clearing data from key tables.
- **Details:** Deletes all rows from the tables `pr_files`, `reviews`, `pull_requests`, and `feedback`.

### `ml_pm2_spda_fav_fs_t15_rr.py`
- **Purpose:** Processes the PR data stored in `pr_data.db` to compute and display the top 15 reviewer recommendations.
- **Details:**  
  - Runs in a terminal environment.
  - Uses metrics like absolute matching score, dynamic activity score, and user feedback to rank reviewers.
  - **Note:** Make sure to update the database before running this script.

### `new-prs-labeler.yml`
- **Purpose:** Automates the assignment of labels to PRs that do not have labels assigned during creation.
- **Details:**  
  - Contains a YAML mapping of file path patterns to labels.
  - Helps ensure that PRs get appropriate labels based on the files they change.

### `.github/workflows/post_recommendations.yml` and `recommendation.py`
- **Purpose:** Automatically posts reviewer recommendations on PRs via GitHub Actions.
- **Details:**  
  - **post_recommendations.yml:** Triggers on PR events (opened, reopened, synchronized) and runs the recommendation process.  
  - **recommendation.py:** Contains the logic to calculate and post the reviewer recommendations (similar to the terminal script but without the feedback loop).

### `pr_data.db`
- **Purpose:** The SQLite database where all PR-related data is stored.
- **Details:**  
  - **Tables include:**  
    - `feedback`: Contains `reviewer` and `fav_rev_points`.  
    - `pr_files`: Stores `pr_id` and file paths.  
    - `pull_requests`: Stores `pr_id`, title, user login, labels, created_at, and updated_at.  
    - `reviews`: Stores `pr_id`, reviewer, review_date, and review state.

### `process_feedback.py` and `process_feedback.yml`
- **Purpose:** Collects and processes feedback from users to update reviewer points.
- **Details:**  
  - **process_feedback.py:** Parses GitHub comments (using a command like `/feedback reviewer_name`) and updates the feedback table in the database.  
  - **process_feedback.yml:** A GitHub Actions workflow that triggers on new issue comments to run the feedback processing script.

### `store_prs2.py`
- **Purpose:** Fetches PR data from GitHub using the settings in `config.ini` and stores the data in `pr_data.db`.
- **Details:**  
  - Creates required tables if they don’t exist.
  - Retrieves details such as PR files, commits, and reviews using GitHub API calls.

### `view_reviewer_data_excel.py`
- **Purpose:** Exports reviewer data into an Excel sheet.
- **Details:**  
  - Queries the database to retrieve the PRs, labels, and file paths associated with a given reviewer.
  - Generates an Excel report summarizing the reviewer’s activity.

## How to Use

1. **Configure:**  
   Update `config.ini` with your GitHub token, repository owner, repository name, and any specific filter settings.

2. **Store PR Data:**  
   Run `store_prs2.py` to fetch and save PR data from GitHub into `pr_data.db`.

3. **Reset Database (if needed):**  
   Use `delete_tables_restart.py` to clear data from the database tables.

4. **Generate Reviewer Recommendations:**  
   - **Terminal:** Run `ml_pm2_spda_fav_fs_t15_rr.py` to compute and display the top 15 reviewers.  
   - **GitHub Actions:** The workflow in `.github/workflows/post_recommendations.yml` automatically runs `recommendation.py` to post recommendations on PRs.

5. **Process Feedback:**  
   Provide feedback through GitHub comments (using the `/feedback reviewer_name` command). The workflow in `process_feedback.yml` runs `process_feedback.py` to update reviewer scores accordingly.

6. **Export Reviewer Data:**  
   Run `view_reviewer_data_excel.py` to generate an Excel report for a specific reviewer’s PR activity.

## GitHub Actions Workflows

- **Post Recommendations:** Automatically computes and posts reviewer recommendations on PR events.
- **Process Feedback:** Triggers on issue comments to update reviewer feedback in the database.

## Conclusion

This system streamlines the PR review process by automating data collection, labeling, reviewer recommendation, and feedback management. Customize the configuration and extend the scripts as needed to fit your development workflow.

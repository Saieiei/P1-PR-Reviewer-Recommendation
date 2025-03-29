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
###2. Initialize or Update the Database
Run:
```bash
python store_prs2.py
```
This fetches the relevant PR data from GitHub and stores everything in pr_data.db.

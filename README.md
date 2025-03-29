# Reviewer Suggestion System

This repository contains a set of scripts and GitHub Actions workflows designed to **automatically recommend reviewers** for Pull Requests (PRs). The system uses metadata such as files changed, labels, past reviewer contributions, and reviewer activity to produce a **ranked list** of potential reviewers.

---

## Table of Contents
1. [Overview](#overview)  
2. [Project Files](#project-files)  
   - [config.ini](#configini)  
   - [delete_tables_restart.py](#delete_tables_restartpy)  
   - [ml_pm2_spda_fav_fs_t15_rr.py](#ml_pm2_spda_fav_fs_t15_rrpy)  
   - [new-prs-labeler.yml](#new-prs-labeleryml)  
   - [post_recommendations.yml](#post_recommendationsyml)  
   - [recommendation.py](#recommendationpy)  
   - [pr_data.db](#pr_datadb)  
   - [process_feedback.py](#process_feedbackpy)  
   - [process_feedback.yml](#process_feedbackyml)  
   - [store_prs2.py](#store_prs2py)  
   - [view_reviewer_data_excel.py](#view_reviewer_data_excelpy)  
3. [Usage Flow](#usage-flow)  
4. [Feedback and Scoring](#feedback-and-scoring)  
5. [GitHub Workflows](#github-workflows)  
6. [Installation and Setup](#installation-and-setup)  

---

## Overview

### What It Does
- **Automatically fetches** PR metadata (labels, files, reviewers) from GitHub.
- **Scores** reviewers based on:
  - Past file/path matches
  - Labels/tags similarity
  - Reviewer’s PageRank (co-reviewer graph)
  - Recent activity (exponential decay)
  - “Favorite reviewer” feedback
- **Suggests top reviewers** in a GitHub PR comment or via a terminal script.

### Key Components
1. **Data Gathering** – Using `store_prs2.py` to populate the SQLite database (`pr_data.db`) with PR details.  
2. **Recommendation Scripts** – 
   - Local/Interactive: `ml_pm2_spda_fav_fs_t15_rr.py`  
   - GitHub Workflow: `recommendation.py` + `post_recommendations.yml`  
3. **Feedback Mechanism** – `process_feedback.py` & `process_feedback.yml` handle `/feedback <reviewer>` comments to boost a reviewer’s “favorite score”.

---

## Project Files

### 1. config.ini
Holds configuration details:
- **github** section: GitHub `token`, `owner`, `repo`  
- **filters** section: date range, label constraints, whether to fetch only closed/merged PRs  
- **database** section: path to the SQLite DB file (e.g., `pr_data.db`)

### 2. delete_tables_restart.py
Clears all rows in `pr_files`, `reviews`, `pull_requests`, and `feedback` tables. Use cautiously to **reset** the database.

```bash
python delete_tables_restart.py

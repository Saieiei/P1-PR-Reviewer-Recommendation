import os
import requests

def fetch_pull_requests(owner="Saieiei", repo="PR-Reviewer-Recommendation"):
    # We’ll use your personal access token from an env variable
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("No GitHub token found. Please set GITHUB_TOKEN environment variable.")

    # GitHub API endpoint to list PRs
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all"
    
    # Add headers to authenticate
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # We’ll fetch the first page, but you can handle pagination as needed
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} {response.text}")
    
    prs = response.json()
    return prs

if __name__ == "__main__":
    # Example usage
    owner = "Saieiei"
    repo = "PR-Reviewer-Recommendation"
    pull_requests = fetch_pull_requests()
    for pr in pull_requests:
        print(f"PR #{pr['number']}: {pr['title']}")

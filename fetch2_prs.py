import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_all_pull_requests(token, owner, repo, state="all", verify_ssl=True):
    pull_requests = []
    page = 1
    per_page = 100

    session = requests.Session()
    # Add the Accept header to ensure we get GitHub’s standard JSON format
    session.headers.update({
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    })

    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)

    session.verify = verify_ssl

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": per_page, "page": page}

        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Print out the response text from GitHub to see the actual error details.
            print(f"HTTP error: {e}")
            print(f"Response status code: {response.status_code}")
            print(f"Response text: {response.text}")
            break
        except requests.exceptions.SSLError as e:
            print(f"SSL error encountered: {e}")
            print("If you are in a trusted network, try setting verify_ssl=False.")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request error encountered: {e}")
            break

        pr_page = response.json()
        if not pr_page:
            # No more PRs returned, so we’re done.
            break

        pull_requests.extend(pr_page)
        page += 1

    return pull_requests

def main():
    github_token = ""  # Make sure this is correct and properly scoped!
    owner = "llvm"
    repo = "llvm-project"

    prs = get_all_pull_requests(github_token, owner, repo, state="all", verify_ssl=True)
    if prs:
        for pr in prs:
            print(f"PR #{pr['number']}: {pr['title']} [{pr['state']}]")
    else:
        print("No pull requests fetched.")

if __name__ == "__main__":
    main()

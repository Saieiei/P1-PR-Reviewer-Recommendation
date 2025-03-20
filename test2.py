import requests

url = "https://api.github.com/search/issues?q=repo:llvm/llvm-project+is:pr+created:2020-01-01..2020-01-20+state:all"
headers = {
    "Authorization": "Bearer ghp_",
    "Accept": "application/vnd.github.v3+json"
}
response = requests.get(url, headers=headers, verify=False)  # <-- disable SSL check
print("Status:", response.status_code)
print("Body:", response.text)

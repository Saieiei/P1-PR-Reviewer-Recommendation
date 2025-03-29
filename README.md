
### `new-prs-labeler.yml`
- **Purpose:** Provides YAML-based mappings from label names to file patterns. If a PR lacks labels, the scripts (or a workflow) can assign labels by matching changed file paths against these patterns.

`new-prs-labeler.yml (Alternate Explanation)`
- **Logic:**
  1. Check changed files in a PR,
  2. Match them against known wildcard patterns (e.g., clang/**, llvm/**),
  3. Assign the corresponding labels (like clang:frontend).

This helps to categorize PRs automatically.

---

#!/bin/bash

# random_commits.sh
# Makes 5-10 random commits to the current Git repo and pushes them.

set -e

# ── Config ────────────────────────────────────────────────────────────────────
COMMIT_FILE="activity.log"          # file that gets updated each commit
MIN_COMMITS=5
MAX_COMMITS=10
# ─────────────────────────────────────────────────────────────────────────────

# Check we're inside a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "❌  Not inside a Git repository. Please run this from your project folder."
  exit 1
fi

# Random number of commits between MIN and MAX (inclusive)
NUM_COMMITS=$(( RANDOM % (MAX_COMMITS - MIN_COMMITS + 1) + MIN_COMMITS ))
echo "🎲  Making $NUM_COMMITS commits..."

# Sample commit message components
VERBS=("Add" "Update" "Refactor" "Fix" "Improve" "Clean up" "Optimize" "Revise" "Tweak" "Polish")
NOUNS=("config" "docs" "README" "styles" "tests" "utils" "helpers" "setup" "workflow" "comments")
ADJECTIVES=("minor" "small" "routine" "quick" "general" "miscellaneous" "various" "incremental")

random_element() {
  local arr=("$@")
  echo "${arr[$((RANDOM % ${#arr[@]}))]}"
}

for i in $(seq 1 "$NUM_COMMITS"); do
  VERB=$(random_element "${VERBS[@]}")
  NOUN=$(random_element "${NOUNS[@]}")
  ADJ=$(random_element "${ADJECTIVES[@]}")
  MSG="$VERB $ADJ $NOUN"

  # Write something to the tracked file so there's an actual change
  echo "[$( date '+%Y-%m-%d %H:%M:%S')] commit $i/$NUM_COMMITS — $MSG" >> "$COMMIT_FILE"

  git add "$COMMIT_FILE"
  git commit -m "$MSG"
  echo "  ✅  ($i/$NUM_COMMITS) $MSG"
done

echo ""
echo "🚀  Pushing to remote..."
git push

echo ""
echo "✨  Done! $NUM_COMMITS commits pushed."
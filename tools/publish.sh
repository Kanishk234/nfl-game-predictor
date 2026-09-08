#!/usr/bin/env bash
# Commit the data this job produced and push it, retrying if another job lands first.
#
# Only data is ever committed. The site is build output: it is gitignored and rebuilt by the
# deploy job from whatever is on main. That is deliberate — committing 24 generated HTML files
# that change every run meant any concurrent job produced a rebase conflict in files nobody
# merges, and the job died *after* making the prediction but before pushing it. On an ephemeral
# runner that prediction is gone, and a prediction cannot be back-dated, so the week would have
# a permanent hole.
#
# Data files are new files with unique names, so a rebase between jobs is conflict-free.

set -euo pipefail

DATA_PATHS="$1"
BRANCH="${2:-main}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# Stage only the paths that exist. `git add` on a missing directory is fatal under `set -e`,
# and aborting here would throw away a prediction that has already been made.
for path in ${DATA_PATHS}; do
  [ -e "${path}" ] && git add "${path}" || echo "no ${path} to stage"
done
if git diff --cached --quiet; then
  # Nothing new to stage. That is the normal no-op case — but only if we are also in step with
  # the remote. A commit made by an earlier attempt that failed to push must still go out, not
  # be reported as "nothing to do" and abandoned.
  git fetch -q origin "${BRANCH}"
  if [ "$(git rev-list --count "origin/${BRANCH}..HEAD")" -eq 0 ]; then
    echo "nothing new to publish"
    exit 0
  fi
  echo "nothing new to stage, but $(git rev-list --count "origin/${BRANCH}..HEAD") commit(s) still unpushed"
fi

new_prediction=$(git diff --cached --name-only -- data/predictions 2>/dev/null | head -1)
if [ -n "${new_prediction}" ]; then
  message="publish $(basename "${new_prediction}" .json) prediction"
else
  message="grade $(date -u +%Y-%m-%d) results"
fi
git commit -m "${message}"
echo "committed: ${message}"

for attempt in 1 2 3; do
  # --autostash so an unrelated dirty file in the working tree can never block the rebase and
  # strand a prediction that has already been committed locally.
  git pull --rebase --autostash origin "${BRANCH}"
  if git push origin "HEAD:${BRANCH}"; then
    echo "pushed on attempt ${attempt}"
    exit 0
  fi
  echo "push rejected (something landed first); retrying [${attempt} of 3]"
  sleep $((attempt * 5))
done

echo "could not push after 3 attempts" >&2
exit 1

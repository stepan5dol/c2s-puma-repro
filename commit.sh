#!/bin/bash
# Commit the move of the analysis image build to GitHub and push. Run in a normal terminal from the repository folder:
#   bash commit.sh
# Pushing needs a working GitHub login (gh auth refresh -h github.com, or an ssh key).
set -euo pipefail
cd "$(dirname "$0")"
git add -A -- apptainer/c2s-analysis.def .github/workflows grid/analyze.slurm grid/submit_cells.sh grid/start.sh \
    grid/build_analysis_image.sh grid/README.md grid/lib/prepare_root.py
git diff --cached --quiet && { echo "nothing to commit"; } || \
git -c user.name=stepan5dol -c user.email=stepan5dol@users.noreply.github.com commit -q -F - <<MSG
Build the analysis image on GitHub; start the grid with one command

The image holds only the Python environment and is published as a release asset tagged with the hash of
its recipe; the analysis job downloads it and links the data from data_src. grid/start.sh queues the
starting cell and stage A.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
git log -1 --oneline
git push origin main
git status -sb | head -1

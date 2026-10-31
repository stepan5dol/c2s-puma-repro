#!/bin/bash
# Put the repository and the data on xdisk in one rsync (one password + Duo), and mark the commit in local git.
# Run in a normal terminal from this folder:  bash deploy.sh
# Result on Puma:  /xdisk/vkalin/dolzhenko/c2s/repo/        the working tree (no .git), incl. grid/tree with its small data files
#                  /xdisk/vkalin/dolzhenko/c2s/data_src/    the 6 data files of the analysis image (~7 GB)
#                  /xdisk/vkalin/dolzhenko/c2s/repo/REVISION  commit hash, tag and date of this deploy
set -euo pipefail
cd "$(dirname "$0")"
A=$HOME/Documents/AzithroGemma
HOST=${HOST:-dolzhenko@filexfer.hpc.arizona.edu}
DEST=/xdisk/vkalin/dolzhenko/c2s

git diff --quiet HEAD || { echo "tracked files differ from HEAD: commit them first (bash commit.sh)" >&2; exit 1; }
for f in "$A/BPD-PH/GSE275938_compiled_counts.csv" "$A/BPD-PH/GSE275938_cell_metadata.csv" "$A/rat.ho.azi.integrated.h5ad" \
         "$A/he_lung_atlas.h5ad" "$A/2022FetalLungIntCounts.h5ad" "$A/retrain_v2/scripts/cell_types_config.json"; do
    [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done

TAG=deployed-$(date +%Y%m%d-%H%M)
git -c user.name=stepan5dol -c user.email=stepan5dol@users.noreply.github.com tag -a "$TAG" -m "copied to $DEST on $(date '+%F %T')"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"; rm -f REVISION' EXIT   # REVISION is only a courier file for rsync
ln -s "$PWD" "$STAGE/repo"
mkdir "$STAGE/data_src"
for f in "$A/BPD-PH/GSE275938_compiled_counts.csv" "$A/BPD-PH/GSE275938_cell_metadata.csv" "$A/rat.ho.azi.integrated.h5ad" \
         "$A/he_lung_atlas.h5ad" "$A/2022FetalLungIntCounts.h5ad" "$A/retrain_v2/scripts/cell_types_config.json"; do
    ln -s "$f" "$STAGE/data_src/$(basename "$f")"
done
printf 'commit %s\ntag %s\ndate %s\n' "$(git rev-parse HEAD)" "$TAG" "$(date '+%F %T')" > REVISION
rsync -aL --partial --info=progress2 --exclude=.git --exclude=__pycache__ --exclude=commit.sh --exclude=deploy.sh \
    "$STAGE/" "$HOST:$DEST/"
echo "done: tag $TAG, $(git rev-parse --short HEAD) is on $HOST:$DEST/repo"

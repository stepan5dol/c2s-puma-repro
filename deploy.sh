#!/bin/bash
# Put the repository and the data on xdisk in one rsync (one password + Duo), then mark the commit with a tag in local git.
# (macOS ships openrsync: no --info=progress2, so --progress is used. -z: the counts csv shrinks ~20x, rat h5ad ~8x, the
#  human atlases ~2.4x; ssh keepalives because filexfer closed an idle connection on 2026-10-30.)
# Run in a normal terminal from this folder:  bash deploy.sh
# Result on Puma:  /xdisk/vkalin/dolzhenko/c2s/repo/        the working tree (no .git), incl. grid/tree with its small data files
#                  /xdisk/vkalin/dolzhenko/c2s/data_src/    the 6 data files of the analysis image (~7 GB)
#                  /xdisk/vkalin/dolzhenko/c2s/repo/REVISION  commit hash, tag and date of this deploy
set -euo pipefail
cd "$(dirname "$0")"
A=$HOME/Documents/AzithroGemma
HOST=${HOST:-dolzhenko@filexfer.hpc.arizona.edu}
DEST=/xdisk/vkalin/dolzhenko/c2s

P=$A/retrain_prompt/pipeline
git diff --quiet HEAD || { echo "tracked files differ from HEAD: commit them first (bash commit.sh)" >&2; exit 1; }
for f in "$A/BPD-PH/GSE275938_compiled_counts.csv" "$A/BPD-PH/GSE275938_cell_metadata.csv" "$A/rat.ho.azi.integrated.h5ad" \
         "$A/he_lung_atlas.h5ad" "$A/2022FetalLungIntCounts.h5ad" "$A/retrain_v2/scripts/cell_types_config.json" \
         "$P/inputs_sha256.txt" "$P/sft_pythia1b.yaml" "$P/sft_gemma2_2b.yaml" "$P/sft_example_registry.csv" \
         "$P/sft_dataset_bidir/train_segments.jsonl" "$P/sft_dataset_bidir/valid_segments.jsonl" "$P/sft_dataset_bidir/test_segments.jsonl" \
         "$P/inference_azi.jsonl" "$P/inference_azi_concise.jsonl" "$P/inference_azi_ratmatch.jsonl"; do
    [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done

TAG=deployed-$(date +%Y%m%d-%H%M)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"; rm -f REVISION' EXIT   # REVISION is only a courier file for rsync
ln -s "$PWD" "$STAGE/repo"
mkdir -p "$STAGE/data_src/dataset" "$STAGE/data_src/inference"
for f in "$A/BPD-PH/GSE275938_compiled_counts.csv" "$A/BPD-PH/GSE275938_cell_metadata.csv" "$A/rat.ho.azi.integrated.h5ad" \
         "$A/he_lung_atlas.h5ad" "$A/2022FetalLungIntCounts.h5ad" "$A/retrain_v2/scripts/cell_types_config.json" \
         "$P/inputs_sha256.txt" "$P/sft_pythia1b.yaml" "$P/sft_gemma2_2b.yaml"; do
    ln -s "$f" "$STAGE/data_src/$(basename "$f")"
done
for f in "$P/sft_dataset_bidir/train_segments.jsonl" "$P/sft_dataset_bidir/valid_segments.jsonl" \
         "$P/sft_dataset_bidir/test_segments.jsonl" "$P/sft_example_registry.csv"; do
    ln -s "$f" "$STAGE/data_src/dataset/$(basename "$f")"
done
for f in "$P/inference_azi.jsonl" "$P/inference_azi_concise.jsonl" "$P/inference_azi_ratmatch.jsonl"; do
    ln -s "$f" "$STAGE/data_src/inference/$(basename "$f")"
done
printf 'commit %s\ntag %s\ndate %s\n' "$(git rev-parse HEAD)" "$TAG" "$(date '+%F %T')" > REVISION
rsync -aLz --partial --progress --timeout=300 -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6' --exclude=.git --exclude=__pycache__ --exclude=commit.sh --exclude=deploy.sh \
    "$STAGE/" "$HOST:$DEST/"
# the tag is set only after the copy succeeded (set -e stops the script before this line if rsync fails)
git -c user.name=stepan5dol -c user.email=stepan5dol@users.noreply.github.com tag -a "$TAG" -m "copied to $DEST on $(date '+%F %T')"
echo "done: tag $TAG, $(git rev-parse --short HEAD) is on $HOST:$DEST/repo"

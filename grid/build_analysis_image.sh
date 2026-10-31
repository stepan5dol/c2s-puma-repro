#!/bin/bash
# Build c2s-analysis.sif on a compute node. Run it yourself in an interactive session, not on the login node:
#   interactive -a vkalin -t 04:00:00 -c 8     (then)     bash build_analysis_image.sh
# Data come from DATA_SRC (default: the folder you scp'd the 7 GB to). Nothing is uploaded anywhere.
set -euo pipefail
DATA=/xdisk/vkalin/dolzhenko/c2s
DATA_SRC=${DATA_SRC:-$DATA/data_src}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$DATA/images" "$DATA/build_tmp" "$DATA/build_cache"
for f in GSE275938_compiled_counts.csv GSE275938_cell_metadata.csv rat.ho.azi.integrated.h5ad he_lung_atlas.h5ad 2022FetalLungIntCounts.h5ad cell_types_config.json; do
    [ -s "$DATA_SRC/$f" ] || { echo "missing $DATA_SRC/$f" >&2; exit 1; }
done
W=$DATA/build_tmp/analysis_$$
mkdir -p "$W/data" && cp "$HERE/apptainer/c2s-analysis.def" "$W/"
for f in "$DATA_SRC"/*; do ln -s "$f" "$W/data/$(basename "$f")"; done
export APPTAINER_TMPDIR=$DATA/build_tmp APPTAINER_CACHEDIR=$DATA/build_cache
( cd "$W" && apptainer build "$DATA/images/c2s-analysis.sif.part" c2s-analysis.def )
mv "$DATA/images/c2s-analysis.sif.part" "$DATA/images/c2s-analysis.sif"
( cd "$DATA/images" && sha256sum c2s-analysis.sif | tee c2s-analysis.sif.sha256 )
apptainer exec "$DATA/images/c2s-analysis.sif" cat /opt/analysis_freeze.txt > "$DATA/images/c2s-analysis.freeze.txt"
rm -rf "$W"
echo "built $DATA/images/c2s-analysis.sif"

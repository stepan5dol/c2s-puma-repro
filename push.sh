#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

HOST="dolzhenko@filexfer.hpc.arizona.edu"
DEST="/xdisk/vkalin/dolzhenko/c2s"
A="/Users/stepandolzhenko/Documents/AzithroGemma"
P="$A/retrain_prompt/pipeline"

echo "== 1. Подготовка конфигов и sha256..."
mkdir -p configs
cp "$A/retrain_v2/hf_runs/pythia_ckpts_meta/sft_pythia1b.yaml" configs/sft_pythia1b.yaml 2>/dev/null || true

# Сгенерировать inputs_sha256.txt локально
python3 -c '
import hashlib
from pathlib import Path
P = Path("'"$P"'")
cfg = Path("configs/sft_pythia1b.yaml")
files = [
    ("sft_pythia1b.yaml", cfg),
    ("dataset/train_segments.jsonl", P / "sft_dataset_bidir/train_segments.jsonl"),
    ("dataset/valid_segments.jsonl", P / "sft_dataset_bidir/valid_segments.jsonl"),
    ("dataset/test_segments.jsonl", P / "sft_dataset_bidir/test_segments.jsonl"),
    ("inference/inference_azi.jsonl", P / "inference_azi.jsonl"),
    ("inference/inference_azi_concise.jsonl", P / "inference_azi_concise.jsonl"),
    ("inference/inference_azi_ratmatch.jsonl", P / "inference_azi_ratmatch.jsonl"),
]
lines = []
for rel, path in files:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{h}  {rel}")
Path("configs/inputs_sha256.txt").write_text("\n".join(lines) + "\n")
'

# Скрипт запуска на Пуме в 1 команду
cat > go.sh <<'GOSH'
#!/bin/bash
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
REPO=$D/repo
if [ -e "$D/grid/cells.tsv" ]; then
    echo "Архивируем старый grid..."
    mv "$D/grid" "$D/grid_old_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$D/data_src"
cp "$REPO/configs/sft_pythia1b.yaml" "$D/data_src/"
cp "$REPO/configs/inputs_sha256.txt" "$D/data_src/"
echo "Запуск обучения 3 ячеек Stage A..."
bash "$REPO/grid/submit_cells.sh" 1e-4:2:42 1e-5:2:42 3e-5:2:42
GOSH
chmod +x go.sh

echo "== 2. Отправка репозитория на xdisk..."
rsync -avzP --exclude='.git' --exclude='lab' --exclude='__pycache__' --exclude='.DS_Store' --exclude='.claude' \
  ./ "$HOST:$DEST/repo/"

echo "== 3. Отправка файлов датасета в data_src..."
rsync -avzP \
  "$P/sft_dataset_bidir/train_segments.jsonl" \
  "$P/sft_dataset_bidir/valid_segments.jsonl" \
  "$P/sft_dataset_bidir/test_segments.jsonl" \
  "$P/sft_example_registry.csv" \
  "$HOST:$DEST/data_src/dataset/"

echo "== 4. Отправка inference_azi*.jsonl..."
rsync -avzP \
  "$P/inference_azi.jsonl" \
  "$P/inference_azi_concise.jsonl" \
  "$P/inference_azi_ratmatch.jsonl" \
  "$HOST:$DEST/data_src/inference/"

rsync -avzP configs/sft_pythia1b.yaml configs/inputs_sha256.txt "$HOST:$DEST/data_src/"

echo ""
echo "=========================================================="
echo "ВСЕ ФАЙЛЫ УСПЕШНО ЗАГРУЖЕНЫ!"
echo "Теперь на Пуме выполните всего одну команду:"
echo "bash /xdisk/vkalin/dolzhenko/c2s/repo/go.sh"
echo "=========================================================="

#!/usr/bin/env bash
set -euo pipefail

# 统一入口：先训练，再根据 metrics 自动绘图
# 用法：
#   bash scripts/run.sh                          # 使用默认环境变量（LOG_INTERVAL=100）
#   GPU_ID=1 MAX_EPOCHS=50 bash scripts/run.sh   # 覆盖部分参数
# 说明：
# - 若找不到 Python 模块 src.train，则跳过训练，仅用现有 metrics 绘图。
# - 绘图优先使用 metrics-final-<tag>.json，找不到则回退到 metrics-latest.json。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="results/train-${STAMP}.log"

echo "[run] cwd=$ROOT_DIR"
echo "[run] GPU_ID=$GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# 1) 检测训练入口是否存在
if "$PYTHON_BIN" - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('src.train') else 1)
PY
then
  echo "[run] Found training module: src.train"
  echo "[run] Starting training..."
  # 将环境变量透传给训练（例如 MAX_EPOCHS/LOG_INTERVAL 等），日志保存到 results/
  set -x
  # 终端显示 stdout+stderr（含tqdm），日志仅保存 stdout（步级与摘要），避免tqdm刷屏进日志
  "$PYTHON_BIN" -u -m src.train 2> >(cat >&2) | tee "$LOG_PATH"
  set +x
else
  echo "[run] WARNING: src.train module not found. Skipping training and using existing metrics."
fi

# 2) 选择最新 metrics 文件（有 final 用 final，否则用 latest 稳定路径）
LATEST_FINAL=$(ls -t results/metrics-final-*.json 2>/dev/null | head -n 1 || true)
if [[ -n "$LATEST_FINAL" ]]; then
  METRICS_FILE="$LATEST_FINAL"
else
  METRICS_FILE="results/metrics-latest.json"
fi

if [[ ! -f "$METRICS_FILE" ]]; then
  echo "[run] ERROR: metrics file not found: $METRICS_FILE"
  echo "[run] Hint: ensure training wrote metrics to results/*.json"
  exit 1
fi

echo "[run] Plotting with metrics: $METRICS_FILE"
METRICS_PATH="$METRICS_FILE" "$PYTHON_BIN" -u -m src.plot_metrics

echo "[run] Done. Logs: $LOG_PATH (if training ran)."
import json, os
import matplotlib.pyplot as plt

def load_metrics(path_candidates):
    for p in path_candidates:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f), p
    raise FileNotFoundError(f"No metrics file found in: {path_candidates}")

def main():
    # 优先读取稳定路径，其次读取最近 run_tag 的 metrics
    candidates = [
        os.path.join('results', 'metrics-latest.json')
    ]
    # 允许用户传入备用文件路径（环境变量）
    alt = os.environ.get('METRICS_PATH')
    if alt:
        candidates.insert(0, alt)
    metrics, used = load_metrics(candidates)
    print(f"[plot] using metrics: {used}")

    losses = metrics.get('epoch_losses', [])
    lrs = metrics.get('epoch_lrs', [])

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(range(1, len(losses)+1), losses, marker='o')
    axs[0].set_ylabel('Avg Loss per Epoch')
    axs[0].grid(True, ls='--', alpha=0.4)

    axs[1].plot(range(1, len(lrs)+1), lrs, marker='o', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('LR')
    axs[1].grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join('results', 'training_curves.png')
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved figure to {out_path}")

if __name__ == '__main__':
    main()
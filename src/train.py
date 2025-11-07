import os
import json
import time
import sys
from datetime import datetime
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .data import IWSLT2017Seq2Seq
from .model import Seq2SeqTransformer
from .utils import build_warmup_cosine_scheduler, count_parameters


def get_env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def get_env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def get_env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "y")


def collate_fn_builder(pad_id):
    def collate(batch):
        src_list, tgt_in_list, tgt_out_list = zip(*batch)
        src_pad = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
        tgt_in_pad = pad_sequence(tgt_in_list, batch_first=True, padding_value=pad_id)
        tgt_out_pad = pad_sequence(tgt_out_list, batch_first=True, padding_value=pad_id)
        return src_pad, tgt_in_pad, tgt_out_pad
    return collate


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    pad_id,
    grad_clip: float = 0.0,
    limit_batches: int | None = None,
    log_interval: int = 0,
    epoch_idx: int = 1,
):
    model.train()
    total_loss = 0.0
    steps = 0
    total_iters = limit_batches if limit_batches is not None else len(loader)
    iterator = enumerate(loader)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=total_iters,
            desc=f"[train] epoch {epoch_idx}",
            ncols=0,
            file=sys.stderr,
            dynamic_ncols=True,
            leave=False,
        )
    for i, batch in iterator:
        src, tgt_in, tgt_out = batch
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt_in, pad_id=pad_id)  # [B, T, V]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        batch_loss = loss.item()
        total_loss += batch_loss
        steps += 1

        # 步级日志输出与tqdm后缀
        lr_now = optimizer.param_groups[0]["lr"]
        if tqdm is not None:
            try:
                # 每步更新进度条后缀，避免过度刷新
                if log_interval and steps % log_interval == 0:
                    iterator.set_postfix({"loss": f"{batch_loss:.4f}", "lr": f"{lr_now:.6f}"})
            except Exception:
                pass
        # 常规打印（每log_interval步一次）
        if log_interval and steps % log_interval == 0:
            msg = f"[train] epoch {epoch_idx} step {steps}/{total_iters} | loss={batch_loss:.4f} | lr={lr_now:.6f}"
            if tqdm is not None:
                try:
                    tqdm.write(msg, file=sys.stdout)
                except Exception:
                    print(msg, flush=True)
            else:
                print(msg, flush=True)

        if limit_batches is not None and steps >= limit_batches:
            break
    avg = total_loss / max(1, steps)
    return avg, steps


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, pad_id, limit_batches=None):
    model.eval()
    total_loss = 0.0
    steps = 0
    for i, (src, tgt_in, tgt_out) in enumerate(loader):
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        logits = model(src, tgt_in, pad_id=pad_id)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item()
        steps += 1
        if limit_batches is not None and steps >= limit_batches:
            break
    avg = total_loss / max(1, steps)
    return avg, steps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    # 数据与基本配置
    src_lang = os.environ.get("SRC_LANG", "en")
    tgt_lang = os.environ.get("TGT_LANG", "de")
    seq_len = get_env_int("SEQ_LEN", 128)
    batch_size = get_env_int("BATCH_SIZE", 32)
    num_workers = get_env_int("NUM_WORKERS", 2)

    # 模型结构（消融相关）
    d_model = get_env_int("D_MODEL", 128)
    d_ff = get_env_int("D_FF", 512)
    num_heads = get_env_int("NUM_HEADS", 4)
    num_layers = get_env_int("NUM_LAYERS", 2)
    enc_layers = os.environ.get("ENC_LAYERS")
    dec_layers = os.environ.get("DEC_LAYERS")
    enc_layers = int(enc_layers) if enc_layers is not None else None
    dec_layers = int(dec_layers) if dec_layers is not None else None
    use_cross_attn = get_env_bool("USE_CROSS_ATTN", True)

    # 训练超参
    max_epochs = get_env_int("MAX_EPOCHS", 5)
    lr = get_env_float("LR", 3e-4)
    weight_decay = get_env_float("WEIGHT_DECAY", 1e-4)
    warmup_ratio = get_env_float("WARMUP_RATIO", 0.1)
    grad_clip = get_env_float("GRAD_CLIP", 1.0)
    limit_train_batches = get_env_int("LIMIT_TRAIN_BATCHES", 0) or None
    limit_valid_batches = get_env_int("LIMIT_VALID_BATCHES", 0) or None
    log_interval = get_env_int("LOG_INTERVAL", 100)
    ckpt_interval = get_env_int("CKPT_INTERVAL", 10)

    # 数据集
    train_ds = IWSLT2017Seq2Seq(split="train", seq_len=seq_len, src_lang=src_lang, tgt_lang=tgt_lang)
    valid_ds = IWSLT2017Seq2Seq(split="validation", seq_len=seq_len, src_lang=src_lang, tgt_lang=tgt_lang)
    pad_id = train_ds.pad_id

    collate = collate_fn_builder(pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    # 模型
    model = Seq2SeqTransformer(
        src_vocab=train_ds.vocab_size,
        tgt_vocab=train_ds.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        enc_num_layers=enc_layers,
        dec_num_layers=dec_layers,
        use_cross_attn=use_cross_attn,
    ).to(device)

    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max_epochs * (limit_train_batches if limit_train_batches is not None else len(train_loader))
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 结果目录与 run tag
    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在 run_tag 中嵌入消融参数以便文件命名与后续绘图识别
    enc_layers_eff = enc_layers if enc_layers is not None else num_layers
    dec_layers_eff = dec_layers if dec_layers is not None else num_layers
    cfg_tag = f"d{d_model}_ff{d_ff}_h{num_heads}_enc{enc_layers_eff}_dec{dec_layers_eff}_cross{1 if use_cross_attn else 0}"
    run_tag = f"{train_ds.dataset_name}_{src_lang}-{tgt_lang}_{cfg_tag}_{stamp}"
    print(f"[train] run_tag={run_tag}")

    metrics = {"epoch_losses": [], "epoch_lrs": [], "val_epoch_losses": []}

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_steps = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            pad_id,
            grad_clip=grad_clip,
            limit_batches=limit_train_batches,
            log_interval=log_interval,
            epoch_idx=epoch,
        )
        val_loss, _ = evaluate(model, valid_loader, loss_fn, device, pad_id, limit_batches=limit_valid_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        metrics["epoch_losses"].append(train_loss)
        metrics["epoch_lrs"].append(lr_now)
        metrics["val_epoch_losses"].append(val_loss)
        # 写入最新指标（稳定路径）
        # 额外记录配置摘要，便于后续图表标题展示
        latest_payload = {**metrics, "config": {
            "d_model": d_model, "d_ff": d_ff, "num_heads": num_heads,
            "enc_layers": enc_layers_eff, "dec_layers": dec_layers_eff,
            "use_cross_attn": use_cross_attn
        }}
        with open(os.path.join("results", "metrics-latest.json"), "w", encoding="utf-8") as f:
            json.dump(latest_payload, f, ensure_ascii=False, indent=2)
        print(
            f"[train] epoch {epoch}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.6f} | steps={train_steps} | {time.time()-t0:.1f}s",
            flush=True,
        )

        # 每若干轮保存checkpoint（包含模型/优化器/调度器与配置、当前指标）
        if ckpt_interval and epoch % ckpt_interval == 0:
            os.makedirs(os.path.join("results", "checkpoints"), exist_ok=True)
            ckpt_path = os.path.join("results", "checkpoints", f"epoch-{epoch:04d}-{run_tag}.pt")
            try:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "metrics": metrics,
                    "config": {
                        "d_model": d_model,
                        "d_ff": d_ff,
                        "num_heads": num_heads,
                        "enc_layers": enc_layers_eff,
                        "dec_layers": dec_layers_eff,
                        "use_cross_attn": use_cross_attn,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "warmup_ratio": warmup_ratio,
                        "grad_clip": grad_clip,
                    },
                }, ckpt_path)
                print(f"[train] saved checkpoint: {ckpt_path}", flush=True)
            except Exception as e:
                print(f"[train] WARN: failed to save checkpoint at epoch {epoch}: {e}", flush=True)

    elapsed = time.time() - start_time
    print(f"[train] finished in {elapsed/60.0:.1f} min")

    # 保存最终成果
    # 1) 模型
    model_path = os.path.join("results", f"model-final-{run_tag}.pt")
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    print(f"[train] saved model to {model_path}")
    # 2) 词表
    vocab_path = os.path.join("results", f"vocab-final-{run_tag}.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"stoi": train_ds.stoi, "itos": train_ds.itos}, f, ensure_ascii=False)
    print(f"[train] saved vocab to {vocab_path}")
    # 3) 指标（final）
    metrics_path = os.path.join("results", f"metrics-final-{run_tag}.json")
    final_payload = {**metrics, "config": {
        "d_model": d_model, "d_ff": d_ff, "num_heads": num_heads,
        "enc_layers": enc_layers_eff, "dec_layers": dec_layers_eff,
        "use_cross_attn": use_cross_attn
    }}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"[train] saved metrics to {metrics_path}")
    # 4) 参数统计
    params_path = os.path.join("results", f"params-{run_tag}.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "seq2seq": count_parameters(model),
            "config": {
                "d_model": d_model, "d_ff": d_ff, "num_heads": num_heads,
                "enc_layers": enc_layers_eff, "dec_layers": dec_layers_eff,
                "use_cross_attn": use_cross_attn
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"[train] saved params to {params_path}")


if __name__ == "__main__":
    main()
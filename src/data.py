import os
import glob
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset


class TinyShakespeare(Dataset):
    def __init__(self, path, seq_len=128):
        with open(path) as f:
            text = f.read()
        vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[ch] for ch in text])
        self.seq_len, self.vocab_size = seq_len, len(vocab)
        self.pad_id = 0

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


class WikiText2Dataset(Dataset):
    """
    WikiText-2 数据集封装。优先使用本地文件 data/wikitext/wikitext-2-raw-v1。
    若本地不存在再尝试从 Hugging Face 加载（需要联网）。
    支持 split: train / validation(valid) / test。
    """

    def __init__(self, split="train", tokenizer=None, seq_len=128, root=None):
        # 1. 选择本地文件路径
        file_map = {
            "train": "wiki.train.raw",
            "validation": "wiki.valid.raw",
            "valid": "wiki.valid.raw",
            "test": "wiki.test.raw",
        }
        if root is None:
            root = os.path.join(".", "data", "wikitext", "wikitext-2-raw-v1")
        local_raw_path = os.path.join(root, file_map.get(split, "wiki.train.raw"))

        texts = None

        # 2. 优先尝试读取本地 Arrow 文件（HF 缓存格式）
        split_name = "validation" if split in ("validation", "valid") else split
        arrow_pattern = os.path.join(root, "0.0.0", "*", f"wikitext-{split_name}.arrow")
        arrow_candidates = glob.glob(arrow_pattern)
        if arrow_candidates:
            ds = HFDataset.from_file(arrow_candidates[0])
            texts = ds["text"]
        elif os.path.exists(local_raw_path):
            # 3. 读取原始 .raw 文本
            with open(local_raw_path, "r", encoding="utf-8") as f:
                text = f.read()
            texts = text.splitlines()
        else:
            # 4. 尝试联网从 HF 加载（若无网络会失败）
            try:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
                texts = ds["text"]
            except Exception as e:
                raise RuntimeError(
                    f"WikiText2Dataset: 本地 Arrow/RAW 均未找到且联网加载失败。请将 wikitext-2-raw-v1 放到 {root}，或恢复网络。原始错误: {e}"
                )

        # 4. 构建 tokenizer（或使用传入的）
        if tokenizer is None:
            vocab = sorted(set("".join(texts)))
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi, self.itos = tokenizer["stoi"], tokenizer["itos"]

        # 5. 编码为整数序列
        all_text = "\n".join(texts)
        self.data = torch.tensor([self.stoi.get(ch, 0) for ch in all_text])
        self.seq_len = seq_len
        self.vocab_size = len(self.stoi)
        self.pad_id = 0

        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


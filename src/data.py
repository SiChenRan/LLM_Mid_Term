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


class IWSLT2017Seq2Seq(Dataset):
    """
    使用本地 data/IWSLT2017_EN_DE/<split>/*.arrow 读取 IWSLT2017 EN↔DE 数据，生成字符级
    (src, tgt_in, tgt_out) 三元组，用于 Encoder-Decoder 训练。

    - 默认方向 en→de，可通过 src_lang/tgt_lang 指定；
    - vocab 共享（包含 src 与 tgt 的字符以及 PAD/BOS/EOS 特殊符号）；
    - 目标序列添加 BOS/EOS：tgt_in=[BOS]+tgt_ids, tgt_out=tgt_ids+[EOS]；
    - 不在此处 padding，交由 DataLoader 的 collate_fn 处理。
    """
    def __init__(
        self,
        split: str = "train",
        root: str | None = None,
        seq_len: int = 128,
        tokenizer: dict | None = None,
        src_lang: str = "en",
        tgt_lang: str = "de",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
    ):
        # 记录数据集标识与方向，便于文件命名
        self.dataset_name = "IWSLT2017"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        if root is None:
            root = os.path.join(".", "data", "IWSLT2017_EN_DE")
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"IWSLT2017Seq2Seq: 未找到 split 目录: {split_dir}")

        arrow_candidates = glob.glob(os.path.join(split_dir, "*.arrow"))
        if not arrow_candidates:
            raise RuntimeError(f"IWSLT2017Seq2Seq: 未在 {split_dir} 找到 Arrow 文件")
        ds = HFDataset.from_file(arrow_candidates[0])

        col = "translation"
        if col not in ds.column_names:
            raise RuntimeError(f"IWSLT2017Seq2Seq: Arrow 文件中缺少列 '{col}'，实际列: {ds.column_names}")

        src_texts, tgt_texts = [], []
        for ex in ds:
            tr = ex[col]
            s = tr.get(src_lang, "")
            t = tr.get(tgt_lang, "")
            if not isinstance(s, str) or not isinstance(t, str):
                continue
            src_texts.append(s)
            tgt_texts.append(t)

        # 构建或沿用 tokenizer（字符级，含特殊符号）
        if tokenizer is None:
            vocab_chars = set()
            for txt in src_texts:
                vocab_chars.update(list(txt))
            for txt in tgt_texts:
                vocab_chars.update(list(txt))
            vocab = [pad_token, bos_token, eos_token] + sorted(vocab_chars)
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi, self.itos = tokenizer["stoi"], tokenizer["itos"]

        self.pad_id = self.stoi.get(pad_token, 0)
        self.bos_id = self.stoi[bos_token]
        self.eos_id = self.stoi[eos_token]
        self.seq_len = seq_len
        self.vocab_size = len(self.stoi)

        def encode(text: str):
            return [self.stoi.get(ch, self.pad_id) for ch in text]

        self.samples = []
        for s, t in zip(src_texts, tgt_texts):
            src_ids = encode(s)[:seq_len]
            # 目标预留 BOS/EOS，至少保留 1 个 token
            tgt_ids = encode(t)[:max(1, seq_len - 2)]
            tgt_in = [self.bos_id] + tgt_ids
            tgt_out = tgt_ids + [self.eos_id]
            self.samples.append((torch.tensor(src_ids), torch.tensor(tgt_in), torch.tensor(tgt_out)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


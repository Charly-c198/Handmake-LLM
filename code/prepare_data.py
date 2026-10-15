"""
把原始文本语料转成训练用的二进制 token 流（.bin）。对应手册第 6 章。

为什么要转成 .bin：
  训练时是从几十亿 token 里随机采样。如果每次都用 Python 逐条解码文本，
  GPU 会一直等 CPU（这就是"数据加载瓶颈"，MFU 上不去的主因）。
  转成一个巨大的 uint16/uint32 数组，用 memmap 只读要用的那一段，
  操作系统按页缓存，CPU 开销几乎为零。

为什么用 memmap 而不是直接 np.load：
  16GB 内存的机器装不下 20GB 的 token 数组。memmap 把它当"虚拟文件"用，
  只把访问到的页读进内存。这是小内存机器能训大语料的关键。

用法:
    # 字符级（零依赖，适合第一次跑通全流程）
    python prepare_data.py --mode char --input corpus.txt --out data/tinystories

    # BPE（需要一个 HuggingFace 的 tokenizers 库）
    python prepare_data.py --mode hf --input corpus.txt --out data/mine \
        --vocab_size 8192

    # 从 HuggingFace 直接拉数据集（需要 datasets 库 + 能访问外网）
    python prepare_data.py --mode hf --hf_dataset roneneldan/TinyStories \
        --out data/tinystories --vocab_size 8192
"""

import argparse
import os
import pickle

import numpy as np


# ---------------------------------------------------------------------------
def write_bin(tokens, path, vocab_size):
    """把 token 序列写成二进制。uint16 支持 vocab<=65535，比 uint32 省一半 IO。"""
    if vocab_size <= 65535:
        dtype = np.uint16
    elif vocab_size <= 2**32 - 1:
        dtype = np.uint32
    else:
        raise ValueError(f"词表太大: {vocab_size}")
    arr = np.array(tokens, dtype=dtype)
    arr.tofile(path)
    # 返回 dtype 的字符串名（'uint16'/'uint32'），train.py 靠它决定怎么读文件
    return arr.nbytes, arr.dtype.name


def split_and_write(all_tokens, out_dir, vocab_size, val_ratio=0.0005):
    os.makedirs(out_dir, exist_ok=True)
    n = len(all_tokens)
    n_val = max(1024, int(n * val_ratio))
    val, train = all_tokens[:n_val], all_tokens[n_val:]
    b1, dt = write_bin(train, os.path.join(out_dir, "train.bin"), vocab_size)
    b2, _ = write_bin(val, os.path.join(out_dir, "val.bin"), vocab_size)
    print(f"[data] train: {len(train):,} token ({b1/1e6:.1f} MB)")
    print(f"[data] val  : {len(val):,} token ({b2/1e6:.1f} MB)")
    return dt


# ---------------------------------------------------------------------------
def build_char(corpus, out_dir):
    """字符级词表：用最低频字符做 unk。最简单，永远不会 OOV 失败。"""
    print("[data] 构建字符级词表...")
    chars = sorted(set(corpus))
    vocab_size = len(chars) + 1          # +1 给 <unk>
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["<unk>"] = 0
    itos = {i: c for c, i in stoi.items()}
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos, "mode": "char"}
    print(f"[data] 字符集大小: {len(chars)}  (语料总长 {len(corpus):,})")
    tokens = [stoi.get(c, 0) for c in corpus]
    dt = split_and_write(tokens, out_dir, vocab_size)
    meta["dtype"] = str(dt)
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[data] meta.pkl 已写入 (vocab_size={vocab_size})")
    return meta


def build_bpe(corpus, out_dir, vocab_size, tokenizer_sample_chars=200_000_000):
    """用 HuggingFace tokenizers 训练 byte-level BPE。这是实战路线。

    为什么用 HF 而不是我们自己写的 bpe.py：
      bpe.py 是 O(n·merges) 的纯 Python，1GB 语料要跑几天。
      HF tokenizers 是 Rust 实现 + 并行 + 增量索引，快 100 倍以上。

    内存注意（2GB 语料实测）：
      若按"tokens.extend(ids)"累积成 Python list，2GB 语料会产生约 5 亿个
      Python int，需要 ~17GB 内存，16GB 机器必然 OOM。
      这里改成"分批 encode_batch -> 立刻转 numpy uint16 -> 累积数组"，
      Python int 只在一批内存在，峰值内存降到几百 MB。
    """
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    except ImportError:
        raise SystemExit(
            "需要 tokenizers 库:  pip install tokenizers\n"
            "或者先用 --mode char 跑通全流程"
        )

    print(f"[data] 训练 byte-level BPE, vocab_size={vocab_size} ...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # 特殊 token 必须在训 tokenizer 时就预留（手册 6.3）：
        # 训完再加会导致 embedding 扩容。endoftext 做样本分隔/停止符，
        # im_start/im_end 留给后面 SFT 对话模板。
        special_tokens=["<|unk|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    # 用语料前若干字符训练 tokenizer 就够了——合并统计会饱和，
    # 没必要把整个 2GB 喂进去，省时间也省内存。
    sample = corpus[:tokenizer_sample_chars]
    print(f"[data] tokenizer 训练样本: {len(sample):,} 字符")
    tokenizer.train_from_iterator([sample], trainer=trainer, length=len(sample))
    del sample

    tok_path = os.path.join(out_dir, "tokenizer.json")
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(tok_path)
    real_vocab = tokenizer.get_vocab_size()
    print(f"[data] 实际词表大小: {real_vocab}, tokenizer 已存到 {tok_path}")

    # 分批并行编码：encode_batch 走多线程，每批转 numpy 后立刻释放 Python int
    dtype = np.uint16 if real_vocab <= 65535 else np.uint32
    CHUNK, BATCH = 1_000_000, 64
    n_chunks = (len(corpus) + CHUNK - 1) // CHUNK
    parts = []
    for b in range(0, n_chunks, BATCH):
        batch = [corpus[(b + j) * CHUNK:(b + j + 1) * CHUNK]
                 for j in range(min(BATCH, n_chunks - b))]
        encs = tokenizer.encode_batch(batch)
        parts.append(np.concatenate([np.asarray(e.ids, dtype=dtype) for e in encs]))
        del batch, encs
        print(f"\r[data] 编码 {min((b + BATCH) * CHUNK, len(corpus)):,}/{len(corpus):,}", end="")
    print()

    tokens = np.concatenate(parts) if len(parts) > 1 else parts[0]
    del parts
    dt = split_and_write(tokens, out_dir, real_vocab)
    meta = {"vocab_size": real_vocab, "mode": "bpe", "tokenizer_path": tok_path,
            "dtype": str(dt)}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    compress = len(corpus) / max(1, len(tokens))
    print(f"[data] 压缩率: {compress:.2f} 字符/token （英文 BPE 通常 3~4，中文 1.5~2.5）")
    return meta


# ---------------------------------------------------------------------------
def load_corpus(args):
    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit("需要 datasets 库:  pip install datasets")
        print(f"[data] 从 HuggingFace 拉取 {args.hf_dataset} ...")
        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        text_field = args.hf_field
        if text_field not in ds.column_names:
            # 常见的几种字段名，自动猜
            for cand in ["text", "story", "content", "document", "raw_content"]:
                if cand in ds.column_names:
                    text_field = cand
                    break
            else:
                raise SystemExit(f"找不到文本字段，可选: {ds.column_names}")
        print(f"[data] 使用字段 '{text_field}', {len(ds):,} 条")
        if args.max_docs:
            ds = ds.select(range(min(args.max_docs, len(ds))))
        return "\n\n".join(ds[text_field])

    if not args.input:
        raise SystemExit("必须提供 --input 或 --hf_dataset")
    files = args.input if isinstance(args.input, list) else [args.input]
    parts = []
    for fp in files:
        print(f"[data] 读取 {fp} ...")
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["char", "hf"], default="char")
    ap.add_argument("--input", nargs="*", default=None, help="本地文本文件，可多个")
    ap.add_argument("--hf_dataset", default=None, help="如 roneneldan/TinyStories")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--hf_field", default="text")
    ap.add_argument("--max_docs", type=int, default=None, help="只取前 N 条（调试用）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--vocab_size", type=int, default=8192)
    args = ap.parse_args()

    corpus = load_corpus(args)
    print(f"[data] 语料总字符数: {len(corpus):,}")

    if args.mode == "char":
        build_char(corpus, args.out)
    else:
        build_bpe(corpus, args.out, args.vocab_size)

    print("\n下一步:  python train.py --data_dir " + args.out)


if __name__ == "__main__":
    main()

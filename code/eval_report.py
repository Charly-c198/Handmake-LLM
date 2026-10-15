"""
P3 评估脚本：把训练日志变成简历上能直接用的三样东西。
  1) train/val loss 曲线图            -> <out_dir>/loss_curve.png
  2) 20 条固定 prompt 的生成样例      -> <out_dir>/samples.md
  3) 一份评测摘要（含 PPL/参数量/吞吐） -> <out_dir>/eval_summary.md

为什么 prompt 要固定：只有每次用同一组 prompt，才能跨 checkpoint / 跨模型
对比"谁生成得更好"。随机 prompt 的评测没有意义。

用法:
    cd D:/llm-work/handmade-llm/code
    python eval_report.py --ckpt D:/llm-work/out/nano/ckpt_final.pt \
        --data_dir D:/llm-work/data/tinystories \
        --log D:/llm-work/out/nano/train.log \
        --out_dir D:/llm-work/out/nano
"""

import argparse
import math
import os
import re
import statistics

import torch
import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接存文件（无 GUI 环境也能跑）
import matplotlib.pyplot as plt

from sample import load_model, load_tokenizer, generate


# 20 个固定 prompt —— 改了就没法跨版本对比了，别随便动
PROMPTS = [
    "Once upon a time",
    "Once upon a time, there was a little girl named",
    "Once upon a time, there was a little boy named",
    "One day, a little dog named Max",
    "The cat sat on the mat and",
    "Tom was very sad because",
    "Lily and her mom went to the park.",
    "A little bird fell out of its nest.",
    "There was a big red ball in the garden.",
    "The sun was shining and the sky was blue.",
    "Sara found a shiny rock on the ground.",
    "One morning, Ben woke up very early.",
    "The little train chugged up the hill.",
    "A kind old man lived in a small house.",
    "Mia wanted to bake a cake for her dad.",
    "The puppy was afraid of the dark.",
    "In the forest, a rabbit met a turtle.",
    "It was a rainy day and Anna was bored.",
    "Jack built a tall tower with his blocks.",
    "The moon was big and round that night.",
]


# ---------------------------------------------------------------------------
def parse_log(path):
    """从 train.log 里抓出所有评测点：iter / train loss / val loss。"""
    pat = re.compile(r"\[iter\s+(\d+)\]\s+train\s+([\d.]+)\s+val\s+([\d.]+)")
    iters, tr, va = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                iters.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                va.append(float(m.group(3)))
    return iters, tr, va


def parse_throughput(path):
    """抓日志里的 tok/s（取中位数，去掉第一步 warmup 的异常值）。"""
    vals = [float(x) for x in re.findall(r"\| ([\d.]+)k tok/s \|", open(path, encoding="utf-8", errors="replace").read())]
    return statistics.median(vals) * 1000 if vals else None


def plot_curve(iters, tr, va, png_path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, tr, label="train", linewidth=1.6)
    ax.plot(iters, va, label="val", linewidth=1.6)
    best_i = iters[va.index(min(va))]
    ax.scatter([best_i], [min(va)], color="red", zorder=5, s=30)
    ax.annotate(f"best val {min(va):.4f} @ {best_i}",
                (best_i, min(va)), textcoords="offset points", xytext=(8, 10), color="red")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 曲线已保存 -> {png_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--log", default=None, help="train.log，用于画曲线；不传就跳过画图")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=20, help="生成多少条（默认全部 20 个 prompt）")
    ap.add_argument("--prompts_file", default=None,
                    help="自定义 prompt 文件（每行一个）。中文模型必须传这个，否则用默认英文 prompt")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # 用哪个 prompt 列表：命令行给的文件优先，否则用内置的英文 20 条
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        if not prompts:
            raise SystemExit(f"prompt 文件是空的: {args.prompts_file}")
    else:
        prompts = PROMPTS

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print("=" * 70)
    print(f"P3 评估 | ckpt={args.ckpt}")
    print("=" * 70)

    # ---- 1. 加载模型 + tokenizer ----
    model, ckpt = load_model(args.ckpt, device)
    cfg = model.cfg
    encode, decode, meta = load_tokenizer(args.data_dir)
    eos_id = None
    if meta["mode"] == "bpe":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(meta["tokenizer_path"])
        eos_id = tok.token_to_id("<|endoftext|>")
    print(f"[eval] 参数量 {model.num_params()/1e6:.2f}M | block_size={cfg.block_size} | "
          f"vocab={cfg.vocab_size} | eos_id={eos_id}")

    # ---- 2. loss 曲线 ----
    n_iters = tr_pts = va_pts = None
    if args.log and os.path.exists(args.log):
        iters, tr_pts, va_pts = parse_log(args.log)
        print(f"[eval] 从日志解析出 {len(iters)} 个评测点")
        if iters:
            plot_curve(iters, tr_pts, va_pts,
                       os.path.join(args.out_dir, "loss_curve.png"),
                       os.path.basename(args.out_dir) + "  train/val loss")
    else:
        print("[eval] 未提供 --log，跳过曲线绘制")

    # ---- 3. 批量生成 ----
    print("\n" + "-" * 70)
    print(f"生成 {args.n_samples} 条样例（temperature={args.temperature} top_k={args.top_k} "
          f"rep_pen={args.repetition_penalty}）")
    print("-" * 70)
    results = []
    for i, p in enumerate(prompts[:args.n_samples]):
        ids = encode(p)
        idx = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = generate(model, idx, args.max_new_tokens, cfg,
                           args.temperature, args.top_k, args.top_p,
                           args.repetition_penalty, eos_id, stream=False)
        gen_ids = out[0, len(ids):].tolist()
        text = decode(gen_ids)
        results.append((p, text, len(gen_ids)))
        print(f"\n[{i+1:2d}/{args.n_samples}] prompt: {p!r}  ({len(gen_ids)} token)")
        print(f"        {text.strip()[:220]}")

    # ---- 4. 写 samples.md ----
    samples_md = os.path.join(args.out_dir, "samples.md")
    with open(samples_md, "w", encoding="utf-8") as f:
        f.write(f"# 生成样例（{os.path.basename(args.out_dir)}）\n\n")
        f.write(f"- checkpoint: `{os.path.basename(args.ckpt)}`（iter={ckpt.get('iter')}, "
                f"best_val={ckpt.get('best_val', float('nan')):.4f}）\n")
        f.write(f"- 采样参数: temperature={args.temperature}, top_k={args.top_k}, "
                f"repetition_penalty={args.repetition_penalty}, max_new_tokens={args.max_new_tokens}\n\n")
        for i, (p, text, n) in enumerate(results, 1):
            f.write(f"## {i}. `{p}`\n\n")
            f.write(f"**续写**（{n} token）：\n\n> {p}{text.strip()}\n\n")
    print(f"\n[write] 样例已保存 -> {samples_md}")

    # ---- 5. 评测摘要 ----
    tp = parse_throughput(args.log) if (args.log and os.path.exists(args.log)) else None
    best_val = min(va_pts) if va_pts else ckpt.get("best_val", float("nan"))
    final_val = va_pts[-1] if va_pts else float("nan")
    summary = os.path.join(args.out_dir, "eval_summary.md")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"# 评测摘要（{os.path.basename(args.out_dir)}）\n\n")
        f.write("| 项目 | 值 |\n|---|---|\n")
        f.write(f"| 参数量 | {model.num_params()/1e6:.2f}M（不含 embedding {model.num_params(True)/1e6:.2f}M）|\n")
        f.write(f"| 层数 / 头数 / 维度 | L{cfg.n_layer} / H{cfg.n_head} / d{cfg.n_embd}（KV 头 {cfg.n_kv_head}）|\n")
        f.write(f"| 上下文长度 | {cfg.block_size} |\n")
        f.write(f"| 词表大小 | {cfg.vocab_size} |\n")
        f.write(f"| 训练步数 | {ckpt.get('iter')} |\n")
        f.write(f"| **best val loss** | **{best_val:.4f}** |\n")
        f.write(f"| final val loss | {final_val:.4f} |\n")
        f.write(f"| **困惑度 PPL** | **{math.exp(min(20, best_val)):.2f}** |\n")
        if tp:
            f.write(f"| 训练吞吐 | {tp/1000:.1f}k tok/s |\n")
        f.write(f"| 生成样例数 | {len(results)} |\n")
    print(f"[write] 摘要已保存 -> {summary}")

    print("\n" + "=" * 70)
    print(f"评测完成：{args.out_dir} 下应有 loss_curve.png / samples.md / eval_summary.md")
    print(f"best val {best_val:.4f}  PPL {math.exp(min(20, best_val)):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
从零预训练脚本。对应手册第 8 章。

针对 RTX 5060 Laptop 8GB 显存做了默认配置。核心优化：
  - memmap 取数据（不占内存）
  - bf16 混合精度（Blackwell 原生支持，不需要 GradScaler）
  - 梯度累积（凑出大 batch，不增加显存）
  - 梯度裁剪 + warmup + cosine 衰减（大模型训练的三件套）
  - 断点续训（笔记本会休眠、会断电，必须能续）

用法:
    python train.py --data_dir data/tinystories
    python train.py --data_dir data/tinystories --n_layer 8 --n_embd 512 --max_iters 20000
    python train.py --data_dir data/tinystories --resume out/ckpt.pt
"""

import argparse
import math
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

# RTX 5060 Laptop 的 bf16 理论峰值（含稀疏是 2 倍，这里取 dense 估算值）。
# 用途只是估算 MFU，填个大概数就行。
PEAK_FLOPS = 120e12


class _NoScaler:
    """非 fp16 训练时的 GradScaler 替身。

    bf16 的指数范围（8 位）和 fp32 一样，不会溢出，所以不需要缩放梯度。
    用一个 no-op 对象顶替，训练循环里就不用到处写 if scaler is not None。
    """
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


# ---------------------------------------------------------------------------
def get_batch(data, block_size, batch_size, device):
    """从 memmap 里随机切 batch_size 段长度为 block_size+1 的窗口。

    x = 窗口前 block_size 个 token
    y = 窗口后 block_size 个 token（即 x 右移一位）
    这就是"预测下一个 token"的训练信号。
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = torch.from_numpy(np.stack([data[i:i + block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64))
    if device.startswith("cuda"):
        # pin_memory 让 H2D 拷贝能异步，省掉 CPU 等待（CPU-only 机器上调用会报错）
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, args):
    """warmup + cosine 衰减。这是训练能收敛的关键，不是可选项。"""
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it > args.max_iters:
        return args.min_lr
    ratio = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(model, data, args, device, ctx, eval_iters=50):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split], args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="out")
    # --- 模型 ---
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--n_kv_head", type=int, default=3, help="GQA，默认是 n_head 的一半")
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    # --- 优化 ---
    p.add_argument("--batch_size", type=int, default=16, help="单次前向的序列条数（显存主要吃这个）")
    p.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95, help="LLM 标准是 0.95，不是 Adam 默认的 0.999")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_iters", type=int, default=10000)
    p.add_argument("--warmup_iters", type=int, default=0, help="0 表示自动 = 2% max_iters")
    # --- 系统 ---
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--compile", action="store_true", help="torch.compile（Windows 上可能失败）")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    # --- 日志 ---
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--sample_interval", type=int, default=0, help="每 N 步打印一段生成，0=关")
    p.add_argument("--sample_prompt", default="Once upon a time",
                   help="训练中途打印样例用的 prompt。中文模型务必改成中文，否则样例无意义")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke", action="store_true", help="只跑 30 步验证流程")
    args = p.parse_args()

    if args.smoke:
        args.max_iters = 30
        args.eval_interval = 15
        args.n_layer, args.n_head, args.n_embd = 2, 2, 64
        args.batch_size, args.grad_accum, args.block_size = 4, 1, 128

    if args.warmup_iters == 0:
        args.warmup_iters = max(50, int(0.02 * args.max_iters))

    # ---- 设备 ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device_type == "cuda":
        torch.cuda.manual_seed(args.seed)
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}, "
              f"算力 sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
        print(f"[env] torch {torch.__version__}, CUDA {torch.version.cuda}, "
              f"显存 {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB")

    # ---- 数据类型 ----
    if args.dtype == "bf16" and device_type == "cpu":
        print("[warn] CPU 上 bf16 很慢，自动切 fp32")
        args.dtype = "fp32"
    ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ctx = (torch.autocast(device_type=device_type, dtype=ptdtype)
           if args.dtype != "fp32" else torch.autocast(device_type=device_type, enabled=False))
    # fp16 必须用 GradScaler 兜住上溢/下溢；bf16 不需要（指数范围和 fp32 一样）
    scaler = torch.amp.GradScaler(device_type) if args.dtype == "fp16" else _NoScaler()
    print(f"[env] dtype={args.dtype}  grad_accum={args.grad_accum}  "
          f"tokens/step={args.batch_size * args.grad_accum * args.block_size:,}")

    # ---- 数据 ----
    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    data_dtype = np.uint16 if meta.get("dtype", "uint16") == "uint16" else np.uint32
    data = {
        "train": np.memmap(os.path.join(args.data_dir, "train.bin"),
                           dtype=data_dtype, mode="r"),
        "val": np.memmap(os.path.join(args.data_dir, "val.bin"),
                         dtype=data_dtype, mode="r"),
    }
    print(f"[data] vocab_size={vocab_size}  "
          f"train={len(data['train']):,} token  val={len(data['val']):,} token")
    print(f"[data] 训练规模 ≈ {args.max_iters * args.batch_size * args.grad_accum * args.block_size:,} token"
          f"  (语料只有 {len(data['train']):,}，会重复采样 "
          f"{args.max_iters * args.batch_size * args.grad_accum * args.block_size / max(1,len(data['train'])):.1f} 轮)")

    # ---- 模型 ----
    cfg = GPTConfig(vocab_size=vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    n_kv_head=args.n_kv_head, dropout=args.dropout)
    model = GPT(cfg)
    print(f"[model] {model.num_params()/1e6:.2f}M 参数 "
          f"(不含 embedding {model.num_params(True)/1e6:.2f}M)")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    start_iter, best_val = 0, float("inf")
    if args.resume:
        print(f"[resume] 从 {args.resume} 续训")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    model.to(device)
    if args.compile:
        print("[opt] torch.compile ...")
        model = torch.compile(model)

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # ---- wandb ----
    if args.wandb:
        import wandb
        wandb.init(project="handmade-llm", config=vars(args))

    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.grad_accum * args.block_size
    t0 = time.time()
    last_log_time, last_log_iter = t0, start_iter

    # ======================= 训练主循环 =======================
    for it in range(start_iter, args.max_iters + 1):
        lr = get_lr(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- 评估 & 存档 ----
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, data, args, device, ctx, args.eval_iters)
            print(f"\n[iter {it:>6d}] train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"ppl {math.exp(min(20, losses['val'])):.2f}  lr {lr:.2e}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                _save(args.out_dir, "ckpt_best.pt", raw_model, optimizer, args, cfg, it, best_val)
            if args.wandb:
                import wandb
                wandb.log({"val": losses["val"], "train": losses["train"], "lr": lr}, step=it)

        if it % args.save_interval == 0 and it > 0:
            _save(args.out_dir, f"ckpt_{it}.pt", raw_model, optimizer, args, cfg, it, best_val)
            _save(args.out_dir, "ckpt.pt", raw_model, optimizer, args, cfg, it, best_val)

        if args.sample_interval and it > 0 and it % args.sample_interval == 0:
            _quick_sample(raw_model, meta, device, args)

        if it == args.max_iters:
            break

        # ---- 前向 + 反向（梯度累积） ----
        for micro in range(args.grad_accum):
            X, Y = get_batch(data["train"], args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.grad_accum          # 关键：归一化，否则等效 lr 放大 grad_accum 倍
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ---- 日志 ----
        if it % args.log_interval == 0:
            now = time.time()
            dt = now - last_log_time
            n = max(1, it - last_log_iter)
            tok_s = tokens_per_iter * n / max(dt, 1e-9)
            # 估算剩余时间
            eta_h = (args.max_iters - it) * dt / max(n, 1) / 3600
            print(f"iter {it:>6d}/{args.max_iters} | loss {loss.item()*args.grad_accum:.4f} "
                  f"| lr {lr:.2e} | {tok_s/1000:.1f}k tok/s | ETA {eta_h:.1f}h", flush=True)
            last_log_time, last_log_iter = now, it

    # ---- 收尾 ----
    dt = time.time() - t0
    print(f"\n[done] 总耗时 {dt/3600:.2f}h, best val {best_val:.4f}")
    _save(args.out_dir, "ckpt_final.pt", raw_model, optimizer, args, cfg, args.max_iters, best_val)
    print(f"[done] 权重在 {args.out_dir}/ckpt_final.pt")
    print(f"       推理:  python sample.py --ckpt {args.out_dir}/ckpt_final.pt --prompt 'Once upon a time'")


def _save(out_dir, name, model, opt, args, cfg, it, best_val):
    path = os.path.join(out_dir, name)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": {k: v for k, v in vars(cfg).items()},
        "args": vars(args),
        "iter": it,
        "best_val": best_val,
    }, tmp)
    os.replace(tmp, path)      # 原子替换，防止存档时断电写出半个文件


def _quick_sample(model, meta, device, args):
    model.eval()
    if meta["mode"] == "char":
        stoi, itos = meta["stoi"], meta["itos"]
        start = "The" if all(ord(c) < 128 for c in list(stoi)[:5]) else "从前"
        ids = [stoi.get(c, 0) for c in start] or [0]
        out = model.generate(torch.tensor([ids], device=device), 80,
                             temperature=0.8, top_k=40, eos_id=None)
        print("[sample] " + "".join(itos.get(int(i), "?") for i in out[0]))
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(meta["tokenizer_path"])
        ids = tok.encode(args.sample_prompt).ids
        eos_id = tok.token_to_id("<|endoftext|>")
        out = model.generate(torch.tensor([ids], device=device), 80,
                             temperature=0.8, top_k=40, eos_id=eos_id)
        print(f"[sample] {args.sample_prompt}" + tok.decode(out[0, len(ids):].tolist()))
    print()
    model.train()


if __name__ == "__main__":
    main()

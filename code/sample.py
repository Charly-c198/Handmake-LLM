"""
推理/采样脚本 + 手写 KV Cache 生成循环。对应手册第 10 章。

这个文件刻意不用 transformers 的 generate，而是自己写循环，
因为你要理解"推理到底发生了什么"——采样策略、KV cache、温度、重复惩罚。

用法:
    # 单次补全
    python sample.py --ckpt out/ckpt_final.pt --prompt "Once upon a time"

    # 交互式对话（需要模型做过 SFT，见手册第 9 章）
    python sample.py --ckpt out/ckpt_final.pt --chat

    # 调采样参数
    python sample.py --ckpt out/ckpt_final.pt --prompt "Lily" --temperature 0.9 --top_p 0.95
"""

import argparse
import os
import pickle

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, KVCache


# ---------------------------------------------------------------------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**{k: v for k, v in ckpt["config"].items()
                       if k in GPTConfig.__dataclass_fields__})
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[load] {ckpt_path}  iter={ckpt.get('iter')}  "
          f"best_val={ckpt.get('best_val', float('nan')):.4f}  "
          f"{model.num_params()/1e6:.2f}M 参数")
    return model, ckpt


def load_tokenizer(data_dir, meta=None):
    if meta is None:
        with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
    if meta["mode"] == "char":
        def enc(s):
            return [meta["stoi"].get(c, 0) for c in s]
        def dec(ids):
            return "".join(meta["itos"].get(int(i), "?") for i in ids)
        return enc, dec, meta
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(meta["tokenizer_path"])
        return (lambda s: tok.encode(s).ids), (lambda ids: tok.decode(ids)), meta


# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, idx, max_new_tokens, cfg,
             temperature=0.8, top_k=50, top_p=None,
             repetition_penalty=1.0, eos_id=None, stop_str=None,
             stream=True, decode_fn=None):
    """手写生成循环。每一步：前向 -> 取最后一个位置的 logits -> 采样 -> 拼回去。

    为什么要有 KV cache：
      没有 cache 时，生成第 t 个 token 要把前 t 个 token 重新前向一遍，
      总计算量 O(T^3)。有 cache 是 O(T^2)。1000 token 的输出差 300 倍。
    """
    B, T = idx.shape
    assert T <= cfg.block_size, f"prompt 长度 {T} 超过 block_size {cfg.block_size}"
    device = idx.device
    cache = KVCache(cfg, B, cfg.block_size, device, model.tok_emb.weight.dtype)

    # --- prefill：把整个 prompt 一次性喂进去（所有位置并行，很快） ---
    # 返回的 logits 是最后一个位置(T-1)的，用来预测第 T 个 token。
    logits, _ = model(idx, cache=cache, start_pos=0)
    logits = logits[:, -1, :].float()
    pos = T                      # pos = 下一个要写入 cache 的位置
    out = idx.clone()
    generated_text = ""

    for step in range(max_new_tokens):
        # --- 重复惩罚：已经出现过的 token，logits 往下压 ---
        if repetition_penalty != 1.0:
            for b in range(B):
                for t in torch.unique(out[b]):
                    v = logits[b, t]
                    logits[b, t] = v / repetition_penalty if v > 0 else v * repetition_penalty

        # --- 贪心 ---
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                # 保留累计概率首次超过 top_p 的那个位置，之后全砍掉
                remove = cumprobs - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, nxt], dim=1)

        if stream and decode_fn is not None:
            ch = decode_fn([nxt[0].item()])
            generated_text += ch
            print(ch, end="", flush=True)
            if stop_str and generated_text.endswith(stop_str):
                break

        if eos_id is not None and (nxt == eos_id).all():
            break

        # --- 把新 token 喂回模型，写到 cache 的 pos 位置，拿下一个 logits ---
        if pos >= cfg.block_size:
            if stream:
                print(f"\n[stop] 达到 block_size={cfg.block_size} 上限")
            break
        logits, _ = model(nxt, cache=cache, start_pos=pos)
        logits = logits[:, -1, :].float()
        pos += 1

    if stream and decode_fn is not None:
        print()
    return out


# ---------------------------------------------------------------------------
CHAT_TEMPLATE = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default=None, help="含 meta.pkl 的目录，默认从 ckpt 里推")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_stream", action="store_true")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, ckpt = load_model(args.ckpt, device)
    cfg = model.cfg

    data_dir = args.data_dir or ckpt["args"].get("data_dir")
    if not data_dir or not os.path.exists(os.path.join(data_dir, "meta.pkl")):
        raise SystemExit("找不到 meta.pkl，请用 --data_dir 指定 prepare_data.py 的输出目录")
    encode, decode, meta = load_tokenizer(data_dir)

    eos_id = meta.get("stoi", {}).get("<|im_end|>")
    if meta["mode"] == "bpe":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(meta["tokenizer_path"])
        eos_id = tok.token_to_id("<|im_end|>")

    print(f"\n[gen] temperature={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} rep_pen={args.repetition_penalty}")
    print("-" * 66)

    # ---------------- 交互式 ----------------
    if args.chat or args.prompt is None:
        print("交互模式。输入内容回车，输入 /exit 退出。\n")
        history = ""
        while True:
            try:
                user = input("你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user in ("/exit", "/quit"):
                break
            history += CHAT_TEMPLATE.format(user=user)
            ids = encode(history)
            if len(ids) > cfg.block_size - args.max_new_tokens:
                ids = ids[-(cfg.block_size - args.max_new_tokens):]
            idx = torch.tensor([ids], device=device)
            print("AI> ", end="", flush=True)
            out = generate(model, idx, args.max_new_tokens, cfg,
                           args.temperature, args.top_k, args.top_p,
                           args.repetition_penalty, eos_id,
                           stop_str="<|im_end|>",
                           stream=not args.no_stream, decode_fn=decode)
            history += decode(out[0, len(ids):].tolist())
        return

    # ---------------- 单次补全 ----------------
    print(f"prompt: {args.prompt!r}\n")
    idx = torch.tensor([encode(args.prompt)], device=device)
    out = generate(model, idx, args.max_new_tokens, cfg,
                   args.temperature, args.top_k, args.top_p,
                   args.repetition_penalty, eos_id,
                   stream=not args.no_stream, decode_fn=decode)
    if args.no_stream:
        print(decode(out[0].tolist()))

    # 展示速度
    n = out.size(1) - idx.size(1)
    print(f"\n[stats] 生成 {n} token, ctx={cfg.block_size}, "
          f"KV cache 占用 {KVCache(cfg,1,cfg.block_size,device,model.tok_emb.weight.dtype).nbytes()/1e6:.1f} MB")


if __name__ == "__main__":
    main()

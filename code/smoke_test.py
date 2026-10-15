"""
端到端冒烟测试。对应手册第 12 章的"验收标准"。

它会真的把你的三个脚本按真实顺序跑一遍：
    prepare_data.py  ->  train.py  ->  sample.py
全部在 CPU 上用玩具规模（1 分钟以内），不需要下载任何数据。

为什么要做这个：
  手搓 LLM 最常见的失败是"数据/代码/权重对不上"——tokenizer 换了、
  vocab_size 改了、block_size 不匹配。先让这个测试全绿，再去碰真数据。

用法:
    python smoke_test.py
"""

import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# Windows 中文环境下子进程默认用 GBK 编码 stdout，遇到 emoji 之类会崩。
# 强制子进程用 UTF-8。
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run(cmd, **kw):
    print(f"\n$ {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=ENV, **kw)
    if r.returncode != 0:
        print("STDOUT:\n", r.stdout[-4000:])
        print("STDERR:\n", r.stderr[-4000:])
        raise SystemExit(f"[FAIL] 命令失败: {' '.join(cmd)}")
    return r.stdout + r.stderr


CORPUS = """Once upon a time, there was a little girl named Lily.
Lily loved to play in the garden with her dog, Max. Every morning,
they ran through the tall green grass and chased the butterflies.
One sunny day, Lily found a tiny bird with a broken wing near the old oak tree.
She took the bird home and made a soft bed for it in a small shoebox.
Every morning, Lily gave the bird seeds and fresh water from the blue bowl.
Soon the bird got better and could fly again. It sang a happy song to say thank you.
Lily waved goodbye as the bird flew up into the bright sky.
""" * 120


def main():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="llm_smoke_")
    data_dir = os.path.join(tmp, "data")
    out_dir = os.path.join(tmp, "out")
    corpus_path = os.path.join(tmp, "corpus.txt")

    print("=" * 70)
    print("第 1 关：环境自检")
    print("=" * 70)
    import torch
    print(f"python   {sys.version.split()[0]}")
    print(f"torch    {torch.__version__}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"cuda     {torch.version.cuda}  {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
        # Blackwell(sm_120) 必须能真的跑一个 kernel，否则装的是错的 wheel
        x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
        assert torch.isfinite(x).all(), "CUDA kernel 输出非法值"
        print("[ok] GPU kernel 可执行")
    else:
        print("cuda     不可用（本测试将用 CPU，没问题）")

    print("\n" + "=" * 70)
    print("第 2 关：模型单元检查（model.py 自带的 assert）")
    print("=" * 70)
    out = run([PY, "model.py"])
    assert "[ok] model.py 全部检查通过" in out, "model.py 自检未通过"
    assert "KV cache 一致性" in out
    print("[ok] 前向/反向/KV cache 正确性")

    print("\n" + "=" * 70)
    print("第 3 关：BPE 分词器往返一致性")
    print("=" * 70)
    out = run([PY, "bpe.py"])
    assert "[ok] 往返编码一致性验证通过" in out, "BPE 往返编码失败"
    print("[ok] BPE encode/decode 无损")

    print("\n" + "=" * 70)
    print("第 4 关：数据准备（字符级）")
    print("=" * 70)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(CORPUS)
    out = run([PY, "prepare_data.py", "--mode", "char",
               "--input", corpus_path, "--out", data_dir])
    m = re.search(r"vocab_size=(\d+)", out)
    assert m, "没解析出 vocab_size"
    vocab_size = int(m.group(1))
    assert os.path.exists(os.path.join(data_dir, "train.bin"))
    assert os.path.exists(os.path.join(data_dir, "meta.pkl"))
    print(f"[ok] 数据落盘, vocab_size={vocab_size}")

    print("\n" + "=" * 70)
    print("第 5 关：训练（CPU，80 步，玩具规模）")
    print("=" * 70)
    out = run([
        PY, "train.py",
        "--data_dir", data_dir, "--out_dir", out_dir,
        "--device", "cpu", "--dtype", "fp32",
        "--n_layer", "2", "--n_head", "2", "--n_embd", "64",
        "--n_kv_head", "1", "--block_size", "128",
        "--batch_size", "8", "--grad_accum", "1",
        "--max_iters", "80", "--eval_interval", "40",
        "--eval_iters", "10", "--log_interval", "20",
        "--save_interval", "80", "--warmup_iters", "10",
    ])
    evals = [(int(a), float(b)) for a, b in
             re.findall(r"\[iter\s+(\d+)\] train [\d.]+  val ([\d.]+)", out)]
    print(f"\n评测点: {evals}")
    assert len(evals) >= 2, "评测次数不足，检查 eval_interval"
    first_val = evals[0][1]
    last_val = evals[-1][1]
    assert last_val < first_val, \
        f"[FAIL] loss 没下降! {first_val:.4f} -> {last_val:.4f}（lr 太大？数据有问题？）"
    print(f"[ok] val loss 下降: {first_val:.4f} -> {last_val:.4f}")

    ckpt = os.path.join(out_dir, "ckpt_final.pt")
    assert os.path.exists(ckpt), "没生成 ckpt_final.pt"
    print(f"[ok] 检查点已保存: {os.path.getsize(ckpt)/1e6:.2f} MB")

    print("\n" + "=" * 70)
    print("第 6 关：推理（加载检查点 + KV cache 生成）")
    print("=" * 70)
    out = run([PY, "sample.py", "--ckpt", ckpt, "--data_dir", data_dir,
               "--prompt", "Once upon", "--max_new_tokens", "40",
               "--temperature", "0.7", "--top_k", "20", "--seed", "0"])
    assert "[stats]" in out, "生成没跑完"
    m = re.search(r"\[stats\] 生成 (\d+) token", out)
    assert m and int(m.group(1)) > 0, "没生成出 token"
    print(f"[ok] 生成 {m.group(1)} 个 token, KV cache 正常")

    print("\n" + "=" * 70)
    print(f"全部通过  用时 {time.time()-t0:.1f}s")
    print("=" * 70)
    print(f"""
下一步（真实训练）:
  1. 准备真数据，比如 TinyStories 英文故事集
  2. python prepare_data.py --mode hf --input your_corpus.txt --out data/mine --vocab_size 8192
  3. python train.py --data_dir data/mine --n_layer 6 --n_embd 384 --max_iters 20000
  4. python sample.py --ckpt out/ckpt_final.pt --prompt "Once upon a time"

临时文件在: {tmp}
""")


if __name__ == "__main__":
    main()

"""
从零实现一个 Llama 风格的自回归 Transformer。对应手册第 7 章。

部件清单（现代 LLM 的"标准件"）：
  - Pre-Norm + RMSNorm          （不是 LayerNorm / Post-Norm）
  - RoPE 旋转位置编码            （不是可学习位置 embedding）
  - GQA 分组查询注意力 + KV Cache （不是 MHA）
  - SwiGLU 前馈网络              （不是 GELU MLP）
  - 无 bias、权重绑定（小模型省参数）

和 nanoGPT 的区别：nanoGPT 是 GPT-2 架构（LayerNorm/GELU/绝对位置编码），
能跑，但已经过时了。学就学 Llama 这套，因为 Cons/Llama/Qwen/DeepSeek 全是它。

用法:
    from model import GPTConfig, GPT
    cfg = GPTConfig(vocab_size=4096, block_size=512, n_layer=6, n_head=6, n_embd=384)
    model = GPT(cfg)
    logits, loss = model(idx, targets)
"""

import math
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512        # 最大上下文长度 T
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384            # 必须能被 n_head 整除
    n_kv_head: Optional[int] = None   # GQA：None 表示等于 n_head（即 MHA）
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd 必须能被 n_head 整除"
        assert self.n_head % self.n_kv_head == 0, "n_head 必须是 n_kv_head 的整数倍"

    @property
    def head_dim(self):
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight

    比 LayerNorm 少一个均值中心化和 bias 项。论文说"效果一样但更快"，
    实际上是现代 LLM 的默认选择。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 关键：用 float32 算方差。bf16 下 x.pow(2) 很容易下溢成 0，
        # 导致 sqrt(0+eps) 除法爆炸 —— 这是最常见的 NaN 来源之一。
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


# ---------------------------------------------------------------------------
# RoPE 旋转位置编码
# ---------------------------------------------------------------------------
def precompute_rope(head_dim, max_seq_len, base=10000.0):
    """预计算 cos/sin 表，形状 (max_seq_len, head_dim//2)。以后不用重算。"""
    # inv_freq[i] = base^(-2i/head_dim), i = 0..head_dim/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (T, D/2) 每个位置 × 每个频率
    return torch.cos(freqs), torch.sin(freqs)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, start_pos=0):
    """x: (B, H, T, D)。就地旋转每个 (x, y) 二维子空间。

    这里用 Llama/HF 的 "前后对半" 约定：
        维度 i 和维度 i + D/2 组成一个二维子空间，一起旋转。
    （另一种是先隔一个取一个的 GPT-J 约定，两者不通用，
      这也是"自己写的模型不能直接加载别人权重"的常见原因。）
    """
    T = x.size(2)
    c = cos[start_pos:start_pos + T].to(x.dtype)     # (T, D/2)
    s = sin[start_pos:start_pos + T].to(x.dtype)
    c = torch.cat([c, c], dim=-1).unsqueeze(0).unsqueeze(0)   # (1,1,T,D)
    s = torch.cat([s, s], dim=-1).unsqueeze(0).unsqueeze(0)
    return x * c + _rotate_half(x) * s


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------
class KVCache:
    """推理时缓存历史 K/V，把每个新 token 的注意力计算从 O(T^2) 降到 O(T)。

    显存占用 = 2 * n_layer * batch * n_kv_head * max_seq * head_dim * dtype_size
    这是 GQA 的价值所在：n_kv_head 减少 4 倍，KV cache 也小 4 倍。
    """

    def __init__(self, cfg, batch_size, max_seq_len, device, dtype):
        shape = (batch_size, cfg.n_kv_head, max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.seen = 0

    def reset(self):
        self.seen = 0

    def nbytes(self):
        return sum(t.numel() * t.element_size() * 2 for t in self.k)


# ---------------------------------------------------------------------------
# 注意力
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout
        # Q 有 n_head 个头，K/V 只有 n_kv_head 个头（GQA 省显存）
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, layer_idx, cache=None, start_pos=0):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, start_pos)
        k = apply_rope(k, cos, sin, start_pos)

        if cache is not None:
            cache.k[layer_idx][:, :, start_pos:start_pos + T] = k
            cache.v[layer_idx][:, :, start_pos:start_pos + T] = v
            k = cache.k[layer_idx][:, :, :start_pos + T]
            v = cache.v[layer_idx][:, :, :start_pos + T]

        # GQA：把 kv 头复制到和 q 头一样的数量
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        Tk = k.size(2)
        dp = self.dropout_p if self.training else 0.0

        if T == 1:
            # 生成阶段：1 个 query 看全部历史，无需 mask
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)
        elif Tk == T:
            # 训练 / prefill 阶段：标准因果掩码
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            # 带 cache 的多 token 前向：需要显式带状掩码
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU 前馈
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """FFN(x) = down( silu(gate(x)) * up(x) )

    为什么 hidden 是 8/3 倍而不是 4 倍？SwiGLU 有 3 个矩阵（gate/up/down）
    而普通 MLP 只有 2 个，用 8/3 倍可以在参数量上与 4 倍普通 MLP 持平。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)      # 对齐到 64，kernel 更快
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, layer_idx, cache=None, start_pos=0):
        # Pre-Norm：先归一化再进子层，残差直通。训练大模型时比 Post-Norm 稳得多。
        x = x + self.attn(self.attn_norm(x), cos, sin, layer_idx, cache, start_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 的残差缩放：把残差分支输出层的初始化标准差除以 sqrt(2*n_layer)，
        # 保证深层网络的残差流方差不会随层数爆炸。
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -----------------------------------------------------------------------
    def forward(self, idx, targets=None, cache=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, \
            f"序列超长: {start_pos}+{T} > {self.cfg.block_size}"

        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos, self.rope_sin
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, i, cache, start_pos)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            return logits, loss
        else:
            # 推理时只算最后一个位置，省一大半计算
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, repetition_penalty=1.0, eos_id=None):
        """简单的采样生成（带 KV cache）。完整版见 sample.py。"""
        self.eval()
        device = idx.device
        B = idx.size(0)
        cache = KVCache(self.cfg, B, self.cfg.block_size, device, self.tok_emb.weight.dtype)

        # prefill：把 prompt 一次性喂进去（所有位置并行）。
        # 返回的 logits 是最后一个位置(T-1)的，用来预测第 T 个 token。
        logits, _ = self(idx, cache=cache, start_pos=0)
        logits = logits[:, -1, :].float()
        pos = idx.size(1)          # pos = 下一个要写入 cache 的位置

        out = idx
        for _ in range(max_new_tokens):
            # --- 重复惩罚：出现过的 token 把 logits 往下压 ---
            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = torch.unique(out[b])
                    vals = logits[b, seen]
                    logits[b, seen] = torch.where(vals > 0, vals / repetition_penalty,
                                                  vals * repetition_penalty)

            # --- 采样 ---
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    p = F.softmax(sl, dim=-1)
                    cum = torch.cumsum(p, dim=-1)
                    sl[cum - p > top_p] = -float("inf")
                    logits = sl.scatter(1, si, sl)
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if pos >= self.cfg.block_size:
                break

            # --- 把新 token 写到 cache 的 pos 位置，拿下一个 logits ---
            logits, _ = self(out[:, -1:], cache=cache, start_pos=pos)
            logits = logits[:, -1, :].float()
            pos += 1
        return out

    # -----------------------------------------------------------------------
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_weights:
                n -= self.lm_head.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, lr, betas=(0.9, 0.95), device_type="cuda"):
        """AdamW 参数分组：2 维以上（权重矩阵）做 weight decay，1 维（norm/bias）不做。

        对 norm 和 embedding 做 weight decay 是有害的，这是标准做法。
        """
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay, no_decay = [], []
        for n, p in params.items():
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
        except Exception:
            # fused AdamW 在某些 torch/平台组合上不可用，回退
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_flops):
        """估算 MFU（模型 FLOPs 利用率）。健康区间 0.2~0.5。
        低于 0.15 说明你被数据加载或 kernel 启动开销卡住了。"""
        N = self.num_params()
        flops_per_token = 6 * N            # 前向 2N + 反向 4N
        flops_per_iter = flops_per_token * self.cfg.block_size * fwdbwd_per_iter
        return flops_per_iter / (dt * peak_flops)

    @classmethod
    def from_pretrained(cls, *a, **kw):
        raise NotImplementedError(
            "我们是从零训练的，不加载别人的权重。"
            "如果你想微调现成模型，请看手册第 9 章（用 transformers + peft）。"
        )


# ---------------------------------------------------------------------------
# 参数量速查 + demo
# ---------------------------------------------------------------------------
def param_breakdown(cfg: GPTConfig):
    h, d, L, V = cfg.n_head, cfg.n_embd, cfg.n_layer, cfg.vocab_size
    kv = cfg.n_kv_head * cfg.head_dim
    attn = d * (h * cfg.head_dim) + d * kv * 2 + (h * cfg.head_dim) * d
    hidden = 64 * ((int(8 * d / 3) + 63) // 64)
    mlp = 3 * d * hidden
    per_layer = attn + mlp + 2 * d          # +2d 是两个 RMSNorm
    emb = V * d
    total = per_layer * L + emb + (0 if cfg.tie_weights else emb)
    return {"attn": attn * L, "mlp": mlp * L, "norm": 2 * d * L,
            "embed": emb, "total": total}


if __name__ == "__main__":
    torch.manual_seed(1337)
    print("=" * 66)
    print("常见配置参数量速查（词表 32000）")
    print("=" * 66)
    presets = {
        "nano (教学)":   dict(n_layer=6,  n_head=6,  n_embd=384,  block_size=512),
        "tiny (玩具)":   dict(n_layer=8,  n_head=8,  n_embd=512,  block_size=512),
        "small (可用)":  dict(n_layer=12, n_head=12, n_embd=768,  block_size=1024),
        "medium":        dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
    }
    for name, kw in presets.items():
        cfg = GPTConfig(vocab_size=32000, **kw)
        m = GPT(cfg)
        b = param_breakdown(cfg)
        print(f"{name:14s} {m.num_params()/1e6:7.2f}M 参数 "
              f"(attn {b['attn']/1e6:5.1f}M / mlp {b['mlp']/1e6:5.1f}M / "
              f"emb {b['embed']/1e6:5.1f}M)")

    # ---- 前向 / 反向 / 生成 冒烟 ----
    print("\n" + "=" * 66)
    print("冒烟测试")
    print("=" * 66)
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg)
    x = torch.randint(0, 256, (2, 64))
    y = torch.randint(0, 256, (2, 64))
    logits, loss = model(x, y)
    print(f"前向: logits {tuple(logits.shape)}  loss {loss.item():.4f}")
    loss.backward()
    print(f"反向: 梯度范数 {sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None).sqrt():.4f}")

    # KV cache 正确性：先把前 8 个 token prefill 进 cache，
    # 再用 cache 单独前向第 9 个 token，结果必须和"一次性全量前向 9 个 token"完全一致。
    # 这是检验 KV cache 实现（位置编码 offset、mask、cache 写入位置）的黄金判据。
    model.eval()
    seq = torch.randint(0, 256, (1, 9))
    with torch.no_grad():
        full_logits, _ = model(seq)                                   # 全量
        cache = KVCache(cfg, 1, cfg.block_size, seq.device, model.tok_emb.weight.dtype)
        model(seq[:, :-1], cache=cache, start_pos=0)                  # prefill 前 8 个
        step_logits, _ = model(seq[:, -1:], cache=cache, start_pos=8)  # 增量第 9 个
    diff = (full_logits[:, -1, :] - step_logits[:, 0, :]).abs().max().item()
    print(f"KV cache 一致性: 最大 logits 误差 = {diff:.2e}  "
          f"{'[ok]' if diff < 1e-3 else '[FAIL] 有 bug!'}")
    assert diff < 1e-3, f"KV cache 实现错误，误差 {diff:.2e}"

    out = model.generate(seq, max_new_tokens=16, temperature=0.8, top_k=50)
    print(f"生成: {tuple(out.shape)} (prompt 9 + new 16 = 25)")
    assert out.size(1) == 25
    print("\n[ok] model.py 全部检查通过")

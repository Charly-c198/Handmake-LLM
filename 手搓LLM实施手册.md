# 手搓 LLM 实施手册

> 面向：**RTX 5060 Laptop 8GB · 16GB 内存 · Windows 11 · Python 3.13**
> 目标：从零训练出属于你自己的语言模型，并且理解每一行为什么这么写。
> 配套代码：`code/` 目录，已通过 `python code/smoke_test.py` 端到端验证。

---

## 0. 先读这一页：你能做到什么，做不到什么

一张表决定你的心理预期。**8GB 显存的边界不是"能不能训模型"，而是"多大的模型、多少数据、多久"。**

| 路线 | 参数量 | 训练数据 | 你这台机器的耗时 | 可行性 |
|---|---|---|---|---|
| **A. 打通全流程**（字符级） | 0.1–1M | 1 MB 文本 | 几十秒 | ✅ 今晚就能跑完 |
| **B. 会讲英文小故事** | 5–15M | TinyStories 200M token | 1–3 小时 | ✅ **强烈推荐的第一站** |
| **C. 能写通顺英文/简单中文** | 30–60M | 1–3B token | 1–3 天挂机 | ✅ 可以，注意散热 |
| **D. 复现 GPT-2 124M** | 124M | 10B token | 5–10 天不间断 | ⚠️ 显存够，时间紧，建议租卡 |
| **E. 从零训 1B+ 通用对话模型** | 1B+ | 20B+ token | 单卡不现实 | ❌ 租 8 卡或放弃 |
| **F. 做一个你自己的助手** | 0.5–7B（微调） | 几千~几万条指令 | 几分钟~几小时 | ✅ 用 Qwen 底座 + LoRA |

**两个必须先建立的认知：**

1. **"从零预训练"和"微调现成模型"是两条完全不同的路。** 手搓的价值在第一条（真正理解原理），实用价值在第二条（做出能用的东西）。这份文档两条都覆盖，按 A→B→C 学原理，最后用 F 出成果。
2. **真正卡住新手的从来不是 Transformer 架构**（那部分三天能学会），而是 **数据管道** 和 **数值稳定性**。所以本文档把 40% 的篇幅给了这两件事。

---

## 1. 三个必须背下来的公式

### 1.1 训练算力

```
C ≈ 6 × N × D
```

- `C` = 总浮点运算次数（FLOPs）
- `N` = 模型参数量（不含 embedding 更准，先忽略）
- `D` = 训练用掉的 token 总数

**为什么是 6**：前向传播每个参数要一次乘加 = 2N；反向传播要算"对输入的梯度"和"对权重的梯度"各一次 = 4N。所以每个 token 平均 6N。

> 例：124M 参数训 10B token → `6 × 1.24e8 × 1e10 = 7.4e18` FLOPs。

### 1.2 训练时间

```
时间(秒) = C / (峰值算力 × MFU)
```

**MFU（Model FLOPs Utilization）** 是"你实际用到了理论峰值的百分之多少"：

| 场景 | 现实 MFU |
|---|---|
| 大厂优化过的大模型训练 | 0.4 – 0.6 |
| **你在笔记本上训小模型** | **0.1 – 0.3** |
| 低于 0.1 | 被数据加载或 Python 开销卡死了，去查瓶颈 |

RTX 5060 Laptop 的 bf16 峰值大约在 **100 TFLOPS** 量级（厂商宣传的 "AI TOPS" 含稀疏和 FP4，别拿来算）。按 MFU=0.2 保守估，**有效算力 ≈ 20 TFLOPS**：

| 模型 | 数据量 | C | 预计耗时 |
|---|---|---|---|
| 10M | 200M token | 1.2e16 | **~10 分钟** |
| 30M | 1B token | 1.8e17 | **~2.5 小时** |
| 60M | 3B token | 1.1e18 | **~15 小时** |
| 124M | 10B token | 7.4e18 | **~4 天** |

> ⚠️ **上面这张表是理论估算，实测偏乐观。** 我在你这台机器上真跑之后测出的数字见下面「实测结果」一节——**请以实测为准**。

### 1.2.1 实测结果（本机真跑，2026-10-09）

torch 2.11.0+cu128 装好后，我用 `D:\llm-work\learn\gpu_bench.py` 真跑了一遍：

| 配置 | 参数量 | 吞吐 | **MFU** | 峰值显存 |
|---|---|---|---|---|
| nano `L6 d384` | 12.9M | **127.8k tok/s** | 0.082 | 1.99 GB |
| tiny `L8 d512` | 27.8M | **70.3k tok/s** | 0.098 | 2.42 GB |
| small `L12 d768` | 84.2M | **28.3k tok/s** | 0.119 | 3.87 GB |
| small `L12 d768, T=1024` | 84.2M | 26.3k tok/s | 0.111 | 3.88 GB |

**按实测吞吐换算的真实训练时间**：

| 配置 | 200M token | 1B token | 3B token |
|---|---|---|---|
| nano `L6 d384` (12.9M) | 0.4 h | **2.2 h** | 6.5 h |
| tiny `L8 d512` (27.8M) | 0.8 h | **4.0 h** | 11.9 h |
| small `L12 d768` (84.2M) | 2.0 h | **9.8 h** | 29.4 h |

**所以真实的目标应该是：**
- **想几小时内看到成果** → tiny `L8 d512` + 200M~1B token（0.8–4 小时）
- **想搞个像样的** → small `L12 d768` + 1B token（约 10 小时，挂一晚）
- **小型模型（<30M）的 MFU 天然就低（0.08–0.12）**，因为 GPU 太小喂不饱。**这不是你代码写得差。**

**显存悬崖（重要发现）**：84M 模型在 `block_size=512` 时，最优 batch 是 **8**；到 16 继续勉强（MFU 0.111）；**24 以上直接超出 8GB**，PyTorch 开始疯狂回收显存，速度崩到 11k tok/s（MFU 0.046）。所以：

> **batch_size 不是越大越好，超过显存后性能是断崖式下跌，不是缓慢下降。** 调参时如果发现吞吐突然掉一半，先怀疑是不是撞到显存上限了。

**先测你自己的真实速度，再定计划。** 直接跑我写好的脚本：

```bash
# 一键测全部配置（约 3 分钟）
D:/llm-work/venv/Scripts/python.exe D:/llm-work/learn/gpu_bench.py
```

或者只测手写版本：

```python
import torch, time
from model import GPT, GPTConfig
dev = "cuda"
cfg = GPTConfig(vocab_size=8192, block_size=512, n_layer=6, n_head=6, n_embd=384, n_kv_head=3)
m = GPT(cfg).to(dev).train()
opt = m.configure_optimizers(0.1, 6e-4)
x = torch.randint(0, 8192, (16, 512), device=dev)
y = torch.randint(0, 8192, (16, 512), device=dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    _, loss = m(x, y)
loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()                      # 预热
t0 = time.time(); N = 20
for _ in range(N):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(x, y)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
dt = (time.time() - t0) / N
tok_s = 16 * 512 / dt
print(f"{tok_s/1000:.1f}k tok/s | MFU≈{6*m.num_params()*16*512/dt/120e12:.3f} | "
      f"1B token 需 {1e9/tok_s/3600:.1f}h")
```

这个数字决定了上面表格里哪一行是你的目标。

### 1.3 显存

混合精度 + AdamW 训练时，**每个参数约 16 字节**：

| 项目 | 字节/参数 | 说明 |
|---|---|---|
| bf16 权重 | 2 | 前向反向用的 |
| fp32 master 权重 | 4 | 优化器更新用的高精度副本 |
| bf16 梯度 | 2 | |
| Adam 一阶/二阶动量（fp32） | 8 | 最大的那部分 |
| **合计** | **≈16** | |

再叠加**激活值**（中间结果），粗略公式：

```
激活显存 ≈ 34 × batch_size × seq_len × d_model × n_layer   字节
```

**你 8GB 卡的实际预算表：**

| 模型规模 | 优化器+参数+梯度 | 激活 (bs=8, T=512) | 总计 | 结论 |
|---|---|---|---|---|
| 10M (L6 d384) | 0.16 GB | 0.5 GB | 0.7 GB | 随便跑 |
| 60M (L8 d512) | 0.96 GB | 1.0 GB | 2.0 GB | 舒服 |
| **124M (L12 d768)** | **2.0 GB** | **1.3 GB** | **3.3 GB** | **推荐上限** |
| 300M (L24 d1024) | 4.8 GB | 1.7 GB (bs=4) | 6.5 GB | 要开梯度检查点 |
| 1B | 16 GB | — | — | ❌ 必须 ZeRO offload，速度降到 1/10 |

**结论：8GB 上从零预训练的现实上限是 ~300M 参数，舒服区间是 ≤150M。**

### 1.4 数据要多少（Chinchilla 定律）

```
D ≈ 20 × N
```

- 10M 参数 → 200M token
- 124M 参数 → 2.5B token（GPT-2 实际用了 10B，属于"过训"）

**但对小模型要反过来用这条定律**：TinyStories 那篇论文发现，10M 的模型喂 100B+ token 仍然在持续变好，远没有饱和。所以：

- **算力最优**（同样算力下 loss 最低）→ D = 20N
- **效果最优**（同样参数量下 loss 最低）→ 数据越多越好，能过训就过训

你的瓶颈是时间不是数据。**先用 20N 跑通，再决定要不要加数据。**

---

## 2. 你这台机器的实测画像

| 项目 | 实测值 | 对训练的影响 |
|---|---|---|
| GPU | RTX 5060 Laptop, **8151 MiB** | 硬上限，见上面预算表 |
| 架构 | **Blackwell, sm_120** | ⚠️ 必须装 CUDA 12.8+ 的 PyTorch，见 3.1 |
| 驱动 / CUDA | 592.01 / CUDA 13.1 | 驱动够新 |
| 功耗 | 95W (笔记本) | 长时间训练会降频 |
| 内存 | 15.8 GB | ⚠️ 数据必须 memmap 流式读 |
| CPU | Ryzen 9 8940HX, 32 逻辑核 | tokenize 数据很快，够用 |
| 磁盘 | D: 剩余 283 GB | 够。1B token 的 uint16 数据 = 2GB |
| Python | 3.13.9 | 需要 torch ≥ 2.6 |
| 工作路径 | `D:\llm-work\` | ✅ 无空格、无 UAC；代码在 `handmade-llm\`，环境在 `venv\` |

**关于 sm_120（最容易踩的坑）**

RTX 50 系是 Blackwell 新架构，它的 GPU 机器码是 `sm_120`。老的 PyTorch wheel（cu118/cu121/cu124）里根本没有 sm_120 的 kernel，装上后：

- `torch.cuda.is_available()` 返回 **True**（驱动没问题）
- 一旦真的跑运算就报 `CUDA error: no kernel image is available for execution on the device`

**所以唯一正确的装法是 cu128 及以上的 wheel。** 装完必须做这个验证：

```python
import torch
print(torch.cuda.get_device_capability(0))   # 必须是 (12, 0)
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
print(x.sum())                                # 不报错才算真装对了
```

---

## 3. 环境搭建

### 3.1 建立独立环境（已验证可用）

```bash
# 1) 建一个不带空格的独立环境目录（重要，见 3.4）
python -m venv D:/llm-work/venv

# 2) 装 PyTorch —— 必须指定 cu128 源
D:/llm-work/venv/Scripts/python.exe -m pip install --upgrade pip
D:/llm-work/venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3) 验证 sm_120
D:/llm-work/venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0)); print((torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum())"
```

**期望输出**：`2.11.0+cu128 (12, 0)` 和一个正常的浮点数。

> 如果 pip 下载慢（2.7GB），加国内镜像只对纯 Python 包有效，torch 的 CUDA wheel 走 `download.pytorch.org` 通常也能跑满带宽。真的卡住可以试 `-i https://mirror.sjtu.edu.cn/pytorch-wheels/cu128`。

**可选包**（按需，不要一次全装）：

```bash
pip install numpy tokenizers datasets          # 数据处理（推荐早装）
pip install transformers peft trl accelerate   # 微调现成模型（第 9 章才需要）
pip install tensorboard                        # 训练可视化
```

### 3.2 WSL2 路线：什么时候必须切过去

原生 Windows 能满足本文档 A/B/C 全程。**出现下面任一情况就切 WSL2**：

| 需求 | 为什么 Windows 不行 |
|---|---|
| `flash-attn` | 没有 Windows 预编译 wheel，自己编要 MSVC 折腾几小时 |
| `bitsandbytes` 做 4-bit 量化 | Windows wheel 近年才有，sm_120 支持仍然不稳 |
| vLLM 部署 | 官方不支持 Windows |
| FSDP / 多卡 | NCCL 在 Windows 上基本不可用 |
| `torch.compile` | Windows 上 triton 支持是后补的，容易编译失败 |

WSL2 的代价：内存要分给虚拟机（你只有 16GB，建议给 8GB），且要单独装一套 CUDA。**先别急着上 WSL，除非真的撞墙。**

### 3.3 建议的目录结构

```
D:\llm-work\                    ← 无空格！所有重活放这里
├── venv\                       ← 独立 Python 环境
├── data\                       ← token 数据（.bin 文件，可能几十 GB）
│   └── tinystories\
│       ├── train.bin
│       ├── val.bin
│       └── meta.pkl
├── out\                        ← 训练产物，checkpoint
└── corpus\                     ← 原始文本

D:\llm-work\handmade-llm\     ← 代码和文档（你正在看的地方）
├── 手搓LLM实施手册.md
├── README.md
└── code\
```

### 3.4 不要做的事（每条都是真实事故）

1. **不要把 venv / 数据 / checkpoint 放在含空格的路径下。** Python 本身没事，但 `subprocess` 调外部工具（ninja、MSVC、llama.cpp）时会因为引号处理炸掉。所以整套东西统一放在 `D:\llm-work\`（无空格、无 UAC）。**如果你哪天把项目挪回带空格的目录（如 `D:\Program Files\...`），这条立刻重新适用。**
2. **不要往你现有的 `.venv` 里装 torch。** 那个环境里已经有 `ag2 / langchain / cartopy` 等一堆包，torch 的 CUDA 依赖可能引发 ABI 冲突。用独立环境。
3. **不要混用 conda 和 pip 装 torch。** conda 装的 torch 和 pip 装的 numpy 经常 ABI 不兼容，表现为莫名其妙的 `ImportError: DLL load failed`。
4. **不要用 fp16 训练。** 用 bf16。fp16 会溢出成 NaN，需要 GradScaler 兜底，徒增复杂度。Blackwell 原生支持 bf16。

---

## 4. 四周路线图

| 周 | 目标 | 产出 | 关键技能 |
|---|---|---|---|
| **1** | 打通全流程 | 一个会拼英文单词的字符级模型 | 理解 tokenizer / 前向反向 / 采样 |
| **2** | 第一次真正的预训练 | 一个会讲简单故事的 10–15M 模型 | BPE、数据管道、loss 曲线判读 |
| **3** | 放大 + 评估 | 30–60M 模型 + 评测报告 | scaling、显存调优、评估方法 |
| **4** | 做出能用的东西 | 你自己的 Qwen-0.5B LoRA 助手 | 微调、量化、llama.cpp 部署 |

**每周的具体验收标准见第 12 章。**

---

## 5. 数据：决定成败的 80%

### 5.1 数据集选择（按学习曲线排序）

| 阶段 | 数据集 | 规模 | 说明 |
|---|---|---|---|
| 第 1 周 | `tinyshakespeare` | ~1 MB | 经典入门，字符级首选 |
| 第 2 周 | **`roneneldan/TinyStories`** | ~2 GB / 400M token | ⭐ **最佳第一站**：3 岁小孩词汇量的英文故事，小模型也能生成连贯文本，反馈极快 |
| 第 3 周 | `HuggingFaceFW/fineweb-edu` 的 `sample-10BT` | ~10B token | 高质量网页文本，业界标准预训练集 |
| 中文 | 见 5.4 | | |

```bash
# 用 datasets 拉 TinyStories
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
print(len(ds), ds.column_names)
ds.to_json('D:/llm-work/corpus/tinystories.jsonl', lines=True)
"
```

### 5.2 为什么 TinyStories 是完美的第一站

- 200M 参数以内的模型就能生成**语法正确、逻辑连贯**的故事（普通网页文本要 1B+ 才不胡言乱语）
- 数据干净、无版权问题、2GB 好下载
- 训练 2 小时就能看到肉眼可见的进步 → **反馈快，你才会坚持**

先在这里做出一个"你会想给朋友看"的模型，再谈放大。

### 5.3 数据清洗（不做的话后面全是坑）

| 问题 | 后果 | 处理 |
|---|---|---|
| **重复数据** | 模型死记硬背，val loss 虚低，泛化差 | MinHash-LSH 去重（`datasketch` 库） |
| HTML 残留 / 乱码 | 词表里塞满垃圾 token | 正则清洗、`ftfy` 修编码 |
| 超长文档 | 撑爆 block_size | 按段落切分，长度 512–2048 token |
| 隐私 / 有毒内容 | 模型输出有害内容 | 简单关键词过滤（学习阶段可略） |
| 不同来源混在一起 | 模型风格混乱 | 记数据配比，别随便改 |

**最低限度**：至少做"精确去重 + 长度过滤"。`fineweb-edu` 之所以好用，就是因为它用一个小分类器筛掉了"教育价值低"的网页——**质量 > 数量**这件事有实测支撑。

### 5.4 中文的特殊性（重要）

中文从零训练比英文难得多，原因不在模型，在数据和分词：

1. **token 效率低**：byte-level BPE 下一个汉字占 3 个字节。词表不够大时，一个汉字会被切成 2–3 个 token，等于把训练成本翻 3 倍。
   - 英文 BPE 约 **3–4 字符/token**
   - 天真的中文 BPE 只有 **1.5–2.5 字符/token**；好的中文 tokenizer（词表 100k+）能做到 **2.5–3.5 字符/token**
2. **数据量差距**：英文 fineweb 有 15T token，中文高质量语料一个数量级不到。**中文最大的瓶颈是数据，不是算力。**
3. **词表要大**：中文建议 **48k–100k**（Qwen 用 151k）。你 8GB 训小模型时，词表 64k 的 embedding 就占 `64000×384×2 = 49M` 参数，比模型主体还大——**小模型 + 大词表是浪费**，建议中文小模型词表 **32k** 起步。

**推荐的中文数据路线**：

| 用途 | 数据集 |
|---|---|
| 通用百科 | `wikimedia/wikipedia`（配置 `20231101.zh`），约 2.5GB / ~1B token |
| 指令/对话 | `BAAI/COIG`、`liwu/MNBVC` 的清洗子集 |
| **窄领域（强烈推荐先做这个）** | 古诗、小说、法律条文、医学——语料小但风格统一，小模型效果立竿见影 |

**新手建议**：如果要做中文，第一站选 **一个窄领域**（比如全唐诗，只有 ~5MB，但你能训出会写古诗的模型），而不是一上来啃通用中文。

### 5.5 数据格式：为什么是 .bin

原始文本 → token 序列 → **二进制文件**。原因：

- 训练时要从几亿 token 里**随机**采样。每次用 Python 解码文本会让 GPU 干等 CPU（这就是 MFU 上不去的主因）。
- 转成一个巨大的 `uint16` 数组，用 `np.memmap` 只读访问到的页，**操作系统按页缓存，CPU 开销几乎为零**。
- `uint16` 支持词表 ≤ 65535，比 `uint32` **省一半 IO 和磁盘**。词表 32k 时一定要用 uint16。

`code/prepare_data.py` 干的就是这件事。

---

## 6. Tokenizer

### 6.1 从零实现 BPE（对应 `code/bpe.py`）

BPE 的全部逻辑就三句话：

1. 把文本按 utf-8 拆成 **256 个字节**作为初始词表
2. 反复统计"**哪两个相邻 id 一起出现最多**"，把这一对合并成一个新 id
3. 重复 `词表大小 - 256` 次

**为什么用 byte-level**：任何字符串都能编码成 utf-8 字节，所以**永远不会出现未登录词（OOV）**。代价是中文一个字占 3 字节。

**为什么要"预切分"（pre-tokenization）**：如果直接在整篇文本上做 BPE，它会把 `"the"` 和 `" there"` 之间的 `e` + 空格合并成一个 token，导致这两个词共享一个奇怪的 token，泛化很差。GPT-2 的做法是先用正则把文本切成"词块"，BPE 只在词块内部合并。

```bash
python code/bpe.py     # 自带 demo：训练 + 编码 + 往返一致性验证
```

**这是理解 tokenizer 最好的方式，但它不是实战方案**——纯 Python 实现是 O(语料 × 合并次数)，1GB 语料要跑几天。

### 6.2 词表大小怎么选

| 词表 | 适用 | 权衡 |
|---|---|---|
| 256–2k | 字符级 / 学习 | 序列超长，模型难学长依赖 |
| **8k–16k** | **英文小模型** | ⭐ 8GB 上训英文的最优区间 |
| 32k | 英文标准 / 中文小模型 | GPT-2 用的就是这个 |
| 48k–100k | 中文 / 多语言 | 中文必须够大 |
| 151k | Qwen / 现代多语言 | embedding 参数巨大，小模型吃不消 |

**铁律：词表越大，embedding 层的参数量越大**（`vocab × d_model × 2`，因为输入输出各一个）。你训 30M 的模型，配 15 万词表，光是 embedding 就占 100M 参数——本末倒置。**小模型配小词表。**

### 6.3 实战用 HuggingFace tokenizers

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<|unk|>", "<|im_start|>", "<|im_end|>"],   # 预留对话用的特殊符号
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),       # 强制包含全部 256 字节
    show_progress=True,
)
tok.train_from_iterator(corpus_iter, trainer=trainer, length=len(corpus))
tok.save("tokenizer.json")
```

**注意那个 `initial_alphabet`**：不传的话 HF 只把"语料里出现过的字节"放进词表，以后遇到没见过的字符会产生 `<unk>`。传了才真正保证 byte-level 无 OOV。

**特殊 token 要提前预留**：`<|im_start|>` / `<|im_end|>` 这类对话标记必须在训练 tokenizer 时就加进去。训完再加会导致 embedding 需要扩容，麻烦。**这是新手最常犯的回溯性错误。**

### 6.4 自己训的 tokenizer 不能和别人共用

tokenizer 变了，token id 的含义就全变了，之前训好的模型权重直接作废。**改了 tokenizer 就从零重训。** 这是"训练配比记录"必须写下来的原因。

---

## 7. 模型架构

### 7.1 组件对照：别学 GPT-2，学 Llama

| 组件 | GPT-2（nanoGPT 用的） | **Llama（你应该学的）** | 为什么换 |
|---|---|---|---|
| 归一化 | LayerNorm | **RMSNorm** | 少一个均值中心化，更快，效果相当 |
| 归一化位置 | Post-Norm | **Pre-Norm** | 深层网络训练稳定得多 |
| 位置编码 | 可学习绝对位置 embedding | **RoPE 旋转位置编码** | 外推性更好，KV cache 更自然 |
| 注意力 | MHA（Q=K=V 头数） | **GQA 分组查询** | KV cache 显存省 4 倍，推理快 |
| 前馈 | GELU MLP (4×) | **SwiGLU** | 同参数量下效果更好 |
| 偏置 | 有 bias | **无 bias** | 省参数，无明显损失 |

**Llama / Qwen / DeepSeek / Mistral / 现代所有开源模型全是这一套。** 学它，你写的东西才有迁移价值。代码见 `code/model.py`。

### 7.2 每个组件一句话原理

- **RMSNorm**：`y = x / sqrt(mean(x²) + eps) × w`。只做缩放不做中心化。**实现时务必用 fp32 算方差**——bf16 下 `x.pow(2)` 容易下溢成 0，`sqrt(0+eps)` 会让除法爆炸，这是最常见的 NaN 来源。
- **RoPE**：把每个 (x,y) 二维子空间按位置旋转一个角度。让注意力天然感知"相对距离"。注意有**两种不兼容的约定**（前后对半 vs 隔一取一），这也是"自己写的模型加载不了别人权重"的常见原因。
- **GQA**：Q 有 `n_head` 个头，K/V 只有 `n_kv_head` 个。推理时 KV cache 大小正比于 `n_kv_head`，砍一半就省一半显存。训练时把 K/V 复制到 Q 的头数即可，**不影响训练速度**。
- **SwiGLU**：`down(silu(gate(x)) × up(x))`。三个矩阵，所以 hidden 维度取 `8/3 × d_model`（普通 MLP 是 4×）才能在参数量上对齐。
- **Pre-Norm + 残差**：`x = x + sublayer(norm(x))`。残差是恒等路径，梯度能直达底层。
- **权重绑定**：输入 embedding 和输出 lm_head 共用同一个矩阵。小模型上能省掉一大块参数，还有正则化效果。

### 7.3 参数量速查

跑 `python code/model.py` 会打印这张表（词表 32000）：

| 配置 | 参数量 | 8GB 能训吗 |
|---|---|---|
| L6 d384 （nano） | ~19M | ✅ 轻松 |
| L8 d512 （tiny） | ~34M | ✅ 轻松 |
| L12 d768 （small） | ~92M | ✅ 舒服 |
| L24 d1024 （medium） | ~280M | ⚠️ 需梯度检查点 |

其中**注意 FFN 占了大约 2/3 的参数**，这是正常现象——Transformer 的大部分"知识"存在 FFN 里。

### 7.4 初始化

不能全零（对称性无法打破），也不能太大（梯度爆炸）。标准做法：

- 权重：`normal_(mean=0, std=0.02)`
- **残差输出层额外缩小**：`o_proj` 和 `down` 的 std 再除以 `sqrt(2 × n_layer)`

第二条是 GPT-2 论文里的技巧。不加的话，层数一多残差流的方差会随层数累积爆炸。**层数 ≥ 12 时必加。**

### 7.5 一个必须理解的正确性检验

`code/model.py` 里有一个断言，你应该理解它在测什么：

```python
# 带 KV cache 逐 token 生成 与 一次性全量前向，logits 必须一致
full_logits, _ = model(prompt)
cache = KVCache(...); model(prompt, cache=cache)
step_logits, _ = model(prompt[:, -1:], cache=cache, start_pos=T-1)
assert (full_logits[:, -1] - step_logits[:, 0]).abs().max() < 1e-3
```

**这是 KV cache 实现是否正确的黄金判据。** 如果你的实现有 bug（位置编码 offset 错了、mask 错了），这个断言会立刻失败。**别跳过它。**

---

## 8. 训练

### 8.1 超参数怎么设

| 参数 | 小模型推荐值 | 说明 |
|---|---|---|
| `learning_rate` | **6e-4**（小模型可到 1e-3） | 最重要的参数。太大 loss 爆炸/震荡，太小不收敛 |
| `min_lr` | lr 的 1/10 | cosine 衰减的终点 |
| `warmup` | 总步数的 **2%** | 前几百步从 0 线性升上来，防早期炸梯度 |
| `beta2` | **0.95** | ⚠️ 不是 Adam 默认的 0.999。LLM 用 0.95 |
| `weight_decay` | 0.1 | 只对权重矩阵加，norm/bias 不加 |
| `grad_clip` | 1.0 | 梯度范数超过 1 就等比例缩小 |
| `tokens/step` | **32k–128k** | batch_size × grad_accum × block_size |
| `block_size` | 512（中文 512，英文 1024） | 上下文长度，注意力是 O(T²)，别贪大 |

**学习率的经验缩放**：batch 增大到 4 倍，lr 大致可以 ×1.4（不是 ×4）。不确定就用 6e-4 起手。

### 8.2 训练循环的每一步在干什么

对应 `code/train.py`：

```
for each step:
    1. 计算 lr（warmup + cosine 衰减）并写入 optimizer
    2. 梯度累积 grad_accum 次，每次：
         - 从 memmap 随机取一个 batch
         - bf16 前向，算 loss
         - loss / grad_accum  ← 关键！不除的话等效 lr 放大 grad_accum 倍
         - 反向
    3. 梯度裁剪（clip_grad_norm_）
    4. optimizer.step()   ← 一个 step 更新一次，但看到 grad_accum 个 batch
    5. optimizer.zero_grad(set_to_none=True)
```

**梯度累积的意义**：显存装不下大 batch，就用多个小 batch 累积梯度，等效于大 batch。这是 8GB 卡能训练的唯一办法。

**`loss / grad_accum` 这一步新手必错。** 忘了除，loss 数值看起来正常，但梯度大了 8 倍，等于 lr 偷偷变成 4.8e-3，训练直接发散。

### 8.3 显存不够的排查顺序

按这个顺序试，**代价从小到大**：

1. **减小 `batch_size`**（最有效，直接减激活值）
2. **减小 `block_size`**（注意力是二次的，512→256 显存减半）
3. 增大 `grad_accum` 补偿（不增显存）
4. **开梯度检查点**（`torch.utils.checkpoint`）：用 30% 的时间换 60% 的激活显存
5. 开 GQA（`n_kv_head` 设成 `n_head/2` 或 `/4`）
6. 降模型规模（`n_layer` / `n_embd`）—— 最后手段

**诊断命令**：在崩溃点之前插一句

```python
print(f"已分配 {torch.cuda.memory_allocated()/2**30:.2f} GB, "
      f"缓存 {torch.cuda.memory_reserved()/2**30:.2f} GB")
```

### 8.4 怎么读 loss 曲线

| 现象 | 诊断 | 处理 |
|---|---|---|
| train 和 val 同步下降 | ✅ 健康 | 继续 |
| **val 开始上升，train 继续下降** | 过拟合 | 加数据 / 加 dropout / 早停 |
| **train 和 val 都下不去** | 欠拟合或 lr 太小 | 加模型 / 加大 lr / 多训 |
| **loss 先降后突然飙到 NaN** | lr 太大 或 数值溢出 | 降 lr，检查 RMSNorm 是否用 fp32 |
| **loss 卡在 `log(vocab_size)` 附近不动** | 模型没学到任何东西 | 检查数据/标签是否错位、lr 是否太小 |
| loss 抖得厉害 | batch 太小 | 加大 grad_accum |
| **loss 值很漂亮但生成是乱码** | 过拟合了，或者数据/推理 tokenizer 不一致 | 对比 train/val，检查 tokenizer |

> `log(vocab_size)` 是随机模型的初始 loss：词表 8192 → 9.01，词表 32000 → 10.37。**看到 loss 从 10.37 降到 3 才算真的学到了东西。**

### 8.5 崩溃排查速查

| 报错 | 原因 | 解决 |
|---|---|---|
| `no kernel image is available` | torch 装的是老 CUDA 版本 | 重装 cu128 |
| `CUDA out of memory` | 见 8.3 | |
| `loss=nan` | lr 太大 / RMSNorm 没走 fp32 / 数据里有空样本 | 依次排查 |
| `RuntimeError: stack expects each tensor to be equal size` | 数据里有长度 < block_size 的片段 | get_batch 里加边界检查 |
| 训练速度逐渐变慢 | 笔记本降频 / 内存泄漏（`zero_grad` 没用 `set_to_none`） | |
| `BrokenPipeError` / dataloader 卡死 | Windows 上 `num_workers>0` | `num_workers=0` |
| 重启后 loss 从零开始 | checkpoint 没存 optimizer 状态 | 存 `optimizer.state_dict()` |

---

## 9. 后训练：从"会续写"到"能对话"

### 9.1 残酷的现实：小模型 SFT 没用

你的 10–60M 模型是**语言模型**（续写），不是**助手**（指令跟随）。对它做指令微调，它顶多学会"看到问句就编一个答案"，不会真的理解指令。

**原因**：指令跟随能力需要基础模型足够强（通常 ≥ 1B 且经过大量预训练）。**这是小模型的物理极限，不是你的技术问题。**

所以第 9 章分成两条路：

### 9.2 路线一：小模型的窄任务微调（可行）

让 30–60M 模型专精一件小事。比如你的模型是训古诗的：

```
输入: <|prompt|>写一首关于秋天的五言绝句<|response|>
输出: 空山新雨后，天气晚来秋。明月松间照，清泉石上流。<|end|>
```

准备 2000–10000 条这样的样本，SFT 一下，它就能做这一件事。**这就是小模型的正确用法：专精，而不是通用。**

### 9.3 路线二：LoRA 微调现成大模型（实用）

这是你真正能做出"有用的东西"的路线。用 Qwen 的现成底座，LoRA 微调。**8GB 完全可行**：

| 底座模型 | 显存需求 | 8GB 可行？ |
|---|---|---|
| Qwen2.5-0.5B | ~3 GB | ✅ 很轻松 |
| Qwen2.5-1.5B | ~5 GB | ✅ 可以，seq ≤ 1024 |
| Qwen2.5-3B (4bit) | ~4 GB | ✅ 需要 4-bit 量化 |
| Qwen2.5-7B (4bit) | ~6 GB | ⚠️ 极限，seq ≤ 512 + 梯度检查点 |

```bash
pip install transformers peft trl accelerate datasets bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_4bit=True,          # 4-bit 量化，显存减 4 倍
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # 全挂上，效果最好
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()    # 通常只有 1-2% 的参数可训练
```

**LoRA 的原理一句话**：冻结原权重，在旁边并联两个小矩阵 A(r×d) 和 B(d×r)，只训练这两个。可训练参数从 5 亿降到 500 万，显存和时间都大降，效果接近全参微调。

### 9.4 指令数据格式（对话模板）

**必须和 tokenizer 的特殊 token 对齐**。Qwen 的格式：

```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
{问题}<|im_end|>
<|im_start|>assistant
{回答}<|im_end|>
```

**做 loss mask**：只在 `assistant` 的回答部分算 loss，问题和 system 部分 target 设成 `-100`（`cross_entropy` 的 `ignore_index`）。否则模型会学着去生成用户的提问。**这是 SFT 最常见的错误。**

```python
# 把非 assistant 部分的 label 置为 -100
labels = input_ids.clone()
labels[~assistant_mask] = -100
```

---

## 10. 推理

### 10.1 KV Cache：推理性能的全部秘密

**没有 cache 时**，生成第 t 个 token 要把前 t 个 token 重新前向一遍，总计算 O(T³)。**有 cache 是 O(T²)。** 生成 1000 token，差 300 倍。代码见 `code/model.py` 的 `KVCache` 和 `code/sample.py`。

**原理**：自回归生成时，历史 token 的 K/V 不会变。把每层的 K/V 缓存起来，新 token 只算自己的 Q，和缓存里的 K/V 做注意力。

**cache 显存占用**：

```
2 × n_layer × batch × n_kv_head × max_seq_len × head_dim × 2 字节
```

例：L12, n_kv_head=6, head_dim=64, seq=512 → `2 × 12 × 1 × 6 × 512 × 64 × 2 = 9.4 MB`。很小。但 seq=8192 时会涨到 150MB——**这就是长上下文推理贵的原因**，也是 GQA 存在的意义。

### 10.2 采样策略

| 参数 | 作用 | 推荐 |
|---|---|---|
| `temperature` | 除以 logits，越大越随机 | 0.7–1.0（创作），0.2–0.5（事实） |
| `top_k` | 只从概率最高的 k 个里采 | 40–50 |
| `top_p`（核采样） | 累积概率到 p 就截断 | 0.9–0.95 |
| `repetition_penalty` | 压低出现过的 token | 1.1–1.3 |
| `temperature=0` | 贪心，永远选最大 | 输出去重、代码 |

**典型翻车**：只调 temperature 不调 top_k/top_p → 模型开始胡言乱语；不设 repetition_penalty → 模型无限重复一句话。**Top-k 和 repetition_penalty 几乎是必设的。**

### 10.3 部署：量化 + llama.cpp

训练出来的是 fp16/bf16 权重，推理时太占显存。**量化到 4-bit 能省 4 倍显存、快 2–3 倍，效果损失很小。**

```bash
# llama.cpp 在 Windows 上有预编译版，最省心的部署路径
# 1) 把 HF 格式转成 GGUF，再量化到 Q4_K_M
#    需要先用 transformers 把我们的 checkpoint 存成 HF 格式
# 2) 跑推理
./llama-cli -m model-q4_k_m.gguf -p "Once upon a time" -n 200
```

对你的自制小模型：直接用自己的 `sample.py` 就行（模型太小，量化的收益不大）。**llama.cpp 主要用在第 9 章微调的 Qwen 上**——q4_k_m 的 Qwen2.5-1.5B 只有 1GB，笔记本上跑得飞快。

---

## 11. 评估

### 11.1 三个层次

| 层次 | 指标 | 怎么做 |
|---|---|---|
| **1. 数值** | val loss / 困惑度 PPL | 每次 eval 打印，`PPL = exp(loss)` |
| **2. 任务** | 语法正确率、指令遵循率 | 自己写规则或用小模型打分 |
| **3. 人类** | "读起来像人话吗" | **自己读 50 条** —— 最有效的评估 |

**PPL 的坑**：PPL 只能跨**同一 tokenizer** 比较。你换了自己的 BPE，PPL 和别人论文里的数字完全没有可比性。**只拿它看自己的训练趋势。**

### 11.2 具体做法

- **TinyStories 模型**：让你的模型续写 20 个故事开头，人工数"有几条语法正确、逻辑通顺"。TinyStories 论文用的就是这个方法（GPT-4 打分），你自己读就行。
- **中文模型**：C-Eval / CMMLU 是标准榜单（`lm-eval-harness`），但**它们对 30M 的模型没有意义**——那是给 7B+ 准备的。你的模型该自己出题。
- **微调的 Qwen**：留出 100 条没训过的指令，人工看它做对多少条。

**关键原则：你的模型有多大，就用匹配它的评测。** 拿 30M 模型去刷 C-Eval 只会得到随机分数然后怀疑人生。

---

## 12. 验收标准（每周的"做完了"长什么样）

### 第 0 关 · 环境
```bash
python code/smoke_test.py
```
全部 6 关通过。这一步同时验证了 prepare_data / train / sample 的完整链路。

### 第 1 关 · 跑通全流程（第 1 周）
- [ ] 用 `code/bpe.py` 训一个 tokenizer，`encode` 后 `decode` 能**无损还原**（含中文、emoji）
- [ ] 用 tinyshakespeare（字符级）训 5000 步，loss 从 ~4.2 降到 **< 1.6**
- [ ] `sample.py` 生成的文本**看起来像莎士比亚**（有空格、换行、`thou/thee` 这类词，即使语义不通）
- [ ] 能说清 `start_pos` 在 KV cache 里代表什么

### 第 2 关 · 第一次真预训练（第 2 周）
- [ ] BPE 词表 8192，在 TinyStories 上训出一个 10–15M 模型
- [ ] val loss **< 2.0**（对应 PPL < 7.4）
- [ ] 输入 `"Once upon a time, there was a little"`，能续写出**语法正确、有完整情节**的 3–5 句话
- [ ] 训练速度 > **10k token/s**（没到这个数说明数据管道有瓶颈）

### 第 3 关 · 放大 + 评估（第 3 周）
- [ ] 30–60M 模型，在 ≥1B token 上训练
- [ ] 能画出 **train/val loss 曲线**并解释形状
- [ ] 用 20 个固定 prompt 生成，人工打分并记录（"语法正确率 X/20"）
- [ ] 跑一次 MFU 估算，能解释为什么是这个值

### 第 4 关 · 做出能用的东西（第 4 周）
- [ ] LoRA 微调 Qwen2.5-0.5B/1.5B，训 500 条你自己的指令数据
- [ ] 微调后模型在留出的 20 条指令上，明显优于微调前
- [ ] 量化成 GGUF q4_k_m，用 llama.cpp 在本地跑起来
- [ ] **能对朋友解释你整个 pipeline 的每一步在干什么**

---

## 13. 资源清单

### 必读的代码（按学习顺序）
| 项目 | 看什么 |
|---|---|
| **karpathy/nanoGPT** | 最小可用的 GPT 训练全流程，300 行。**先读它** |
| **karpathy/minbpe** | BPE 的教学实现，比我的还精简 |
| **karpathy/llm.c** | 纯 C 实现的训练，理解"下面到底发生了什么" |
| **meta-llama/llama**（或 HF 的 modeling_llama.py） | 工业级 Llama 实现 |
| **karpathy/LLM101n** | 从零造一个完整的 LLM 应用栈，课程形式 |

### 必读的论文
| 论文 | 一句话 |
|---|---|
| *Attention Is All You Need* (2017) | Transformer 的起源 |
| *Language Models are Unsupervised Multitask Learners* (GPT-2) | 初始化、权重绑定 |
| *Language Models are Few-Shot Learners* (GPT-3) | scaling 的开端 |
| *Training Compute-Optimal LLMs* (Chinchilla) | D ≈ 20N 的来源 |
| **LLaMA / LLaMA 2** | 现代架构的全部改动 |
| *RoFormer* | RoPE 的数学推导 |
| *GQA* | 分组查询注意力 |
| *GLU Variants Improve Transformer* | SwiGLU 的来历 |
| **TinyStories** | 小模型能力边界的关键发现 |
| LoRA / QLoRA | 微调的全部原理 |
| FlashAttention | 为什么注意力能这么快 |
| ZeRO / Megatron-LM | 多卡训练（等你上云再看） |

### 中文资料
- **李沐《论文精读》系列**（B站）—— 读论文的入门
- **《动手学深度学习》**（zh.d2l.ai）—— 补深度学习基础

---

## 附录 A：命令速查

```bash
# 环境（一次性）
python -m venv D:/llm-work/venv
D:/llm-work/venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 0. 验证环境 + 全链路
cd "D:/llm-work/handmade-llm/code"
python smoke_test.py

# 1. 分词器 demo
python bpe.py

# 2. 准备数据（字符级，最快）
python prepare_data.py --mode char --input D:/llm-work/corpus/tinyshakespeare.txt --out D:/llm-work/data/shake

# 3. 准备数据（BPE，实战）
python prepare_data.py --mode hf --input D:/llm-work/corpus/tinystories.txt --out D:/llm-work/data/tiny --vocab_size 8192

# 4. 训练（124M 级，8GB 上要挂机）
python train.py --data_dir D:/llm-work/data/tiny --out_dir D:/llm-work/out \
  --n_layer 6 --n_head 6 --n_embd 384 --block_size 512 \
  --batch_size 16 --grad_accum 8 --max_iters 20000 \
  --learning_rate 6e-4 --eval_interval 500 --sample_interval 2000

# 5. 断点续训
python train.py --data_dir D:/llm-work/data/tiny --out_dir D:/llm-work/out --resume D:/llm-work/out/ckpt.pt

# 6. 推理
python sample.py --ckpt D:/llm-work/out/ckpt_final.pt --prompt "Once upon a time" --temperature 0.8 --top_k 50 --repetition_penalty 1.1
```

## 附录 B：一句话总结每一章

1. **算力** `C≈6ND`，**时间** `C/(峰值×MFU)`，**显存** 16 字节/参数
2. **sm_120 必须用 cu128 的 torch**，否则 `no kernel image`
3. **数据和 checkpoint 放无空格路径**
4. **TinyStories 是最好的第一站**，反馈快你才坚持得下来
5. **中文第一站选窄领域**（古诗/小说），别一上来啃通用中文
6. **BPE 三句话**：字节起步、合并最高频对、重复 N 次
7. **学 Llama 不学 GPT-2**：RMSNorm + RoPE + GQA + SwiGLU
8. **`loss / grad_accum` 别忘**，warmup 和 cosine 不是可选项
9. **小模型 SFT 学不会指令跟随**，那是 1B+ 的能力；小模型该专精
10. **KV cache 是推理性能的全部**，O(T³) → O(T²)
11. **PPL 只能和自己比**，tokenizer 不同就没有可比性
12. **验收标准比进度重要**——每关做完再进下一关

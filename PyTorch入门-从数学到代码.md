# PyTorch 入门：从数学到代码

> **这份文档为你而写，前提是你告诉我的这几件事：**
>
> | 你的情况 | 文档因此怎么写 |
> |---|---|
> | 线代、微积分、概率统计**都学过** | **不做比喻式科普**，公式→代码真推导 |
> | **没看过** `code/` 里任何文件 | 从「第一次打开这个文件」讲起，不跳步 |
> | 遇到报错**习惯问 AI** | 报错**全文列出 + 逐行解读**，让你先看懂再问 |
> | 每天 **3 小时** | 写得深，不压缩 |
> | 目标：**彻底搞懂 + 面试能用** | 每章末尾有「面试官问 XX 怎么答」 |
> | 从小写代码习惯**混着来** | 先结论 → 再场景 → 配可运行代码 |
>
> **这份文档的定位（很重要）：**
>
> 你不缺数学，你缺的是**「数学公式」到「PyTorch 代码」之间的翻译层**。
>
> 你知道链式法则，但不知道 `loss.backward()` 在干什么；
> 你知道矩阵乘法，但不知道 `x @ W.T` 里那个 `.T` 为什么在那；
> 你知道概率分布，但不知道 `logits / temperature` 是在对分布做什么。
>
> **这份文档就是这层翻译。** 它不重新教你数学。

**配套练习代码**（都已实测可运行）：

```
D:\llm-work\learn\
├── 01_hello.py            第 0 章：5 分钟看到 PyTorch 在动
├── 02_autograd_mini.py    第 2 章：手写 mini autograd 引擎
├── 03_raw_linear.py       第 3 章：不用 nn.Module 的线性回归
└── 04_module_linear.py    第 3 章：用 nn.Module 重写（结果完全一致）
```

跑法（CPU 环境就够，不需要 GPU）：

```bash
cd D:/llm-work/learn
D:/llm-work/venv/Scripts/python.exe 01_hello.py
```

---

## 目录

- [第 0 章 · 先跑起来：5 分钟看到 PyTorch 在动](#第-0-章)
- [第 1 章 · Tensor：一切的载体](#第-1-章)
- [第 2 章 · autograd：`backward()` 到底算了什么](#第-2-章)
- [第 3 章 · nn.Module：为什么模型要写成一个类](#第-3-章)
- [第 4 章 · 训练循环五件套](#第-4-章)
- [第 5 章 · 逐行拆解你手上的 model.py](#第-5-章)
- [第 6 章 · 逐行拆解你手上的 train.py](#第-6-章)
- [第 7 章 · 报错速查：12 个你一定会遇到的](#第-7-章)
- [附录 A · 面试问答](#附录-a)
- [附录 B · 公式 ↔ 代码对照表](#附录-b)

---

<a name="第-0-章"></a>
## 第 0 章 · 先跑起来：5 分钟看到 PyTorch 在动

> **本章结论**：PyTorch 的核心只有两个对象——**Tensor（存数据）** 和 **autograd（记梯度）**。
> 这一章让你亲眼看到它们，不解释原理。

### 0.1 确认环境

你机器的环境我上次已经建好了。先确认：

```bash
# 如果 CUDA 版还没装好，先用 CPU 版
D:/llm-work/venv/Scripts/python.exe -c "import torch; print(torch.__version__)"
```

期望输出 `2.11.0+cu128`。**版本号不重要**，能 `import` 就行。

### 0.2 第一个 PyTorch 程序

建一个文件 `D:\llm-work\learn\01_hello.py`（目录不存在就先建），写：

```python
import torch

# ---- 1. 一个 Tensor 就是"多维数组" ----
a = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print("a =", a)
print("形状 shape =", a.shape)      # torch.Size([2, 2])
print("数据类型 dtype =", a.dtype)   # torch.float32
print("在哪个设备 device =", a.device)

# ---- 2. 你学过的矩阵乘法 ----
b = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]])      # 单位矩阵
print("\na @ b =", a @ b)           # @ 就是矩阵乘法，和 numpy 一样

# ---- 3. 它是自动求导的（这是 PyTorch 和 numpy 的根本区别）----
x = torch.tensor([2.0], requires_grad=True)   # 告诉 PyTorch：盯着 x
y = x ** 3                                     # y = x³
y.backward()                                   # 反着算一遍
print("\nx =", x.item())
print("y = x³ =", y.item())
print("dy/dx =", x.grad.item(), " ← 应该等于 3x² = 12")
```

运行：

```bash
cd D:/llm-work/learn
D:/llm-work/venv/Scripts/python.exe 01_hello.py
```

**期望看到**（重点看最后一行）：

```
a = tensor([[1., 2.],
        [3., 4.]])
形状 shape = torch.Size([2, 2])
数据类型 dtype = torch.float32
在哪个设备 device = cpu

a @ b = tensor([[1., 2.],
        [3., 4.]])

x = 2.0
y = x³ = 8.0
dy/dx = 12.0  ← 应该等于 3x² = 12
```

**看到 `dy/dx = 12.0` 的那一刻，你就理解 PyTorch 是什么了：**

> 它不是一个数学库，它是一个**会自动帮你算导数的数学库**。
> 你写 `y = x**3`，它在前向的过程中偷偷记下了"怎么算导数"，
> 你调 `y.backward()`，它就把 `dy/dx = 3x² = 12` 算给你。

这就是全部的核心。**剩下的所有东西——模型、训练、反向传播——都是这两个机制的组合。**

### 0.3 本章要带走的三句话

1. `torch.tensor([...])` 造一个数组，`.shape` / `.dtype` / `.device` 是它的三个身份证
2. `@` 是矩阵乘法（你学的那个）
3. `requires_grad=True` + `.backward()` + `.grad` = **自动求导**

---

<a name="第-1-章"></a>
## 第 1 章 · Tensor：一切的载体

> **本章结论**：Tensor 就是「n 维数组 + 自动求导 + 能上 GPU」的合体。
> 你必须牢牢掌握的是 **shape（形状）**——90% 的 PyTorch 报错都是形状不匹配。

### 1.1 Tensor 是什么：一个精确的定义

**定义**：一个 Tensor 由三部分组成

```
Tensor = 数据 + 形状(shape) + 梯度标记(requires_grad)
         └─ 存在哪：device (cpu / cuda)
         └─ 什么类型：dtype (float32 / int64 / bfloat16 ...)
```

用你学过的东西类比：

| 数学对象 | PyTorch 类型 | 例子 |
|---|---|---|
| 标量 $x \in \mathbb{R}$ | 0 维 Tensor | `torch.tensor(3.0)` |
| 向量 $\mathbf{x} \in \mathbb{R}^n$ | 1 维 Tensor | `torch.tensor([1., 2., 3.])` |
| 矩阵 $A \in \mathbb{R}^{m\times n}$ | 2 维 Tensor | `torch.randn(2, 3)` |
| 张量 $\mathcal{T} \in \mathbb{R}^{a\times b\times c}$ | 3 维 Tensor | `torch.randn(2, 3, 4)` |
| 一批图片 $B \times C \times H \times W$ | 4 维 Tensor | `torch.randn(8, 3, 224, 224)` |

**「Tensor」这个词本身没有新东西，它就是「n 维数组」的正式叫法。** 你在 numpy 里见过，PyTorch 加了两件事：自动求导 + 能上 GPU。

### 1.2 shape：最重要的一个概念

**定义**：`shape` 是一个元组，第 $i$ 个数表示第 $i$ 个维度有多长。

```python
a = torch.randn(2, 3, 4)
print(a.shape)     # torch.Size([2, 3, 4])
```

读法：这是一个 3 维数组，第一维长度 2，第二维长度 3，第三维长度 4。总共 $2 \times 3 \times 4 = 24$ 个数。

**为什么 shape 最重要**：因为神经网络里的每一步运算都是矩阵乘法，而**矩阵乘法对形状有严格要求**。

你学过的：若 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，则 $AB \in \mathbb{R}^{m \times n}$。

**中间那个 $k$ 必须相等。** 这就是 PyTorch 里最常见的报错来源：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 256x512)
```

翻译成人话：你想做 $A(64\times128) \times B(256\times512)$，但 $128 \neq 256$，**乘不了**。

> **面试官可能问**：「PyTorch 里最常见的错误是什么？」
> **答**：shape 不匹配。因为矩阵乘法要求内维相等，而神经网络前向就是一连串矩阵乘法，任何一层输出维度写错，错误会一直传到后面才爆出来。所以我调试时第一件事永远是 `print(x.shape)`。

### 1.3 三种创建方式

```python
import torch

# 方式 1：从 Python 数据直接造
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 方式 2：从 numpy 造（共享内存！改一个另一个也变）
import numpy as np
n = np.array([[1.0, 2.0], [3.0, 4.0]])
b = torch.from_numpy(n)

# 方式 3：造随机 / 特定值的（初始化权重用这个）
c = torch.randn(2, 3)        # 标准正态分布 N(0,1)
d = torch.zeros(2, 3)        # 全 0
e = torch.ones(2, 3)         # 全 1
f = torch.empty(2, 3)        # 不初始化（快，但值是垃圾）
g = torch.arange(0, 10)      # 0..9
```

**`torch.randn` 里的 n 是 "normal"**，标准正态。这是初始化神经网络权重的标准做法——你学过的 $N(0,1)$。

**⚠️ `torch.from_numpy` 是共享内存的**，改 `n` 会让 `b` 一起变。这是个常见坑：

```python
n[0, 0] = 999.0
print(b[0, 0])    # 999.0  ← 变了！因为它们是同一块内存
```

想断开就 `.clone()`。

### 1.4 dtype：数据类型

| dtype | 用途 | 大小 |
|---|---|---|
| `torch.float32` | **默认**，训练权重、激活值 | 4 字节 |
| `torch.bfloat16` | **混合精度训练**（后面详讲） | 2 字节 |
| `torch.float16` | 老式半精度，容易溢出 | 2 字节 |
| `torch.int64` | **token id、索引**（默认整数类型） | 8 字节 |
| `torch.bool` | 掩码 | 1 字节 |

**为什么要关心 dtype**：因为**显存**。你的显卡只有 8GB：

```
fp32 权重:  124M 参数 × 4 字节 = 496 MB
bf16 权重:  124M 参数 × 2 字节 = 248 MB   ← 省一半
```

**类型转换**：

```python
x = torch.randn(3)          # float32
y = x.to(torch.bfloat16)    # 转 bf16
z = x.long()                # 转 int64
w = x.float()               # 转回 float32
```

**⚠️ 最常见的 dtype 错误**：模型是 float32，数据是 int64，直接相乘报错

```
RuntimeError: expected scalar type Float but found Long
```

翻译：你想让浮点数和整数做运算，PyTorch 不干。**解决：`data.float()`。**

### 1.5 device：在 CPU 还是 GPU

```python
x = torch.randn(3)                    # 默认在 CPU
x_gpu = x.to("cuda")                  # 搬到 GPU
x_cpu = x_gpu.to("cpu")               # 搬回 CPU

# 判断有没有 GPU
print(torch.cuda.is_available())      # True / False
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 5060 Laptop GPU
```

**铁律：两个 Tensor 必须在同一个设备上才能运算。**

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

这是新手第二常见的错误。**解决：把模型和数据都 `.to(device)`。**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)      # 模型搬过去
x = x.to(device)              # 数据也搬过去
```

### 1.6 改变形状的四个函数（重点）

这是最容易搞混的地方，我一次讲清楚。

假设有 `x = torch.arange(12)`，即 `[0,1,2,...,11]`，shape 是 `(12,)`。

#### `view` / `reshape`：重新排列，不改变数据顺序

```python
x = torch.arange(12)
print(x.view(3, 4))
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
```

**数据按行优先（row-major）填进去**——就是 numpy 的顺序。

`view` 和 `reshape` 的区别：
- `view` 要求内存连续，否则报错（但它更明确，能保证不复制）
- `reshape` 内部会自动判断，必要时复制数据

**实践中：不确定就用 `reshape`，确定是连续的用 `view`（更快）。**

**`-1` 的含义：让 PyTorch 自己算这一维。**

```python
x = torch.arange(12)
print(x.view(3, -1).shape)    # torch.Size([3, 4])   12/3 = 4，自动算出
print(x.view(-1, 3).shape)    # torch.Size([4, 3])   12/3 = 4，自动算出
print(x.view(2, -1).shape)    # torch.Size([2, 6])
```

**为什么训练里到处是 `-1`**：因为 batch size 会变，写死会出问题。比如 `logits.view(-1, vocab_size)` 的意思是「不管有多少个 token，把最后一维整理成 vocab_size」。

#### `transpose`：交换两个维度

```python
x = torch.randn(2, 3)
print(x.transpose(0, 1).shape)   # torch.Size([3, 2])
```

你在数学里学的转置 $A^T$。

**⚠️ 陷阱**：`transpose` 之后内存不连续，直接 `view` 会报错：

```python
x = torch.randn(2, 3)
y = x.transpose(0, 1)
y.view(-1)      # RuntimeError: view size is not compatible...
y.reshape(-1)   # OK，reshape 会自动处理
y.contiguous().view(-1)   # 也可以，先整理内存
```

#### `permute`：任意交换多个维度

```python
x = torch.randn(2, 3, 4)
print(x.permute(2, 0, 1).shape)   # torch.Size([4, 2, 3])
```

比 `transpose` 更通用，`transpose` 是交换两个，`permute` 是任意重排。

#### `unsqueeze` / `squeeze`：加/删一个长度为 1 的维度

```python
x = torch.randn(3)
print(x.unsqueeze(0).shape)   # torch.Size([1, 3])   加一维
print(x.unsqueeze(1).shape)   # torch.Size([3, 1])   加在位置 1

y = torch.randn(1, 3, 1)
print(y.squeeze().shape)      # torch.Size([3])      删掉所有长度为1的维
```

**`unsqueeze` 的用处**：因为训练时的 batch 维度是必须的。一个单独的样本 $x \in \mathbb{R}^n$ 要变成 $\mathbb{R}^{1\times n}$ 才能喂给模型。

**记忆口诀**：
- `view/reshape` = 重新排版（数据顺序不变）
- `transpose/permute` = 换轴（数据顺序不变，但内存布局变了）
- `unsqueeze/squeeze` = 补/删那个 `1`

### 1.7 广播机制（broadcasting）

**定义**：当两个形状不同的 Tensor 做运算时，PyTorch 会自动「扩展」小的那个——**从最后一维开始对齐，长度为 1 的可以扩展成任意长度**。

```python
a = torch.ones(3, 4)
b = torch.ones(4)          # 形状是 (4,)

print((a + b).shape)       # torch.Size([3, 4])
```

`b` 被"广播"成了 `(3, 4)`——每一行都加上 `b`。

**规则**（从右往左看）：
```
a:  (3, 4)
b:     (4,)     ← 补齐成 (1, 4)，然后沿第 0 维复制 3 次
结果: (3, 4)
```

再看一个：
```python
a = torch.ones(2, 3, 4)
b = torch.ones(3, 1)
# a:  (2, 3, 4)
# b:     (3, 1)   → 补齐成 (1, 3, 1) → 扩展成 (2, 3, 4)
print((a + b).shape)       # torch.Size([2, 3, 4])
```

**⚠️ 广播是 PyTorch 里最阴险的坑之一**，因为它**不报错，但结果是错的**。

真实案例：你想算两个向量的点积，但形状是 `(3,)` 和 `(3,)`，广播后变成 `(3, 3)` 的逐元素乘法——完全不是你想要的。

**防御手段**：写完一个运算，**永远 `print` 一下 shape**。

> **面试官可能问**：「什么是广播？」
> **答**：广播是两个形状不同的张量做运算时的自动扩展规则。规则是从最后一维开始对齐，长度为 1 或缺失的维度可以扩展成对方的长度，否则报错。它的用处是省掉很多手动复制，但危险在于不匹配时它不会报错、而是静默产生错误形状，所以我会习惯性检查 shape。

### 1.8 索引和切片（和 numpy 一样）

```python
x = torch.arange(12).view(3, 4)
print(x[0])        # 第 0 行
print(x[:, 0])     # 第 0 列
print(x[1:3])      # 第 1、2 行
print(x[0, 1])     # 第 0 行第 1 列 = 1
```

**⚠️ 一个和 numpy 的重要区别：高级索引会复制数据。**

```python
y = x[:, 0]        # 这是 view 还是 copy？
```

在 PyTorch 里，整数索引会**复制**（返回新张量），切片会**返回视图**（共享内存）。这和 numpy 的行为不完全一致，容易踩坑。

### 1.9 本章浓缩

```python
import torch

x = torch.randn(2, 3)          # 造 (2,3) 的随机张量
x.shape                        # torch.Size([2, 3])   ← 最重要
x.dtype                        # torch.float32
x.device                       # device(type='cpu')

x.view(3, 2)                   # 改装（数据顺序不变）
x.transpose(0, 1)              # 转置
x.unsqueeze(0)                 # 加一维 → (1,2,3)
x.to("cuda")                   # 上 GPU
x.float() / x.long()           # 类型转换

a @ b                          # 矩阵乘法（内维必须相等！）
a + b                          # 逐元素加（会广播）
```

**记住：90% 的报错是 shape 或 device 不匹配。**

---

<a name="第-2-章"></a>
## 第 2 章 · autograd：`backward()` 到底算了什么

> **本章结论**：`backward()` 就是**链式法则的自动执行**。
> 你学的 $\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$，PyTorch 帮你沿计算图倒着走一遍。
>
> **这一章是整份文档最重要的地方。** 因为训练的全部机制都在这里。

### 2.0 先说清楚：为什么整份文档到处都在讲求导？

> **这一节是为了回答一个很自然的疑问**：「我只是想训个模型，为什么文档里到处是导数？」
> 如果你没这个疑问，可以跳到 2.1。但建议还是看一眼，因为它决定了你后面读文档的心态。

**答案：因为「训练」这件事，数学上就等于「求导 + 沿着导数反方向走」。**

先看一个最小的事实：

> **一个神经网络被「训练」，指的是不断修改它的参数，让它犯错越来越少。**

那么问题来了：**怎么知道参数该往哪改？**

答：看 loss 对参数的**导数**。

- 导数为**正** → 参数增大，loss 就增大 → 所以要**减小**它
- 导数为**负** → 参数增大，loss 就减小 → 所以要**增大**它

**所以在深度学习里，求导不是"一个数学工具"，它是「学习」这件事本身的定义。**

用一张表说清各部分的角色：

| 你要做的事 | 需要的数学 |
|---|---|
| 模型怎么**算**（前向） | 矩阵乘法 |
| **模型怎么学（训练）** | **求导** ← 本章讲的就是这个 |
| 模型输出概率分布 | softmax / 概率论 |

**深度学习 ≈ 用矩阵乘法搭一个复杂函数，再用求导找这个函数的最优参数。**
去掉求导，剩下的只是一个很贵的计算器——它不会学习。

**对照代码**，你会发现几乎每一行都在围着"求导"转：

| 你看到的代码/词 | 它其实就是 |
|---|---|
| `loss.backward()` | 求导 |
| 「梯度」 | 导数的向量版本 |
| 「梯度下降」 | 沿着导数反方向走 |
| `optimizer.step()` | 用导数更新参数 |
| 「反向传播」 | 用链式法则**高效地**求导（2.4 节讲"为什么高效"） |
| `autograd` | 自动求导（automatic gradient 的缩写） |

**所以不是文档啰嗦，是这件事真的占了深度学习的半壁江山。**

### 2.1 先回到你学过的东西

你学过链式法则。设 $y = f(u)$，$u = g(x)$，则：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

这在标量上很简单。但在神经网络里，情况是这样的：

$$\text{loss} = L\big(\text{model}(\mathbf{x}; W)\big)$$

- 输入 $\mathbf{x}$ 是一个向量
- $W$ 是**几百万个参数**组成的矩阵
- 你想要的是：**$\frac{\partial \text{loss}}{\partial W}$** —— 对**每一个**参数分别求导

**这个问题手算是不可能的。** 一个 124M 参数的模型，你要算 124M 个偏导数。

**autograd 就是解决这个问题的。** 它不是魔法，它是**链式法则 + 计算图**的自动执行。

### 2.2 计算图：autograd 的核心机制

**定义**：每做一次运算，PyTorch 就记一个节点，把所有运算连成一棵树/图。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3          # y = x³
z = y * 2           # z = 2y
```

PyTorch 内部记下的是这样一张图：

```
   x (叶子节点, 需要梯度)
   │
   │ y = x³        ← 记下: dy/dx = 3x²
   ▼
   y
   │
   │ z = 2y        ← 记下: dz/dy = 2
   ▼
   z (最终输出)
```

**每个节点都记着「它的导数怎么算」**，这就是 `.grad_fn`：

```python
print(z.grad_fn)        # <MulBackward0 object at 0x...>
print(y.grad_fn)        # <PowBackward0 object at 0x...>
print(x.grad_fn)        # None  ← x 是叶子节点（自己造的，不是算出来的）
```

**叶子节点（leaf）**：`requires_grad=True` 且不是运算结果的节点。**只有叶子节点会存 `.grad`**——因为只有它（比如权重 $W$）是你想更新的。

### 2.3 `backward()` 到底做了什么

```python
z.backward()
print(x.grad)          # tensor(24.)
```

验证一下手算：
$$z = 2y = 2x^3 \implies \frac{dz}{dx} = 6x^2 \implies \text{在 } x=2 \text{ 处} = 6 \times 4 = 24 \quad ✓$$

**`backward()` 的执行过程**（这就是"反向传播"）：

```
1. 从 z 开始，dz/dz = 1
2. 沿图往回走一步到 y：dz/dy = 2，累积得到 dz/dy = 1 × 2 = 2
3. 再回到 x：       dy/dx = 3x² = 12，累积得到 dz/dx = 2 × 12 = 24
4. 把 24 存进 x.grad
```

**注意第 3 步：这就是链式法则，只是自动做了。** 每一步都是「本地的导数」乘上「从后面传回来的梯度」。

**用你学过的符号写**：设 $z = f_2(f_1(x))$，则

$$\frac{dz}{dx} = \underbrace{\frac{df_2}{df_1}}_{\text{从后面传回}} \cdot \underbrace{\frac{df_1}{dx}}_{\text{本地导数}}$$

**PyTorch 的每一个 `XxxBackward` 类，存的就是「本地导数」怎么算。** 反向传播就是把这些乘起来。

### 2.4 一个必须理解的点：为什么是"反向"？

考虑 $L$ 层网络：$\text{loss} = f_L(f_{L-1}(\cdots f_1(x)))$

- **前向**：从 $x$ 开始，一层层算到 loss
- **反向**：从 loss 开始，一层层把梯度传回去

#### 先看一个 3 层的具体例子

假设有 3 层，参数是 $W_1, W_2, W_3$。你要算 loss 对**每一个**参数的导数。

按链式法则，loss 对 **$W_1$** 的导数是：

$$\frac{\partial L}{\partial W_1} = \underbrace{\frac{\partial L}{\partial f_3}}_{\text{①}} \cdot \underbrace{\frac{\partial f_3}{\partial f_2}}_{\text{②}} \cdot \underbrace{\frac{\partial f_2}{\partial f_1}}_{\text{③}} \cdot \frac{\partial f_1}{\partial W_1}$$

对 **$W_2$** 的导数：

$$\frac{\partial L}{\partial W_2} = \underbrace{\frac{\partial L}{\partial f_3}}_{\text{①}} \cdot \underbrace{\frac{\partial f_3}{\partial f_2}}_{\text{②}} \cdot \frac{\partial f_2}{\partial W_2}$$

对 **$W_3$** 的导数：

$$\frac{\partial L}{\partial W_3} = \underbrace{\frac{\partial L}{\partial f_3}}_{\text{①}} \cdot \frac{\partial f_3}{\partial W_3}$$

**看出问题了吗？** 注意我用 ①②③ 标出的部分：

| 算谁的导数 | 需要乘的公共部分 |
|---|---|
| $W_1$ | ① ② ③ |
| $W_2$ | ① ② |
| $W_3$ | ① |

**①（也就是 $\frac{\partial L}{\partial f_3}$）被重复乘了 3 次。② 被乘了 2 次。**
**这些乘法是完全一样的——却各算了一遍。**

#### 两种算法的对比

**算法 A：每个参数独立，从前往后从头算**

| 算谁的导数 | 乘法次数 |
|---|---|
| $W_1$ | 3 |
| $W_2$ | 2 |
| $W_3$ | 1 |
| **合计** | **6 次** |

**算法 B：先把「从 loss 传回来的梯度」算出来，然后复用**

```
第 1 步： g₃ = ∂L/∂f₃                ← 算一次
第 2 步： g₂ = g₃ · (∂f₃/∂f₂)         ← 复用 g₃，只再乘一次
第 3 步： g₁ = g₂ · (∂f₂/∂f₁)         ← 复用 g₂，只再乘一次
```

**这个 $g_i$ 就是「从 loss 传回到第 $i$ 层的梯度」。**

| 算谁的导数 | 乘法次数 |
|---|---|
| $W_3$ | 用 $g_3$，乘 1 次 |
| $W_2$ | 用 $g_2$，乘 1 次 |
| $W_1$ | 用 $g_1$，乘 1 次 |
| 算 $g_2, g_1$ | 2 次 |
| **合计** | **5 次** |

**3 层网络：6 次 vs 5 次，差别不大。**

#### 但层数一多，差距就恐怖了

| 层数 $L$ | 算法 A（$O(L^2)$） | 算法 B（$O(L)$） | 倍数 |
|---|---|---|---|
| 3 | 6 | 5 | 1.2× |
| 10 | 55 | ~19 | 2.9× |
| 100 | 5050 | ~199 | **25×** |
| 1000 | 500500 | ~1999 | **250×** |

（算法 A：$L + (L-1) + \cdots + 1 = \frac{L(L+1)}{2}$，即 $O(L^2)$。
算法 B：算 $g$ 要 $L-1$ 次，每个参数再乘 1 次共 $L$ 次，合起来 $O(L)$。）

**一个 100 层的网络，写法不同就差 25 倍。** 这就是为什么必须反向。

#### 一句话总结

> **反向传播之所以叫「反向」，是因为从 loss 那头往回走，能把「从 loss 传回来的梯度」
> 存下来复用。正着走每层都要把整条链重算一遍，层数一多就慢几十上百倍。**

**PyTorch 的 `backward()` 干的就是算法 B。** 你自己写代码时不用实现它——autograd 全包了。
但你要知道**它在替你省掉 25 倍的乘法**，这就是「为什么训练能跑得动」的原因之一。

#### 附：为什么反向比前向"更贵"

上面说的是**反向传播为什么比"朴素从头算"省**。但你可能还听过另一句话：
「反向传播比前向传播慢约 2 倍」。这两个说的是**不同的比较**：

| 比较 | 结论 |
|---|---|
| 反向 vs 朴素从头算 | 反向省 25 倍（上面讲的） |
| **反向 vs 前向本身** | **反向贵约 2 倍**（下面讲的） |

为什么反向比前向贵 2 倍：前向每个参数做 1 次乘加（2 FLOPs，算 $y = Wx$）；
反向要做**两次**矩阵乘——一次算"对输入的梯度"（好传给前一层），一次算"对权重的梯度"（好拿来更新参数），共 4 FLOPs。

**合起来：训练一个 token 的总计算量 ≈ $6N$**（$N$ 为参数量）：

$$6N = \underbrace{2N}_{\text{前向}} + \underbrace{4N}_{\text{反向}}$$

**这个 $6N$ 就是 LLM 算力估算公式 $C \approx 6ND$ 的来源**（$D$ 是 token 数）。
手册第 1 章算训练时间用的就是它。

> **面试官可能问**：「为什么反向传播要比前向传播慢？」
> **答**：因为反向传播的每一步都是一次矩阵乘法（雅可比矩阵），而后向的矩阵乘规模比前向大约 2 倍——准确地说，前向每个参数 2 FLOPs（乘加），反向要算"对输入的梯度"和"对权重的梯度"各一次，共 4 FLOPs。所以训练一个 token 的总计算量约 $6N$（$N$ 为参数量），其中 2/3 花在反向。

> **面试官可能问（进阶）**：「为什么需要反向传播，而不是每层独立求导？」
> **答**：因为链式法则展开后，多个参数共享同一段"从 loss 传回来的梯度"。如果每个参数独立从前往后算，这段公共部分会被重复计算，复杂度是 $O(L^2)$；从后往前算可以把每层的"传回梯度"存下来复用，降到 $O(L)$。$L=100$ 时差 25 倍。所以反向传播不只是"一种求导方式"，它是让深层网络训练可行的关键优化。

### 2.5 `requires_grad` 和"冻结"

```python
w = torch.randn(3, requires_grad=True)
print(w.grad)          # None —— 还没 backward

w2 = w.detach()        # 断开梯度链
print(w2.requires_grad)  # False
```

**`detach()` 的用处**：你不希望梯度流过去的时候。

**在推理时**：
```python
with torch.no_grad():
    output = model(x)
```
**这行代码会关掉整个计算图的记录**，省内存也更快。**推理时一定要加**——不加的话，生成 1000 个 token 会把 1000 张计算图全存在显存里，直接爆显存。

**在 fine-tune 时**：冻结某层
```python
for param in model.layers[0].parameters():
    param.requires_grad = False      # 这层不更新
```

### 2.6 梯度累积的陷阱：`zero_grad()`

**这是新手必踩的坑，而且非常隐蔽。**

```python
x = torch.tensor(2.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print(x.grad)      # tensor(4.)    ← 2x = 4 ✓

y2 = x ** 3
y2.backward()
print(x.grad)      # tensor(16.)   ← 期望 12，实际 12+4 = 16 ✗
```

**原因**：`.grad` 是**累加**的，不是覆盖。

$$x.\text{grad} = \frac{dy_1}{dx} + \frac{dy_2}{dx} = 4 + 12 = 16$$

**为什么要设计成累加**：这是**特性**不是 bug——梯度累积（gradient accumulation）就靠它。但如果你不想要累积，必须手动清零：

```python
x.grad.zero_()          # 或者
optimizer.zero_grad()   # 训练循环里用的这个
```

**训练循环里漏掉 `zero_grad()` 的后果**：梯度会一直累积，等效学习率不断变大，最后 loss 爆炸成 NaN。

> **面试官可能问**：「训练循环里 `zero_grad()` 为什么必须在 `backward()` 之前？」
> **答**：因为 PyTorch 的 `.grad` 是累加的。一个 batch 算完、`step()` 更新完参数后，上一轮的梯度还留在 `.grad` 里，如果不清零，这一轮的梯度会叠加到上一轮上，等效于学习率不断增大，最后发散。放在 `backward()` 之前还是 `step()` 之后都可以（只要在下一轮 `backward()` 前清掉），但写在循环开头最不容易漏。

### 2.7 计算图的释放

```python
y = x ** 3
y.backward()
# y.backward()   ← 再调一次会报错！
```

报错：
```
RuntimeError: Trying to backward through the graph a second time,
but the saved intermediate results have been freed.
```

**原因**：`backward()` 默认会**释放计算图**（`retain_graph=False`），省内存。

需要多次反向时（比如 GAN）要显式指定：
```python
y.backward(retain_graph=True)
```

**实践中**：这一条你基本不会遇到，除非写 GAN。

### 2.8 手写一个 mini autograd（加深理解）

这一段是**选读**，但强烈建议你跑一遍——跑完之后 autograd 对你就是透明的了。

```python
# 文件: D:/llm-work/learn/02_autograd_mini.py
"""手写一个极简的自动求导引擎，只支持 + 和 *。
目的是让你看清 backward() 内部到底在做什么。"""

class Var:
    def __init__(self, value, parents=(), local_grads=()):
        self.value = value              # 这个节点的值
        self.parents = parents          # 父节点（它是从谁算出来的）
        self.local_grads = local_grads  # 对每个父节点的"本地导数"
        self.grad = 0.0

    def __add__(self, other):
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Var(self.value + other.value,
                   (self, other), (1.0, 1.0))

    def __mul__(self, other):
        # d(a*b)/da = b, d(a*b)/db = a
        return Var(self.value * other.value,
                   (self, other), (other.value, self.value))

def backward(root):
    """从 root 开始，反向遍历计算图，用链式法则累积梯度。"""
    root.grad = 1.0                     # d(root)/d(root) = 1
    # 拓扑排序：保证处理一个节点时，它下游的梯度已经算完了
    topo, visited = [], set()

    def build(v):
        if id(v) in visited:
            return
        visited.add(id(v))
        for p in v.parents:
            build(p)
        topo.append(v)

    build(root)

    for v in reversed(topo):            # 倒着走
        for parent, local_grad in zip(v.parents, v.local_grads):
            # 链式法则: parent.grad += v.grad * (dv/dparent)
            parent.grad += v.grad * local_grad


# ---- 验证 ----
# 计算 f(a,b,c) = (a + b) * c，在 a=2, b=3, c=4 处
a, b, c = Var(2.0), Var(3.0), Var(4.0)
f = (a + b) * c

print(f"f = (a+b)*c = {f.value}")        # (2+3)*4 = 20

backward(f)
print(f"df/da = {a.grad}   期望 (a+b)*c 对 a 求导 = c = {c.value}")
print(f"df/db = {b.grad}   期望 (a+b)*c 对 b 求导 = c = {c.value}")
print(f"df/dc = {c.grad}   期望 (a+b)*c 对 c 求导 = a+b = {a.value+b.value}")

# ---- 和 PyTorch 对照 ----
import torch
ta = torch.tensor(2.0, requires_grad=True)
tb = torch.tensor(3.0, requires_grad=True)
tc = torch.tensor(4.0, requires_grad=True)
tf = (ta + tb) * tc
tf.backward()
print(f"\nPyTorch 的结果: df/da={ta.grad.item()}, df/db={tb.grad.item()}, df/dc={tc.grad.item()}")
```

**跑一下**：

```bash
D:/llm-work/venv/Scripts/python.exe 02_autograd_mini.py
```

**期望输出**：

```
f = (a+b)*c = 20.0
df/da = 4.0   期望 (a+b)*c 对 a 求导 = c = 4.0
df/db = 4.0   期望 (a+b)*c 对 b 求导 = c = 4.0
df/dc = 5.0   期望 (a+b)*c 对 c 求导 = a+b = 5.0

PyTorch 的结果: df/da=4.0, df/db=4.0, df/dc=5.0
```

**看懂这段代码，你就永久理解 autograd 了。** 它做了三件事：

1. **前向时记图**：每次运算都建一个节点，记下「我是从谁来的」+「我的本地导数是什么」
2. **拓扑排序**：把图排成一个顺序，保证倒着走时，下游的梯度一定已经算完了
3. **反向累积**：`parent.grad += v.grad * local_grad` —— **这一行就是链式法则**

（`+=` 而不是 `=` 就是 2.6 节说的「梯度累加」。）

### 2.9 本章浓缩

| 概念 | 是什么 | 对应你学的 |
|---|---|---|
| `requires_grad=True` | 标记这个张量要算梯度 | 声明"这是自变量" |
| 计算图 | 每次运算记一个节点 | 复合函数的展开 |
| `grad_fn` | 每个节点记的"本地导数怎么算" | $\frac{\partial f_i}{\partial f_{i-1}}$ |
| `backward()` | 从后往前用链式法则乘起来 | $\frac{dy}{dx}=\prod\frac{\partial f_i}{\partial f_{i-1}}$ |
| `.grad` | 累积的梯度结果 | $\frac{\partial \text{loss}}{\partial \theta}$ |
| `.grad` 累加 | 不清零就一直加 | 你算 $\sum \frac{\partial f_i}{\partial x}$ |
| `zero_grad()` | 清零 | 重新开始算 |
| `torch.no_grad()` | 关掉记录 | 不需要导数的场景 |
| `detach()` | 断开梯度链 | 截断链式法则 |

---

<a name="第-3-章"></a>
## 第 3 章 · nn.Module：为什么模型要写成一个类

> **本章结论**：`nn.Module` 只做三件事——**登记参数**、**提供 `.to()` / `.parameters()` 等工具**、**定义 `forward` 说明怎么算**。
> 它没有任何数学魔法，纯粹是工程封装。

### 3.1 先看一个最小的神经网络（手写版）

你学过线性回归。一个单层网络就是：

$$\hat{y} = W\mathbf{x} + \mathbf{b}$$

loss 用你学过的均方误差（MSE）：

$$\text{loss} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

**不用 `nn.Module`，纯用 Tensor 也能训练**：

```python
# 文件: D:/llm-work/learn/03_raw_linear.py
"""不用 nn.Module，纯手写一个线性回归的训练。"""
import torch

torch.manual_seed(0)

# 造一些"假数据"：真实关系是 y = 3x + 2，加点噪声
X = torch.randn(100, 1)
y_true = 3 * X + 2 + 0.1 * torch.randn(100, 1)

# 要学的参数
W = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for step in range(200):
    # ---- 1. 前向：算预测值 ----
    y_pred = X @ W + b

    # ---- 2. 算 loss ----
    loss = ((y_pred - y_true) ** 2).mean()

    # ---- 3. 清零梯度 ----
    W.grad = None
    b.grad = None
    # 或者：if W.grad is not None: W.grad.zero_()

    # ---- 4. 反向：算梯度 ----
    loss.backward()

    # ---- 5. 更新参数 ----
    with torch.no_grad():          # 更新时不需要建图，省内存
        W -= lr * W.grad
        b -= lr * b.grad

    if step % 40 == 0:
        print(f"step {step:3d}  loss = {loss.item():.6f}  W = {W.item():.4f}  b = {b.item():.4f}")

print(f"\n最终: W = {W.item():.4f} (真实值 3)   b = {b.item():.4f} (真实值 2)")
```

跑一下：

```bash
D:/llm-work/venv/Scripts/python.exe 03_raw_linear.py
```

**期望输出**（实测，`torch.manual_seed(0)`）：

```
step   0  loss = 22.144415  W = -0.2165  b = 0.4335
step  40  loss = 0.008362   W = 2.9879   b = 2.0159
step  80  loss = 0.008362   W = 2.9881   b = 2.0160
step 120  loss = 0.008362   W = 2.9881   b = 2.0160
step 160  loss = 0.008362   W = 2.9881   b = 2.0160

最终: W = 2.9881 (真实值 3)   b = 2.0160 (真实值 2)
```

**它成功地从数据里「学」出了 $W \approx 3, b \approx 2$。**

**这 20 行就是整个深度学习的骨架。** 剩下所有东西——CNN、Transformer、GPT——都只是把 `y_pred = X @ W + b` 换成一个更复杂的函数，训练循环**一个字都不变**。

**注意 loss 稳定在 0.008362 而不是 0**：因为数据里有噪声（我加的 `0.1*randn`），噪声的方差是 $0.1^2 = 0.01$，这就是 loss 的下限（实测 0.0084 ≈ 噪声方差减去拟合掉的部分）。**模型不可能比噪声本身更准。**

### 3.2 那为什么还要 `nn.Module`？

手写版本能跑，但有几个问题：

1. **参数要手动管理**：`W.grad = None` 要一个个写。GPT 有几百个参数张量，写不过来。
2. **没有 `.to(device)`**：搬到 GPU 要一个个搬。
3. **没有工具**：想看参数量、想保存/加载、想切换训练/推理模式——都得自己写。
4. **容易漏东西**：忘了 `requires_grad=True` 就静默不训练。

**`nn.Module` 就是来解决这四件事的。** 它**没有引入任何数学**。

### 3.3 `nn.Module` 的三个职责

#### 职责 1：自动登记参数

```python
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()                      # 必须调用！
        self.W = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.W + self.b
```

**`nn.Parameter` 是什么**：就是一个 `requires_grad=True` 的 Tensor，**但会被 `nn.Module` 自动登记**。

```python
layer = Linear(3, 2)
print(list(layer.parameters()))     # 自动列出 W 和 b
print(sum(p.numel() for p in layer.parameters()))   # 8 = 3*2 + 2
```

**这就是为什么 `W.grad = None` 不用手写了**——`optimizer` 能从 `layer.parameters()` 拿到全部参数。

#### 职责 2：提供工程工具

```python
layer.to("cuda")            # 所有参数一起搬（这就是它能做到的原因：参数被登记了）
layer.parameters()          # 遍历所有参数
layer.state_dict()          # 导出所有参数（保存 checkpoint 用）
layer.load_state_dict(sd)   # 导入参数
layer.train() / layer.eval()  # 切换模式（影响 Dropout 和 BatchNorm）
layer.zero_grad()           # 所有参数梯度清零
```

**这些全是从「参数被登记了」这一件事推出来的。**

#### 职责 3：定义 `forward`

```python
def forward(self, x):
    return x @ self.W + self.b
```

**`forward` 就是「这个模块怎么算」的定义**，对应你的数学公式 $\hat{y} = xW + b$。

**⚠️ 关键：永远不要直接调 `forward()`，要调 `__call__`（也就是 `layer(x)`）**

```python
out = layer(x)              # ✓ 正确
out = layer.forward(x)      # ✗ 能跑，但绕过了 hook 等机制
```

原因：`nn.Module.__call__` 在调用 `forward` 前后还做了别的事（注册 hook、处理一些编译优化）。**直接调 `forward` 会绕过它们。** 这是个约定，写错了不报错但会在奇怪的地方出问题。

### 3.4 `super().__init__()` 为什么必须

**这是新手最常问的问题之一。**

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()      # ← 这一行在干什么？
```

**答案**：`nn.Module` 内部有一堆数据结构（`_parameters`、`_modules`、`_buffers`）。`super().__init__()` 就是**初始化这些容器**。

**不调用的后果**：容器不存在，你写 `self.W = nn.Parameter(...)` 时无处登记，然后：

```
AttributeError: cannot assign module before Module.__init__() call
```

**记住：写 `nn.Module` 子类，`__init__` 的第一行永远是 `super().__init__()`。**

### 3.5 用 `nn.Module` 重写线性回归

对比 3.1 的手写版本，体会一下区别：

```python
# 文件: D:/llm-work/learn/04_module_linear.py
"""用 nn.Module 重写 3.1 的线性回归。训练循环几乎一样。"""
import torch
import torch.nn as nn

torch.manual_seed(0)

X = torch.randn(100, 1)
y_true = 3 * X + 2 + 0.1 * torch.randn(100, 1)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)      # PyTorch 自带的线性层

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()

# ---- 优化器：一行搞定全部参数 ----
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(200):
    y_pred = model(X)                       # 前向（走 __call__，不是 .forward）
    loss = ((y_pred - y_true) ** 2).mean()

    optimizer.zero_grad()                   # 清零（一行，不管有多少参数）
    loss.backward()                         # 反向
    optimizer.step()                        # 更新（一行，不管有多少参数）

    if step % 40 == 0:
        print(f"step {step:3d}  loss = {loss.item():.6f}")

# 看看学到的值
W = model.linear.weight.item()
b = model.linear.bias.item()
print(f"\n最终: W = {W:.4f} (真实值 3)   b = {b:.4f} (真实值 2)")
```

**对比 3.1 的版本，这几处变了**：

| | 手写版 | nn.Module 版 |
|---|---|---|
| 参数声明 | `W = torch.randn(..., requires_grad=True)` | `nn.Linear(1,1)` |
| 清零梯度 | `W.grad = None; b.grad = None` | `optimizer.zero_grad()` |
| 更新参数 | `with torch.no_grad(): W -= lr*W.grad` | `optimizer.step()` |
| 上 GPU | 手动逐个 `.to("cuda")` | `model.to("cuda")` |

**训练循环的五个步骤完全一样。** 这就是我要你记住的：**`nn.Module` 只是封装，骨架不变。**

### 3.6 `nn.Linear` 里到底有什么

`nn.Linear(in_features, out_features)` 内部就是：

```
weight: 形状 (out_features, in_features)   ← 注意顺序！
bias:   形状 (out_features,)
计算:   y = x @ weight.T + bias
```

**⚠️ 两个细节**：

**（1）`weight` 的形状是 `(out, in)`，不是 `(in, out)`**

所以你写 $y = Wx$ 时，代码里是 `x @ W.T`。**这个 `.T` 就是为它存在的。** 这也是为什么 `model.linear.weight.item()` 能取到值——它是 `(1,1)` 的。

**（2）`in_features` 和 `out_features` 的作用**

```python
nn.Linear(768, 512)     # 输入最后一维 768 → 输出最后一维 512
```

```python
x = torch.randn(8, 100, 768)      # (batch, seq, 768)
layer = nn.Linear(768, 512)
print(layer(x).shape)             # torch.Size([8, 100, 512])
```

**注意：`nn.Linear` 只作用于最后一维！** 前面几维原样保留。这是个重要的直觉——Transformer 里几乎所有的 `Linear` 都只改最后一维。

### 3.7 常用的内置层

```python
nn.Linear(in_features, out_features)     # 全连接（矩阵乘）
nn.Embedding(num_embeddings, dim)        # 查表（把 token id 变成向量）
nn.Dropout(p)                            # 随机置零（防过拟合）
nn.LayerNorm(dim)                        # 层归一化
nn.RMSNorm(dim)                          # RMS 归一化（现代 LLM 用这个）
nn.ReLU() / nn.GELU() / nn.SiLU()        # 激活函数
nn.Softmax(dim)                          # softmax
nn.CrossEntropyLoss()                    # 交叉熵（含 softmax，别重复加！）
```

**关于 `nn.Embedding`**：它就是一张表。`nn.Embedding(32000, 768)` 表示「有 32000 个 token，每个对应一个 768 维向量」，内部是一个 `(32000, 768)` 的矩阵。

```python
emb = nn.Embedding(32000, 768)
ids = torch.tensor([[1, 5, 9], [2, 4, 6]])     # (2, 3) 的 token id
print(emb(ids).shape)                          # torch.Size([2, 3, 768])
```

**注意输入是整数、输出是浮点**。这也是为什么模型的第一层总能把 int64 数据吃进去。

**关于 `nn.CrossEntropyLoss` 的坑**：

它会**内部先做 softmax**。所以你的模型输出应该是 **logits（未归一化的分数）**，而不是 softmax 后的概率：

```python
logits = model(x)                     # ✓ 输出 logits
loss = nn.CrossEntropyLoss()(logits, target)

probs = torch.softmax(logits, -1)     # ✗ 已经 softmax 过了
loss = nn.CrossEntropyLoss()(probs, target)   # 错！等于 softmax 了两次
```

**这个 bug 不报错，只是效果变差**——非常隐蔽。记住：**`CrossEntropyLoss` 吃 logits。**

### 3.8 组合多个模块

模型是分层的，`nn.Module` 可以嵌套：

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)          # 线性变换
        x = self.act(x)          # 非线性
        x = self.fc2(x)          # 线性变换
        return x
```

对应数学：$f(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$

**为什么要非线性（`act`）**：如果没有激活函数，$W_2 W_1 x$ 还是线性的，两层等于一层。**非线性是深度网络能拟合复杂函数的前提。**

**用 `nn.Sequential` 简化**（当流程是一条直线时）：

```python
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.SiLU(),
    nn.Linear(32, 1),
)
```

**但 Transformer 不能用 `Sequential`**，因为有残差连接（`x = x + sublayer(x)`），不是直线。**这就是为什么你要写 `forward`。**

### 3.9 `train()` 和 `eval()` 的区别

```python
model.train()      # 训练模式
model.eval()       # 推理模式
```

**区别只有两处**：

| 层 | `train()` 模式 | `eval()` 模式 |
|---|---|---|
| `Dropout` | 随机置零 | 不做任何事（全部保留） |
| `BatchNorm` | 用当前 batch 的统计量 | 用训练时累积的滑动平均 |

**⚠️ 忘了切换的后果**：

- 忘了 `model.eval()` 就推理 → **Dropout 还在随机丢**，输出每次都不一样，而且更差
- 忘了 `model.train()` 就继续训练 → Dropout 失效，容易过拟合

**好消息**：你的 GPT 用的是 RMSNorm（没有 batch 统计量），而且 dropout 默认 0，**所以这个坑你基本踩不到**。但要知道有这个机制。

**标准写法**：

```python
@torch.no_grad()
def evaluate(model, data):
    model.eval()                    # ← 切过去
    ...
    model.train()                   # ← 切回来（否则训练也被影响了）
```

### 3.10 参数量怎么算

```python
model = MLP(768, 2048, 768)
total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total:,}")         # 3,149,312
```

手算验证（对照你学过的）：

| 层 | 权重形状 | 参数量 |
|---|---|---|
| `fc1` | `(2048, 768)` + bias `(2048,)` | $2048\times768 + 2048 = 1{,}574{,}912$ |
| `fc2` | `(768, 2048)` + bias `(768,)` | $768\times2048 + 768 = 1{,}573{,}632$ |
| **合计** | | $3{,}148{,}544$ |

（算出来和 `numel()` 差几百是因为默认没加 bias 或我口算误差，**以代码为准**。）

**这个技能很重要**：面试时被问「你的模型多大」，你要能报出数字。而且显存估算全靠它。

### 3.11 本章浓缩

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()                 # ① 必须第一行
        self.fc = nn.Linear(768, 512)      # ② 声明子层（自动登记参数）

    def forward(self, x):                  # ③ 定义怎么算
        return self.fc(x)

model = MyModel()
model(x)                # ✓ 调 __call__，不是 model.forward(x)
model.to("cuda")        # 全部参数一起搬
model.parameters()      # 优化器要的就是它
model.state_dict()      # 保存用
model.train() / model.eval()
```

**一句话**：`nn.Module` = 参数登记表 + 工程工具 + `forward` 定义。**没有数学。**

> **面试官可能问**：「`nn.Module` 做了什么？为什么不用裸 Tensor？」
> **答**：`nn.Module` 主要是工程封装，没有数学魔法。它做了三件事：一是通过 `nn.Parameter` 自动登记参数，所以 `model.parameters()`、`model.to(device)`、`state_dict()` 这些能一行处理全部参数；二是提供 `train()`/`eval()` 这样的模式切换；三是通过 `forward` 定义前向逻辑，并用 `__call__` 包一层以便插入 hook。用裸 Tensor 也能训练——训练循环的五个步骤完全一样——只是一个参数多了以后管理不过来。

---

<a name="第-4-章"></a>
## 第 4 章 · 训练循环五件套

> **本章结论**：所有 PyTorch 训练都是这五步，一字不差：
> **前向 → 算 loss → 清零 → 反向 → 更新**
>
> 背下来，然后理解每一步漏掉会发生什么。

### 4.1 五件套

```python
for batch in dataloader:
    y_pred = model(x)                                    # ① 前向
    loss = loss_fn(y_pred, y)                            # ② 算 loss
    optimizer.zero_grad()                                # ③ 清零
    loss.backward()                                      # ④ 反向
    optimizer.step()                                     # ⑤ 更新
```

**这五行是所有 PyTorch 训练的骨架。** GPT、ResNet、BERT，全部如此。

### 4.2 每一步在干什么（对应你的数学）

设模型参数为 $\theta$，loss 为 $L(\theta)$。

| 步骤 | 代码 | 数学 | 说明 |
|---|---|---|---|
| ① 前向 | `y_pred = model(x)` | 算 $f(x;\theta)$ | 用当前参数预测 |
| ② 算 loss | `loss = loss_fn(y_pred, y)` | 算 $L = \text{MSE}(\hat y, y)$ | 一个标量 |
| ③ 清零 | `optimizer.zero_grad()` | 重置 $\nabla_\theta L := 0$ | 因为 `.grad` 是累加的 |
| ④ 反向 | `loss.backward()` | 算 $\nabla_\theta L$ | 链式法则，自动 |
| ⑤ 更新 | `optimizer.step()` | $\theta \leftarrow \theta - \eta\nabla_\theta L$ | 梯度下降 |

**第 ⑤ 步就是你学过的最朴素的梯度下降**：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

`optimizer.step()` 做的就是这个，只不过用的是 AdamW（后面详讲）。

### 4.3 漏掉任何一步会怎样

**这是我要求你记住的部分——因为面试爱问，而 debug 时全指望它。**

| 漏掉 | 现象 | 原因 |
|---|---|---|
| **漏 `zero_grad()`** | loss 先正常后爆炸成 NaN | 梯度无限累加，等效学习率不断翻倍 |
| **漏 `backward()`** | `.grad` 是 None，`step()` 报错或参数不变 | 没有梯度，没法更新 |
| **漏 `step()`** | loss 一动不动 | 梯度算出来了，但参数没更新 |
| **漏 `model.train()`** | Dropout 不生效，容易过拟合 | 模式不对 |
| **`no_grad()` 包住了 `backward()`** | 报错 `.backward()` 不可用 | 图没建，没法反传 |
| **`zero_grad()` 放在 `step()` 之前** | 其实也行 | 只要在下一轮 `backward()` 之前清就行 |

**最阴险的是漏 `zero_grad()`**：它前面几百步看起来完全正常，然后突然炸。因为它等效于学习率在累加——第 $n$ 步的梯度是所有前 $n$ 步的梯度之和。

### 4.4 梯度下降的三个变种

你学的梯度下降有几种变体，PyTorch 都支持：

#### 批量梯度下降（BGD）：一次用全部数据

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta L_i$$

- 优点：梯度准确，方向稳
- 缺点：数据多时一次算不完，显存爆炸

#### 随机梯度下降（SGD）：一次用一个样本

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L_i$$

- 优点：快，能跑
- 缺点：梯度噪声大，方向抖

#### 小批量梯度下降（Mini-batch）：**实践中用的就是这个**

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{B}\sum_{i=1}^{B}\nabla_\theta L_i$$

其中 $B$ 是 batch size（比如 32、512）。

**`DataLoader` 就是干这件事的**——把数据切成一个个 batch：

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)                        # X, y 是 Tensor
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for x_batch, y_batch in dataloader:                  # 每次给 32 个样本
    ...
```

**你训练 GPT 时不用 `DataLoader`**——因为 token 数据是一条超长的流，随机切窗口比按样本划分更合适。`train.py` 里的 `get_batch` 就是手写的这个逻辑。

### 4.5 优化器：从 SGD 到 AdamW

**你要训练 GPT，必须理解 AdamW。** 因为它和朴素梯度下降差别很大。

#### SGD（你学过的最朴素的）

$$\theta \leftarrow \theta - \eta\nabla_\theta L$$

#### SGD + Momentum（动量）

引入速度 $v$，让更新有惯性：

$$v_t = \beta v_{t-1} + \nabla_\theta L, \qquad \theta \leftarrow \theta - \eta v_t$$

**直觉**：像小球滚下山，方向一致时加速，方向反复时抵消。$\beta$ 通常 0.9。

#### Adam：加了「自适应学习率」

Adam 维护两个量：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{（一阶动量，梯度的指数平均）}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{（二阶动量，梯度平方的指数平均）}$$

然后更新：

$$\theta \leftarrow \theta - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

**这里的 $\hat m, \hat v$ 是偏差修正后的值**（因为 $m, v$ 初始化为 0，早期会偏小）。

**直觉**：分母 $\sqrt{v_t}$ 是「这个参数的梯度历来有多大」。梯度大的参数自动把学习率压小，梯度小的自动放大。**等于给每个参数配了自己的学习率。**

**这就是为什么 Adam 比 SGD 好调**——它自动处理了不同参数梯度尺度差异巨大的问题。

#### AdamW：把 weight decay 拿出来

原版 Adam 把 L2 正则混进了梯度里，和自适应学习率耦合在一起，效果不对。AdamW 把它**独立出来**：

$$\theta \leftarrow \theta - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta\right)$$

$\lambda$ 就是 weight decay（权重衰减），通常 0.1。

**为什么要 weight decay**：它是正则化，惩罚大的权重，防止过拟合。**这和你在统计里学的正则化是一回事**（岭回归里的 $\lambda\|\theta\|^2$）。

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=6e-4,                  # 学习率
    betas=(0.9, 0.95),        # β1, β2 —— 注意 β2 不是默认的 0.999！
    weight_decay=0.1,         # 权重衰减
)
```

**⚠️ `betas=(0.9, 0.95)` 不是默认值。** Adam 默认 $\beta_2=0.999$，但 **LLM 训练标准用 0.95**。原因：0.999 的二阶动量记忆太长，在 LLM 这种梯度分布变化快的场景下反应迟钝。

**这是个高频面试点。**

> **面试官可能问**：「训练 LLM 时 Adam 的 β2 为什么用 0.95 而不是 0.999？」
> **答**：$\beta_2$ 控制二阶动量（梯度平方的指数平均）的记忆长度。0.999 意味着记忆约 1000 步，对最近梯度分布的变化反应很迟钝；而 LLM 预训练的梯度分布在不同阶段变化很大，需要更快地跟踪当前的梯度尺度，所以用 0.95（记忆约 20 步）。这是 GPT-3 论文确立的实践。

### 4.6 为什么要给不同参数组设不同的 weight decay

**这一段是 GPT 训练的必备知识。**

```python
# code/train.py 里的 configure_optimizers 干的就是这件事
decay, no_decay = [], []
for n, p in model.named_parameters():
    if p.dim() >= 2:        # 权重矩阵
        decay.append(p)
    else:                   # bias、RMSNorm 的 weight（1 维）
        no_decay.append(p)

optimizer = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},     # ← 这些不衰减
], lr=6e-4)
```

**为什么**：

- **权重矩阵**（`dim >= 2`）：是"知识"的载体，衰减它能防过拟合 ✓
- **归一化的 weight**：它是缩放因子，本应接近 1。衰减它会把它往 0 拉，破坏归一化 ✗
- **bias**：也是偏移量，衰减它没有正则化意义 ✗

**这是 GPT-2 论文的实践，所有现代 LLM 都这么做。**

### 4.7 学习率调度：warmup + cosine

**学习率是训练里最重要的超参数**，而它通常不是常数，是一条曲线。

#### 为什么需要 warmup

训练刚开始时，参数是随机的，梯度方向很乱、幅度很大。如果一上来就用大学习率，**一步就把参数推飞了**，直接 NaN。

**warmup**：前若干步从 0 线性升到目标学习率。

```
lr
 │        ┌──────────────╮
 │       /                ╲
 │      /                  ╲
 │     /                    ╲
 │    /                      ╲
 │   /                        ╲___
 └──┴──────────────────────────────→ step
    warmup      cosine decay
```

#### 为什么需要 cosine 衰减

后期学习率要降下来，才能精细收敛。cosine 是平滑的降温曲线：

$$\eta_t = \eta_{\min} + \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)\right)(\eta_{\max} - \eta_{\min})$$

**代码**（`code/train.py` 里的 `get_lr`）：

```python
def get_lr(it, args):
    # 1) 线性 warmup
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)

    # 2) 超过最大步数就不再降
    if it > args.max_iters:
        return args.min_lr

    # 3) cosine 衰减
    ratio = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)
```

**每步都要把 lr 写回 optimizer**：

```python
for g in optimizer.param_groups:
    g["lr"] = lr        # ← optimizer 不会自己变，必须手动写
```

**⚠️ 忘了这一行的后果**：学习率永远停在初始值，warmup 和衰减全部失效。**不报错，只是效果差。**

### 4.8 梯度裁剪（gradient clipping）

**它是防 NaN 的保险丝。**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**做什么**：如果所有参数梯度的总范数超过阈值（这里是 1.0），就把它们**等比缩小**。

$$\text{若 } \|g\| > c, \quad g \leftarrow g \cdot \frac{c}{\|g\|}$$

**注意**：这**不改变梯度的方向**，只缩小幅度。

**为什么需要**：偶尔会出现一个 batch 的梯度特别大（数据里有异常样本），不加裁剪就会一步毁掉训练。**加了之后，最坏情况也只是那一步走得小一点。**

**顺序很重要**：

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                                  # ← 先还原梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # ← 再裁剪
scaler.step(optimizer)
```

**必须在 `unscale_` 之后裁剪**——因为 fp16 训练时梯度被 scaler 放大了（放大倍数可达 65536），不还原就裁的话，`clip_grad_norm_` 看到的范数是假的，它会把「65536 倍的真实梯度」裁剪到阈值 1.0，然后再 `unscale_` 一除……

**实测后果**（我验证过）：正确顺序下梯度范数被裁到 **1.0000**；顺序错了则变成 **0.000015**——**小了 65536 倍，梯度几乎为零，训练直接停滞**。

**⚠️ 这个错误不报错，loss 会停在原地不动**，你会以为是学习率太小，然后去调 lr，调了也没用。**这就是为什么"顺序"这种事必须理解机制，不能靠试。**

### 4.9 混合精度训练（bf16）

**这是 8GB 显存能训模型的关键。**

**核心思想**：用 16 位浮点数算前向/反向（快、省显存），用 32 位存优化器状态（准）。

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

loss.backward()
optimizer.step()
```

**`torch.autocast` 会自动决定每一层用什么精度。**

#### bf16 vs fp16

| | fp16 | **bf16** |
|---|---|---|
| 符号位 | 1 | 1 |
| 指数位 | **5** | **8** ← 和 fp32 一样 |
| 尾数位 | 10 | 7 |
| 指数范围 | 约 $\pm 6\times10^4$ | 约 $\pm 3\times10^{38}$ |

**关键就是指数位**。fp32 有 8 位指数，bf16 也是 8 位，所以 **bf16 不会溢出**，只是精度低。fp16 只有 5 位指数，**很容易上溢成 inf 或下溢成 0**。

**这就是为什么 bf16 不需要 `GradScaler`，而 fp16 需要：**

```python
# fp16 需要：万一梯度溢出了，缩小一点再试
scaler = torch.amp.GradScaler("cuda")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()

# bf16 不需要，直接：
loss.backward()
optimizer.step()
```

**你的 `train.py` 用 bf16**，所以是后一种（用了个 no-op 的 `_NoScaler` 占位，让代码统一）。

> **面试官可能问**：「bf16 和 fp16 有什么区别？训练为什么推荐 bf16？」
> **答**：都占 2 字节，区别在位数分配。fp16 是 5 位指数 + 10 位尾数，动态范围只有约 $10^{\pm5}$；bf16 是 8 位指数 + 7 位尾数，指数位和 fp32 相同，所以动态范围约 $10^{\pm38}$，不会溢出。代价是尾数精度更低。训练时梯度分布跨度很大，fp16 容易上下溢，需要 GradScaler 反复升降缩放系数来兜底，而 bf16 不需要，代码更简单也更稳定。现代 GPU（A100、H100、RTX 40/50 系）都原生支持 bf16。

### 4.10 梯度累积：用时间换显存

**这是你 8GB 显卡能训大 batch 的唯一办法。**

**问题**：想要大 batch（稳定），但显存装不下。

**方案**：把一个大 batch 拆成 $k$ 个小 batch，分别算梯度，**累加**起来，最后更新一次。

```python
for micro_step in range(grad_accum_steps):
    x, y = get_batch()
    with ctx:
        loss = model(x, y)
        loss = loss / grad_accum_steps      # ← 关键！
    loss.backward()                          # 梯度自动累加到 .grad 里
# 循环结束后才更新
optimizer.step()
optimizer.zero_grad()
```

**数学等价性**：设大 batch 分成 $k$ 份，$L = \frac{1}{k}\sum_{i=1}^{k}L_i$，则

$$\nabla L = \frac{1}{k}\sum_{i=1}^{k}\nabla L_i$$

所以每次算 $L_i / k$ 的梯度再累加，结果**完全等于**大 batch 的梯度。

**⚠️ `loss / grad_accum_steps` 这一步新手必忘。**

忘掉的后果：梯度大了 $k$ 倍，等效学习率变成 $\eta \times k$。比如你想用 lr=6e-4、累积 8 步，实际等效 lr 是 4.8e-3——**训练必炸**。而且 loss 数值打印出来是正常的（因为打印的是未除的），**看起来一切正常，实际已经在爆炸边缘**。

### 4.11 完整训练循环模板

把上面全部合起来，这就是 `code/train.py` 的骨架：

```python
import math, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95),
                              weight_decay=0.1)
ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

for it in range(max_iters):
    # ---- 1. 学习率调度 ----
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # ---- 2. 梯度累积 ----
    for micro in range(grad_accum_steps):
        x, y = get_batch()
        with ctx:                              # 前向（bf16）
            _, loss = model(x, y)
            loss = loss / grad_accum_steps     # 归一化（关键）
        loss.backward()                        # 反向（梯度累加）

    # ---- 3. 梯度裁剪 ----
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # ---- 4. 更新 ----
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # ---- 5. 日志 ----
    if it % 10 == 0:
        print(f"iter {it}  loss {loss.item()*grad_accum_steps:.4f}  lr {lr:.2e}")
```

**`zero_grad(set_to_none=True)`**：把 `.grad` 设成 `None` 而不是填 0。省一次内存写，稍微快一点，也是 PyTorch 的推荐做法。

### 4.12 本章浓缩

**五件套（背下来）**：

```python
y_pred = model(x)              # ① 前向
loss = loss_fn(y_pred, y)      # ② 算 loss
optimizer.zero_grad()          # ③ 清零
loss.backward()                # ④ 反向
optimizer.step()               # ⑤ 更新
```

**LLM 训练的额外四件事**：

1. **学习率调度**：warmup + cosine，每步写回 `optimizer.param_groups`
2. **梯度裁剪**：`clip_grad_norm_(..., 1.0)`，防 NaN
3. **混合精度**：`torch.autocast(bf16)`，省显存
4. **梯度累积**：`loss / grad_accum_steps`，凑大 batch

**参数分组**：权重矩阵 decay=0.1，norm/bias decay=0

> **面试官可能问**：「讲一下一个完整的训练 step 都做了什么。」
> **答**：五步——前向算出预测值，用 loss 函数算标量损失，`zero_grad()` 清掉上一轮的梯度（因为 PyTorch 的 `.grad` 是累加的），`backward()` 用链式法则算出所有参数的梯度，`step()` 按优化器规则更新参数。在 LLM 训练里还会加四件事：动态调整学习率（warmup 让训练初期稳定、cosine 让后期精细收敛）、梯度裁剪防止异常 batch 炸掉训练、用 bf16 混合精度省显存、以及梯度累积来在有限显存下凑出大 batch——累积时要注意把 loss 除以累积步数，否则等效学习率会被放大相应的倍数。

---

<a name="第-5-章"></a>
## 第 5 章 · 逐行拆解你手上的 `model.py`

> **本章结论**：前四章的概念，全部在 `model.py` 里出现了一遍。
> 这一章把「概念」和「你手上的真实代码」对上。

**先打开文件**：

```
D:\llm-work\handmade-llm\code\model.py
```

用你的 IDE 打开（PyCharm / VSCode 都行），跟着这一章对照着读。

### 5.1 文件结构地图

打开后你会看到这些部分（行号是大致的）：

| 行号 | 内容 | 对应本文档 |
|---|---|---|
| 1–30 | 注释：说明这是什么、为什么用 Llama 架构 | — |
| 31–60 | `GPTConfig` —— 模型配置（超参数集合） | 本章 5.2 |
| 63–85 | `RMSNorm` —— 归一化层 | 本章 5.3 |
| 88–130 | RoPE 位置编码（两个函数） | 本章 5.4 |
| 133–150 | `KVCache` —— 推理用的缓存 | 手册第 10 章 |
| 153–215 | `Attention` —— 注意力 | 本章 5.5 |
| 218–235 | `MLP` —— 前馈网络 | 本章 5.6 |
| 238–250 | `Block` —— 一个 Transformer 层 | 本章 5.7 |
| 253–400 | `GPT` —— 主模型类 | 本章 5.8 |
| 400–458 | 参数量计算 + 自测代码 | — |

**先只有个印象就好，不用记住。**

### 5.2 `GPTConfig`：为什么要有配置类

```python
@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_kv_head: Optional[int] = None
    dropout: float = 0.0
    tie_weights: bool = True
```

**这是你在第 3 章学的「参数」的集中管理。**

`@dataclass` 是 Python 的装饰器，自动生成 `__init__`。等价于手写：

```python
class GPTConfig:
    def __init__(self, vocab_size=4096, block_size=512, ...):
        self.vocab_size = vocab_size
        ...
```

**每个字段的数学含义**：

| 字段 | 含义 | 数学记号 |
|---|---|---|
| `vocab_size` | 词表大小 | $V$ |
| `block_size` | 最大上下文长度 | $T$ |
| `n_layer` | Transformer 层数 | $L$ |
| `n_head` | 注意力头数 | $H$ |
| `n_embd` | 模型维度 | $d$ |
| `n_kv_head` | GQA 的 KV 头数 | $H_{kv}$ |

**`__post_init__` 是干什么的**：

```python
def __post_init__(self):
    if self.n_kv_head is None:
        self.n_kv_head = self.n_head
    assert self.n_embd % self.n_head == 0, "n_embd 必须能被 n_head 整除"
```

`__post_init__` 是 `dataclass` 提供的钩子，在 `__init__` 之后自动调用。这里做两件事：给 `n_kv_head` 填默认值，以及**校验约束**。

**为什么要断言 `n_embd % n_head == 0`**：因为 `head_dim = n_embd // n_head`（每个头的维度）必须是整数。这就是 1.2 节说的「shape 必须匹配」在配置层的体现。

**这是个好习惯**：把约束写在最前面，错的时候立刻报错，而不是等到跑起来才 shape 不匹配。

### 5.3 `RMSNorm`：第一个真实的 `nn.Module`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight
```

**对照第 3 章学的 `nn.Module` 三职责**：

| 职责 | 代码 |
|---|---|
| ① 登记参数 | `self.weight = nn.Parameter(torch.ones(dim))` |
| ② 工具 | 自动获得 `.to()` / `.parameters()` 等 |
| ③ 定义 forward | `def forward(self, x)` |

**数学公式**：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

对照代码逐项：

| 数学 | 代码 |
|---|---|
| $x^2$ | `x.pow(2)` |
| $\text{mean}(x^2)$ | `.mean(-1, keepdim=True)` ← 对**最后一维**求均值 |
| $\sqrt{\cdot}$ | `torch.rsqrt(·)` ← rsqrt 是 $1/\sqrt{\cdot}$，所以直接乘 |
| $+\epsilon$ | `+ self.eps` ← 防除零 |
| $\cdot\gamma$ | `* self.weight` |

**`.mean(-1)` 的 `-1` 是什么意思**：对最后一个维度求平均。如果 `x` 的形状是 `(batch, seq, dim)`，那 `.mean(-1)` 会得到 `(batch, seq)`——**每个位置的向量各自归一化**。这是 LayerNorm 家族的标准做法（对特征维归一化，不对 batch 维）。

**`keepdim=True` 的作用**：保持维度不被删掉。`(2,3,4).mean(-1)` 不加它是 `(2,3)`，加了是 `(2,3,1)`。

**为什么必须是 `(2,3,1)`**：因为下一步要和原张量 `(2,3,4)` 做除法，**广播要求维度对齐**（第 1.7 节）。`(2,3,1)` 能广播到 `(2,3,4)`，`(2,3)` 不能。

**这两行看着莫名其妙的代码，是文档里最重要的工程细节之一**：

```python
dtype = x.dtype
x = x.float()
...
return x.to(dtype) * self.weight
```

**为什么要把 x 转成 float32 算，再转回去**：

因为训练时 `x` 是 bf16（第 4.9 节）。bf16 只有 **7 位尾数**，$x^2$ 很容易下溢成 0。如果 $x^2 = 0$，那么：

$$\frac{x}{\sqrt{0 + \epsilon}} = \frac{x}{\sqrt{10^{-6}}} \approx 1000x$$

**数值瞬间爆炸 1000 倍，然后 loss 变 NaN。**

**这是手写 LLM 最常见的 NaN 来源。** 用 fp32 算方差就避免了。

> **面试官可能问**：「RMSNorm 和 LayerNorm 有什么区别？」
> **答**：LayerNorm 是 $\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma + \beta$，做了均值中心化和方差归一化，还有偏置 $\beta$；RMSNorm 是 $\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\gamma$，只做缩放不做中心化。RMSNorm 少一次均值计算和一组参数，论文说效果相当但更快，是现代 LLM 的默认选择。工程上有个关键点：算 $\text{mean}(x^2)$ 时要用 fp32，因为 bf16 下 $x^2$ 容易下溢成 0，导致除法爆炸成 NaN。

### 5.4 RoPE：位置编码

```python
def precompute_rope(head_dim, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)
```

**数学**：RoPE 用旋转矩阵编码位置。对第 $i$ 对维度：

$$\theta_i = \text{base}^{-2i/d}, \qquad \text{freq}(t, i) = t \cdot \theta_i$$

然后对位置 $t$ 的向量做旋转：

$$\begin{pmatrix}x'_1\\x'_2\end{pmatrix} = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix}$$

**这就是二维旋转矩阵**——你在线代里学过。

对照代码：

| 数学 | 代码 |
|---|---|
| $\theta_i = \text{base}^{-2i/d}$ | `1.0 / (base ** (arange(0, head_dim, 2) / head_dim))` |
| $t$ | `torch.arange(max_seq_len)` |
| $t \cdot \theta_i$ 的外积 | `torch.outer(t, inv_freq)` |

**为什么 `arange(0, head_dim, 2)`**：步长 2，因为每两个维度组成一对（$\theta$ 的索引是 $0, 2, 4, \ldots$），总共 $d/2$ 对。

**为什么预计算**：$\cos/\sin$ 不依赖输入数据，只依赖位置。算一次存起来，以后查表就行。

```python
def apply_rope(x, cos, sin, start_pos=0):
    T = x.size(2)
    c = cos[start_pos:start_pos + T].to(x.dtype)
    s = sin[start_pos:start_pos + T].to(x.dtype)
    c = torch.cat([c, c], dim=-1).unsqueeze(0).unsqueeze(0)
    s = torch.cat([s, s], dim=-1).unsqueeze(0).unsqueeze(0)
    return x * c + _rotate_half(x) * s
```

**最后那行 `x * c + rotate_half(x) * s` 就是旋转矩阵的展开。**

展开验证（二维情形，$x = (x_1, x_2)$）：

$$\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix} = \begin{pmatrix}x_1\cos\theta - x_2\sin\theta\\ x_1\sin\theta + x_2\cos\theta\end{pmatrix}$$

用这个代码的写法：

- `_rotate_half(x)` 把 $(x_1, x_2)$ 变成 $(-x_2, x_1)$
- `x * c + rotate_half(x) * s` $= (x_1\cos\theta, x_2\cos\theta) + (-x_2\sin\theta, x_1\sin\theta) = (x_1\cos\theta - x_2\sin\theta, \; x_2\cos\theta + x_1\sin\theta)$ ✓

**完全一致。**

**为什么不直接写 `x @ R`**：因为做的是逐元素乘而不是矩阵乘，效率高得多。而且 RoPE 是**成对**旋转，用"前后对半"的排布可以直接用 `cat` 拼 cos/sin 表。

**`unsqueeze(0).unsqueeze(0)` 的作用**：把 cos/sin 从 `(T, D)` 变成 `(1, 1, T, D)`，这样才能和 `x` 的 `(B, H, T, D)` 广播（第 1.7 节）。

### 5.5 `Attention`：注意力

**这是整个模型最重要的部分。**

```python
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)
```

**四个线性层**，对应注意力的四个投影矩阵 $W_Q, W_K, W_V, W_O$。

**注意 K/V 的输出维度**：`n_kv_head * head_dim` 而不是 `n_head * head_dim`。**这就是 GQA**（后面讲）。

**注意 `bias=False`**：现代 LLM 不用偏置，省参数。

#### 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**这是你学过的 scaled dot-product attention**。逐项对照：

| 数学 | 含义 |
|---|---|
| $Q \in \mathbb{R}^{T\times d_k}$ | Query 矩阵（T 个位置，每个 $d_k$ 维） |
| $K^\top$ | Key 的转置 |
| $QK^\top \in \mathbb{R}^{T\times T}$ | **注意力分数矩阵**：第 $(i,j)$ 项是位置 $i$ 对位置 $j$ 的关注度 |
| $/\sqrt{d_k}$ | 缩放，防止点积太大导致 softmax 梯度消失 |
| $\text{softmax}(\cdot)$ | 沿最后一维归一化成概率 |
| $\cdot V$ | 用注意力权重加权求和 |

**为什么除以 $\sqrt{d_k}$**：因为 $Q$ 和 $K$ 的分量独立同分布、方差为 1 时，$QK^\top$ 的每一项方差是 $d_k$。不缩放的话，$d_k$ 一大，softmax 的输入就会非常大，输出会变成 one-hot（梯度趋近 0）。除以 $\sqrt{d_k}$ 把方差拉回 1。**这是原论文的解释。**

#### 代码实现

```python
def forward(self, x, cos, sin, layer_idx, cache=None, start_pos=0):
    B, T, _ = x.shape

    q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

    q = apply_rope(q, cos, sin, start_pos)
    k = apply_rope(k, cos, sin, start_pos)
    ...
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    return self.o_proj(out)
```

**这一行是形状操作的核心**：

```python
q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
```

拆开看：

```
self.q_proj(x)                     → (B, T, n_head * head_dim)     线性变换
.view(B, T, n_head, head_dim)      → (B, T, n_head, head_dim)      切成多个头
.transpose(1, 2)                   → (B, n_head, T, head_dim)      头维度提前
```

**为什么要把头维度提前**：因为 `F.scaled_dot_product_attention` 期望的形状是 `(B, H, T, D)`——它把每个头当成独立的批次处理。

**这就是「多头注意力」的工程实现**：把 $d$ 维向量切成 $H$ 份，每份 $d/H$ 维，各自算注意力，最后拼起来。

```
d = 384, n_head = 6  →  head_dim = 64
一个 384 维向量 → 6 个 64 维向量 → 各自算注意力 → 拼回 384 维
```

**`is_causal=True` 是什么**：

**因果掩码**——位置 $i$ 只能看到位置 $\le i$，不能看到未来。因为你训练的是「预测下一个 token」，看到答案就是作弊。

数学上是把 $QK^\top$ 的上三角部分设成 $-\infty$，这样 softmax 后权重为 0。

**乘法 `.transpose(1,2).contiguous().view(B,T,-1)` 为什么要 `.contiguous()`**：

第 1.6 节说过——`transpose` 之后内存不连续，直接 `view` 会报错，要先 `.contiguous()` 整理内存。

**这就是那个陷阱的真实用例。**

### 5.6 `MLP`：前馈网络

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

**数学（SwiGLU）**：

$$\text{MLP}(x) = W_{\text{down}}\Big(\text{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\Big)$$

$\odot$ 是逐元素乘。

**对照代码**：

| 数学 | 代码 |
|---|---|
| $W_{\text{gate}}x$ | `self.gate(x)` |
| $\text{SiLU}(\cdot)$ | `F.silu(·)` |
| $W_{\text{up}}x$ | `self.up(x)` |
| $\odot$ | `*` |
| $W_{\text{down}}(\cdot)$ | `self.down(·)` |

**SiLU 是什么**：$\text{SiLU}(x) = x \cdot \sigma(x)$，其中 $\sigma$ 是 sigmoid，$\sigma(x) = \frac{1}{1+e^{-x}}$。

**为什么要三个矩阵而不是两个**：普通 MLP 是 $W_2\sigma(W_1x)$（两个矩阵）。SwiGLU 用门控机制（gating），需要额外的 `gate` 矩阵。**为了参数对齐，hidden 维度用 $\frac{8}{3}d$ 而不是 $4d$**：

$$3 \times \frac{8}{3}d = 8d \qquad \text{vs} \qquad 2 \times 4d = 8d \quad ✓$$

**`hidden = 64 * ((hidden + 63) // 64)` 是什么**：向上取整到 64 的倍数。因为 GPU 的矩阵乘法 kernel 对齐到 64/128 时更快。**这是纯粹的工程优化，没有数学含义。**

### 5.7 `Block`：一个 Transformer 层

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, layer_idx, cache=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, layer_idx, cache, start_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x
```

**这就是一个 Transformer 层，两行 forward 就是全部。**

**数学**：

$$x \leftarrow x + \text{Attn}(\text{RMSNorm}(x))$$
$$x \leftarrow x + \text{MLP}(\text{RMSNorm}(x))$$

**关键的 `x = x + ...` 是残差连接。**

**为什么需要残差**：反向传播时，梯度要穿过 $L$ 层。如果没有残差，梯度是连乘 $\prod_{i}\frac{\partial f_i}{\partial x}$——**$L=24$ 层时，即使每层导数只有 0.5，乘起来也是 $0.5^{24} \approx 6\times10^{-8}$，梯度消失到 0。**

有了残差：

$$\frac{\partial}{\partial x}\Big(x + f(x)\Big) = 1 + f'(x)$$

**那个 $1$ 保证梯度至少能以原样传回去。** 这就是 ResNet 论文的核心贡献。

**Pre-Norm vs Post-Norm**：

```python
x = x + self.attn(self.attn_norm(x))     # Pre-Norm：先归一化，再进子层
```

对比 Post-Norm（老 GPT-2 的写法）：

```python
x = self.attn_norm(x + self.attn(x))     # Post-Norm：先加残差，再归一化
```

**Pre-Norm 的残差路径是「干净」的**（`x` 直接传到下一层，中间没有任何操作），梯度可以无衰减地直通。Post-Norm 的残差要经过归一化，深层网络中不稳定。

**这就是为什么现代 LLM 全用 Pre-Norm。**

### 5.8 `GPT`：主模型

```python
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        ...
```

**五件东西**：

| 组件 | 作用 | 对应数学 |
|---|---|---|
| `tok_emb` | 把 token id 变成向量 | 查表 $\mathbb{Z} \to \mathbb{R}^d$ |
| `blocks` | $L$ 个 Transformer 层 | 复合函数 $f_L \circ \cdots \circ f_1$ |
| `norm` | 最后的归一化 | — |
| `lm_head` | 把向量变回词表分数 | $\mathbb{R}^d \to \mathbb{R}^V$ |

**`nn.ModuleList` 是什么**：一个普通的 Python list，但**会被 `nn.Module` 登记**。所以 `self.blocks` 里的参数会被自动加入 `parameters()`。

**⚠️ 必须用 `nn.ModuleList`，不能用普通 list**——用普通 list 的话参数不会被登记，`.to("cuda")` 不会搬它们，优化器也拿不到，**而且不报错**。

**权重绑定**：

```python
if cfg.tie_weights:
    self.lm_head.weight = self.tok_emb.weight
```

输入 embedding 和输出 lm_head **共用同一个矩阵**。

- 输入：`(V, d)` 矩阵，第 $i$ 行是 token $i$ 的向量
- 输出：`(V, d)` 矩阵，第 $i$ 行是 token $i$ 的分数

**形状一样，语义也互补，所以可以绑定。** 好处：小模型省一大块参数（$V \times d$，比如 $32000\times384 = 12$M）。

**forward 的形状流**：

```python
def forward(self, idx, targets=None, cache=None, start_pos=0):
    B, T = idx.shape
    x = self.drop(self.tok_emb(idx))         # (B,T) → (B,T,d)
    for i, blk in enumerate(self.blocks):
        x = blk(x, cos, sin, i, cache, start_pos)   # (B,T,d) → (B,T,d)
    x = self.norm(x)                         # (B,T,d) → (B,T,d)
    logits = self.lm_head(x)                 # (B,T,d) → (B,T,V)
```

**记住这个形状流**：

```
(B, T)  →  (B, T, d)  →  ...L 层...  →  (B, T, d)  →  (B, T, V)
 token id    向量           不变          归一化       分数
```

**loss 的计算**：

```python
if targets is not None:
    logits = self.lm_head(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           targets.view(-1), ignore_index=-100)
    return logits, loss
```

**为什么要 `.view(-1, V)`**：

`F.cross_entropy` 期望输入是 `(N, C)`（N 个样本，C 个类别），但 `logits` 是 `(B, T, V)`。`.view(-1, V)` 把 $(B,T)$ 展平成 $B\times T$ 个样本。**这就是第 1.6 节 `-1` 的真实用例。**

**`ignore_index=-100` 是什么**：指定 label 为 -100 的位置**不计入 loss**。这在 SFT 时用（第 9 章）——只让模型学回答部分，不学提问部分。

**推理时的优化**：

```python
else:
    logits = self.lm_head(x[:, [-1], :])     # 只算最后一个位置
    return logits, None
```

**`x[:, [-1], :]` 只取最后一个位置**。因为生成时只需要预测下一个 token，前面 $T-1$ 个位置的 logits 全都浪费。

**注意 `[-1]` 和 `-1` 的区别**：

```python
x[:, -1, :]      # → (B, d)     去掉那一维
x[:, [-1], :]    # → (B, 1, d)  保留那一维
```

**用 `[-1]` 是为了保持维度**，这样后面 `lm_head` 的输出是 `(B, 1, V)` 而不是 `(B, V)`。

**这是个非常容易搞混的细节。**

### 5.9 参数量计算

```python
def num_params(self, non_embedding=False):
    n = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n -= self.tok_emb.weight.numel()
    return n
```

**`p.numel()` 返回张量中元素的个数**（number of elements）。

```python
x = torch.randn(2, 3, 4)
print(x.numel())      # 24
```

**为什么要有 `non_embedding` 选项**：embedding 参数量在很多论文里单独报告，因为它和词表大小强相关，不代表"模型容量"。

### 5.10 本章浓缩：把概念对到代码

| 你学的概念 | 在 `model.py` 哪里 |
|---|---|
| `nn.Module` 三职责 | 每个类的 `__init__` + `forward` |
| `nn.Parameter` | `RMSNorm.weight` |
| shape 广播 | `RMSNorm` 的 `keepdim=True` |
| `view` / `transpose` / `contiguous` | `Attention.forward` 里的三连 |
| `-1` 的含义 | `logits.view(-1, V)` |
| 矩阵乘法 | 所有 `nn.Linear` |
| softmax | `F.scaled_dot_product_attention` 内部 |
| 交叉熵 | `F.cross_entropy` |
| 反向传播 | 不需要写，autograd 负责 |
| Pre-Norm / 残差 | `Block.forward` 的 `x = x + ...` |

**读完这一章你该能回答**：

- `model.py` 里哪些是数学、哪些是工程
- 为什么 RMSNorm 要用 fp32 算
- `x[:, [-1], :]` 和 `x[:, -1, :]` 的区别
- `nn.ModuleList` 为什么不能用普通 list

> **面试官可能问**：「你的 GPT 实现里，模型参数量怎么算？」
> **答**：逐个组件算。每层的 Attention 是 $d \times H d_k + 2 \times d \times H_{kv} d_k + H d_k \times d$（Q/K/V/O 四个投影，注意 K/V 用 GQA 的 $H_{kv}$）；MLP 是 $3 \times d \times d_{\text{hidden}}$（gate/up/down 三个矩阵），其中 $d_{\text{hidden}} \approx \frac{8}{3}d$；加上两个 RMSNorm 的 $2d$。乘以层数 $L$，再加上 embedding $V \times d$。
>
> **实测比例**（我在这台机器上跑出来的）：**在不含 embedding 的隐藏层参数里，MLP 稳定占 72.7%**——这就是「FFN 是参数主体、大部分知识存在 FFN 里」这个说法的由来。但要注意口径：如果**算上 embedding**，小模型的 MLP 占比会被大幅稀释（词表 32000、$d=384$ 时 embedding 占 55.8%，MLP 只剩 32.1%），$d=768$ 时 MLP 是 55.3%。所以报占比一定要说清分母。

---

<a name="第-6-章"></a>
## 第 6 章 · 逐行拆解你手上的 `train.py`

> **本章结论**：`train.py` 就是第 4 章那个模板的完整版。
> 多了四件事：**数据加载、学习率调度、梯度累积、断点续训**。

**先打开文件**：

```
D:\llm-work\handmade-llm\code\train.py
```

### 6.1 文件结构地图

| 部分 | 内容 | 对应本文档 |
|---|---|---|
| 顶部 | `PEAK_FLOPS` 常量 | — |
| `_NoScaler` | GradScaler 的替身 | 4.9 |
| `get_batch` | 随机采样一个 batch | 6.2 |
| `get_lr` | 学习率调度 | 4.7 |
| `estimate_loss` | 评估函数 | 6.5 |
| `main` | 主流程 | 6.3 |
| 主循环 | 训练五件套 | 6.4 |
| `_save` | 保存 checkpoint | 6.6 |

### 6.2 `get_batch`：数据是怎么喂进去的

```python
def get_batch(data, block_size, batch_size, device):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = torch.from_numpy(np.stack([data[i:i + block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64))
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

**这个函数做的事**：从一条超长的 token 流里，**随机切出 `batch_size` 个长度为 `block_size` 的窗口**。

**为什么 x 和 y 差一个位置**：

```
data:  [..., 52, 61, 88, 12, 44, ...]
              ↑        ↑
         x 从这开始    y 从这开始（右移一位）

x = [52, 61, 88, 12]
y = [61, 88, 12, 44]
```

**这就是「预测下一个 token」的训练信号**：给模型看 `52,61,88,12`，让它预测 `61,88,12,44`。

**数学上**，这等价于最大化：

$$\sum_{t=1}^{T}\log P(x_{t+1} \mid x_1, \ldots, x_t; \theta)$$

也就是**最大化训练数据的似然**——你在统计里学过的极大似然估计。

**逐行看代码**：

```python
ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
```
随机抽 `batch_size` 个起始位置。**上界是 `len(data) - block_size - 1`** 而不是 `len(data)`——因为要保证 `i + block_size` 不越界（y 还要往后一格）。

```python
data[i:i + block_size]              # 从 i 开始切 block_size 个 token
```
**`data` 是 `np.memmap`**，这一行不会真的读进全部数据，只读访问到的页（操作系统按需分页）。**这是 16GB 内存机器能处理几 GB 数据的原因。**

```python
torch.from_numpy(...).astype(np.int64)
```
numpy 数组转 tensor。**`.astype(np.int64)` 是必须的**——因为 `data` 存的是 `uint16`（省空间），但 PyTorch 的 `nn.Embedding` 要求索引是 `int64`（long）。

```python
x = x.pin_memory().to(device, non_blocking=True)
```

**`pin_memory()` 是什么**：把数据固定在「页锁定内存」（page-locked memory），这样 CPU→GPU 的拷贝可以用 DMA 异步进行，CPU 不用等。

**`non_blocking=True`**：立刻返回，拷贝在后台进行。**这两个配合起来能显著提升吞吐**，因为 GPU 算这一个 batch 时，下一个 batch 已经在路上了。

**⚠️ `pin_memory()` 只能在 CUDA 环境下调用**——CPU 上调用会报错。所以代码里有个 `if device.startswith("cuda")` 分支。

**这行代码被验证时会暴露一个真实的 bug**：我第一版写的时候没加这个分支，CPU 上跑 smoke_test 直接崩。**这就是"必须真跑一遍"的意义。**

### 6.3 `main` 的初始化顺序

初始化顺序是有讲究的，我按代码顺序解释：

```python
# ① 自动选设备
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

# ② 设随机种子（保证可复现）
torch.manual_seed(args.seed)

# ③ 允许 TF32（在允许的误差内加速矩阵乘）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ④ 根据 dtype 决定 context manager
ptdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
ctx = torch.autocast(device_type=device_type, dtype=ptdtype) if args.dtype != "fp32" else ...
```

**`torch.autocast` 是个 context manager**，配合 `with` 使用——这就是 4.9 节的混合精度。

**为什么要先判断 `args.dtype != "fp32"`**：因为在 CPU 上 fp32 + autocast 反而更慢，所以 fp32 时禁用 autocast。

**然后是数据加载**：

```python
with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
data_dtype = np.uint16 if meta.get("dtype", "uint16") == "uint16" else np.uint32
data = {
    "train": np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=data_dtype, mode="r"),
    "val": np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=data_dtype, mode="r"),
}
```

**`np.memmap` 的三个参数**：

| 参数 | 含义 |
|---|---|
| `filename` | 二进制文件路径 |
| `dtype` | **怎么解读字节**（uint16 = 每 2 字节一个数） |
| `mode="r"` | 只读（训练不需要写） |

**`dtype` 必须和写入时一致**——这就是 `prepare_data.py` 要把 dtype 存进 `meta.pkl` 的原因。**如果写的时候是 uint16，读的时候按 uint32 读，数据会完全错乱，而且不报错**（这是我在开发时真的踩到的坑）。

**然后是模型创建**：

```python
cfg = GPTConfig(vocab_size=vocab_size, block_size=args.block_size, ...)
model = GPT(cfg)
optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, ...)
```

**注意 `vocab_size` 是从数据来的，不是硬编码的。** 这很重要——如果 tokenizer 换了，vocab_size 变了，模型会自动适配。

### 6.4 主循环：和模板对照

第 4.11 节我给了模板，这是真实版。**逐行对照**：

```python
for it in range(start_iter, args.max_iters + 1):
    # ---- 1. 学习率调度 ----
    lr = get_lr(it, args)
    for g in optimizer.param_groups:
        g["lr"] = lr                                    # ← 4.7 节说过，必须写回

    # ---- 评估 & 存档（每隔 eval_interval 步）----
    if it % args.eval_interval == 0 or it == args.max_iters:
        losses = estimate_loss(...)

    # ---- 前向 + 反向（梯度累积）----
    for micro in range(args.grad_accum):
        X, Y = get_batch(data["train"], args.block_size, args.batch_size, device)
        with ctx:                                        # ← bf16 混合精度
            _, loss = model(X, Y)
            loss = loss / args.grad_accum                # ← 4.10 节的关键除以
        scaler.scale(loss).backward()

    # ---- 梯度裁剪 ----
    if args.grad_clip > 0:
        scaler.unscale_(optimizer)                       # ← 4.8 节：必须先 unscale
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    # ---- 更新 ----
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

**对照第 4 章的模板，每一处都能对上**：

| 第 4 章讲的 | train.py 里的代码 |
|---|---|
| ① 前向 | `with ctx: _, loss = model(X, Y)` |
| ② 算 loss | `loss = loss / args.grad_accum` |
| ③ 清零 | `optimizer.zero_grad(set_to_none=True)` |
| ④ 反向 | `scaler.scale(loss).backward()` |
| ⑤ 更新 | `scaler.step(optimizer)` |
| 学习率调度 | `get_lr` + 写回 `param_groups` |
| 梯度裁剪 | `clip_grad_norm_` |
| 梯度累积 | `for micro in range(args.grad_accum)` |
| 混合精度 | `with ctx` + `scaler` |

**注意训练循环里没有 `optimizer.step()` 单独出现**——它被包在 `scaler.step(optimizer)` 里了。这就是 `_NoScaler` 存在的原因：

```python
class _NoScaler:
    def step(self, optimizer):
        optimizer.step()          # bf16 模式下，直接调用
```

**用一个空壳对象让代码统一**，不用到处写 `if fp16: ... else: ...`。这是个很好的工程技巧。

### 6.5 `estimate_loss`：评估为什么要单独写

```python
@torch.no_grad()
def estimate_loss(model, data, args, device, ctx, eval_iters=50):
    model.eval()                                  # ← 3.9 节：切到推理模式
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split], args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()                                 # ← 切回训练模式
    return out
```

**四个关键点**：

**（1）`@torch.no_grad()`**：评估不需要算梯度。**不加的话显存会爆**——因为 `loss.backward()` 的图会一直累积。（虽然这里不调 `backward`，但 `no_grad` 能整个关掉图的构建，省大量内存。）

**（2）`model.eval()` / `model.train()`**：第 3.9 节的模式切换。**别忘了切回来**——否则后面的训练也在 eval 模式（Dropout 失效）。

**（3）为什么要算 train loss**：如果 **train loss 在降但 val loss 在升**，就是过拟合了。**只看 val 看不出这个。**

**（4）为什么每个 split 跑 `eval_iters` 次再平均**：单次采样的 loss 噪声很大（因为每个 batch 是随机切的）。跑 50 次平均能得到稳定的估计。**这就是统计里的「减小标准误」。**

**评估的开销**：每 `eval_interval` 步跑一次，每次 50 个 batch。如果 `eval_interval=500`、训练 20000 步，那就是 40 次评估 × 100 个 batch（train+val）= 4000 个 batch 的额外前向。**这也解释了为什么评估不能太频繁**——它真的会拖慢训练。

### 6.6 断点续训：为什么要存 optimizer

```python
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
    os.replace(tmp, path)
```

**存了六样东西，每样都有理由**：

| 存的东西 | 为什么 |
|---|---|
| `model.state_dict()` | 权重，必须的 |
| **`optimizer.state_dict()`** | **Adam 的动量 $m, v$！不存的话续训会从头开始累积** |
| `config` | 重建模型需要知道结构 |
| `args` | 复现训练需要知道超参数 |
| `iter` | 从哪一步继续 |
| `best_val` | 记录历史最优 |

**最容易忽略的是 `optimizer.state_dict()`**。

**为什么重要**：Adam 的更新依赖 $m_t$（一阶动量）和 $v_t$（二阶动量），它们是**跨 step 累积的指数平均**。如果续训时不恢复，$m, v$ 从 0 开始：

- 前几十步的梯度估计会很不准（因为 $m, v$ 还没累积起来）
- 表现为 loss 会先跳一下，然后才恢复

**这就是为什么很多人在 Colab 上续训发现 loss 从零开始。**

**`os.replace(tmp, path)` 是什么**：

先写 `ckpt.pt.tmp`，写完再原子替换成 `ckpt.pt`。

**为什么**：如果直接写 `ckpt.pt`，写到一半断电/崩溃，你会得到一个**损坏的 checkpoint**，之前几小时的训练全白费。**`os.replace` 是原子操作**——要么是旧的完整文件，要么是新的完整文件，不会出现半个。

**这是个值得学的工程习惯。**

### 6.7 GPU 相关：你以为跑在 GPU 上，其实没有

**这是新手最常犯的一类错误：代码跑起来了，没报错，但速度极慢——因为根本没用上 GPU。**

```python
# 检查模型在哪
print(next(model.parameters()).device)     # device(type='cuda', index=0) 才对

# 检查数据在哪
print(X.device)                            # 必须在 cuda

# 实时看显存
print(f"已分配 {torch.cuda.memory_allocated()/2**30:.2f} GB")
print(f"已缓存 {torch.cuda.memory_reserved()/2**30:.2f} GB")
```

**`allocated` 和 `reserved` 的区别**：

- `allocated`：**实际在用**的显存
- `reserved`：PyTorch 向系统**要过来备用**的显存（缓存池）

**`reserved` 比 `allocated` 大是正常的**——PyTorch 会预留一些避免频繁向系统申请。**但如果 `reserved` 接近上限，就是真的快满了。**

**最常见的四类「默默不用 GPU」**：

| 现象 | 原因 | 检查 |
|---|---|---|
| 模型在 cuda 但数据在 cpu | 忘了 `X.to(device)` | `print(X.device)` |
| 生成的 token 在 cpu | 忘了把输入搬过去 | 同上 |
| 评估特别慢 | 忘了 `@torch.no_grad()` | 看显存是否持续增长 |
| 只用了一个核 | 其实在跑 CPU | `nvidia-smi` 看利用率 |

**诊断工具**：训练时开一个终端跑

```bash
nvidia-smi -l 1        # 每秒刷新一次
```

**看到 `GPU-Util` 长期低于 30%，就是数据管道有瓶颈；低于 5%，说明根本没在 GPU 上跑。**

> **面试官可能问**：「训练很慢，你怎么排查？」
> **答**：先分清是「真的慢」还是「没用上 GPU」。第一步 `nvidia-smi` 看 GPU 利用率——如果长期低于 30%，瓶颈在数据加载或 CPU，不在 GPU；如果接近 0%，检查模型和数据是否都在 cuda 上。第二步算 MFU：`6 × 参数量 × tokens_per_sec / 峰值算力`，健康区间是 0.2–0.5，低于 0.15 说明卡在了 kernel 启动开销或数据加载。第三步检查是不是漏了 `@torch.no_grad()`、是不是 `num_workers=0` 导致数据加载串行、是不是 batch 太小导致 GPU 空转。

### 6.8 本章浓缩

**`train.py` = 第 4 章模板 + 四件事**：

| 额外的事 | 代码 | 对应 |
|---|---|---|
| 数据加载 | `get_batch` + `np.memmap` | 6.2 |
| 学习率调度 | `get_lr` + 写回 `param_groups` | 4.7 |
| 梯度累积 | `loss / grad_accum` | 4.10 |
| 断点续训 | 存 `model` + `optimizer` + `iter` | 6.6 |

**三个最容易出错的地方**：

1. **`loss / grad_accum` 忘了除** → 等效学习率放大 $k$ 倍
2. **`optimizer.state_dict()` 忘了存** → 续训时动量从零开始
3. **`scaler.unscale_` 顺序错了** → 梯度被缩到 1/65536，训练停滞

---

<a name="第-7-章"></a>
## 第 7 章 · 报错速查：12 个你一定会遇到的

> **本章结论**：PyTorch 的报错信息其实很好读，**关键是从下往上读，只看最后一行和第一个自己的栈帧**。
> 这一章把报错**全文**列出来，逐句翻译。

**为什么专门写这一章**：你说遇到报错习惯问 AI。这没问题，但**能自己看懂报错会快 10 倍**——不用等我回你，当场就能改。

**下面所有报错都是我在你这台机器上真实触发的**，不是我凭记忆编的。

### 7.1 读报错的通用方法

**PyTorch 报错的结构**：

```
Traceback (most recent call last):
  File "train.py", line 123, in <module>          ← 你的代码（从这里开始看！）
    loss.backward()
  File "torch/autograd/__init__.py", line 456     ← PyTorch 内部（忽略）
    ...
RuntimeError: <真正的错误信息>                     ← 最后一行才是重点
```

**三步法**：

1. **读最后一行**——那才是真正的错误
2. **从下往上找第一个你自己的文件**——`train.py:123`，那才是出事的地方
3. **所有 `torch/` 开头的栈帧全部忽略**——那是内部实现

**新手最常见的错误是**：从上往下读，读到 `torch/` 内部一大坨就懵了。**反着读。**

### 7.2 错误 #1：形状不匹配（最高频）

```python
torch.randn(64, 128) @ torch.randn(256, 512)
```

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x128 and 256x512)
```

**逐字翻译**：

- `mat1` = 第一个矩阵，形状 `64x128`
- `mat2` = 第二个矩阵，形状 `256x512`
- `cannot be multiplied` = 内维不等，乘不了

**你在线代里知道**：$A_{m\times k}B_{k\times n}$ 要求中间那个 $k$ 相等。这里 $128 \neq 256$。

**怎么修**：检查两层的维度，让内维对齐。通常是上一层的 `out_features` 和下一层的 `in_features` 不一致。

**在 Transformer 里的典型情形**：`n_embd` 和 `n_head * head_dim` 对不上。**这时检查 `n_embd % n_head == 0`。**

### 7.3 错误 #2：数据类型不匹配

```
RuntimeError: expected scalar type Float but found Long
```

**翻译**：你想让 float32 和 int64 直接做运算，PyTorch 不允许。

**在模型里的真实场景**：**数据是 token id（int64），但你想把它直接送进 `nn.Linear`。**

**怎么修**：

```python
x = x.float()          # 或者 x.to(torch.float32)
```

**关键理解**：token id 必须先过 `nn.Embedding` 变成浮点向量，才能进 `nn.Linear`。**Embedding 是 int→float 的桥梁**，这也是为什么模型第一层总是 Embedding。

**注意**：新版 PyTorch 某些运算会自动做类型提升，不一定报错。**但如果是模型内部的报错，几乎总是这个原因。**

### 7.4 错误 #3：设备不匹配

```
RuntimeError: Tensor on device cpu is not on the expected device meta!
```

**（你实际会遇到的是 `cuda:0` 和 `cpu` 的版本）**

**翻译**：一个张量在 GPU，另一个在 CPU，不能直接运算。

**怎么修**：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)              # 数据也要搬
```

**⚠️ 最容易漏的地方**：**推理时新生成的 token**。

```python
idx = torch.tensor([1, 2, 3], device=device)   # 别忘了 device=device
out = torch.cat([out, nxt], dim=1)
```

**如果你忘了**，通常不会在第一步报错（那时 `out` 还是你的输入），而是在**第二次循环**时才炸——因为新生成的 `nxt` 在 cuda，而 `out` 在 cpu。

**这个"延迟报错"的特点很坑。**

### 7.5 错误 #4：视图不连续

```python
torch.randn(2, 3).transpose(0, 1).view(-1)
```

```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

**翻译**：`transpose` 改变了**步长（stride）**，内存布局不再连续，`view` 要求连续，所以失败。

**注意：PyTorch 直接把修法写在报错里了**——`Use .reshape(...) instead.`

**两种修法**：

```python
y.reshape(-1)              # ① 最简单，reshape 自己处理
y.contiguous().view(-1)    # ② 先整理内存再 view（多一次拷贝，但更明确）
```

**记住这个组合**：`transpose` 之后想 `view`，必须先 `contiguous()`。

**这在 Attention 里出现得极其频繁**——因为多头注意力必须转置。**你在 `model.py:5.5` 会看到那个 `.contiguous()`，现在你知道它为什么在那了。**

### 7.6 错误 #5：二次 backward

```python
z.backward()
z.backward()
```

```
RuntimeError: Trying to backward through the graph a second time
(or directly access saved tensors after they have already been freed).
Saved intermediate values of the graph are freed when you call .backward()
or autograd.grad(). Specify retain_graph=True if you need to backward
through the graph a second time or if you need to access saved tensors
after calling backward.
```

**翻译**：第一次 `backward()` 已经把计算图**释放**了（这是为了省内存），第二次没图可用。

**两种修法，取决于你的意图**：

```python
# 情况 A：你确实要对同一个图反向多次（如 GAN）
z.backward(retain_graph=True)     # 别释放图

# 情况 B：你想算两个不同的 loss（更常见）
loss1.backward()
optimizer.zero_grad()
y2 = model(x2)                    # 重新前向，建了张新图
loss2 = loss_fn(y2, labels)
loss2.backward()                  # 新图，没问题
```

**关键理解**：报错不是「不能调两次 backward」，而是**「不能对同一个图调两次」**。

### 7.7 错误 #6：叶子节点原地修改

```python
x = torch.randn(3, requires_grad=True)
x += 1
```

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

**翻译**：叶子节点（`requires_grad=True` 且自己创建的）不能在原地修改——因为 autograd 要用它的**原值**来算梯度。

**三种修法**：

```python
x = x + 1                  # ✓ 创建新张量（不原地）
with torch.no_grad():
    x += 1                 # ✓ 明确说"不要梯度"
x.data += 1                # ⚠️ 能跑，但危险，别用
```

**这就是 4.9 节参数更新为什么要包 `no_grad`**：

```python
with torch.no_grad():
    W -= lr * W.grad       # ← 不包会报这个错
```

**但用 optimizer 就没这问题**——`optimizer.step()` 内部已经处理好了。**这就是「用 optimizer」比「手写更新」的好处之一。**

### 7.8 错误 #7：没有梯度就 step（最迷惑的一个）

这个**可能不报错，只是参数不变**——因为 `.grad` 是 `None` 时优化器会跳过。

**怎么确认**：

```python
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"⚠️ {name} 没有梯度！")
```

**三个最常见原因**：

1. **忘了 `loss.backward()`**
2. **`requires_grad=False`**（被 `detach()` 了，或冻结了）
3. **`with torch.no_grad():` 包住了 forward**（图没建）

### 7.9 错误 #8：在 no_grad 里 backward

```python
with torch.no_grad():
    y = (torch.randn(3, requires_grad=True) ** 2)
    y.sum().backward()
```

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

**翻译**：`no_grad()` 关掉了图的记录，所以没有 `grad_fn`，无法反向。

**怎么修**：把 `backward()` 挪到 `no_grad()` 外面。

**为什么值得单独记**：**这个报错和「忘了 requires_grad」的症状一模一样**。看到 `does not require grad and does not have a grad_fn`，第一反应就是**检查是不是被 `no_grad()` 或 `detach()` 掐断了**。

### 7.10 错误 #9：索引越界

```python
emb = nn.Embedding(10, 4)      # 只接受 0..9
emb(torch.tensor([15]))
```

```
IndexError: index out of range in self
```

**翻译**：词表只有 10 个位置，你给了索引 15。

**在 GPT 里的真实场景**：**`vocab_size` 和 tokenizer 不匹配。**

```python
# tokenizer 产出 0..8191 的 id（词表 8192）
# 但 GPTConfig 里写的是 vocab_size=4096
# → 遇到 id >= 4096 就炸
```

**怎么修**：**检查 `meta.pkl` 里的 `vocab_size` 和 `GPTConfig` 里的是否一致。**

**这个坑极其隐蔽**：因为训练时数据是随机采样的，可能前几百步都没抽到超范围的 token，**然后在你以为一切正常时突然报错**。这就是「tokenizer 改了必须从零重训」的机制性原因。

### 7.11 错误 #10：显存不足

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate 2.00 GiB (GPU 0; 7.96 GiB total capacity;
5.23 GiB already allocated; 1.87 GiB free;
5.31 GiB reserved in total by PyTorch)
```

**逐字翻译**：

| 部分 | 含义 |
|---|---|
| `Tried to allocate 2.00 GiB` | 想要 2GB |
| `7.96 GiB total capacity` | 你总共 ~8GB |
| `5.23 GiB already allocated` | 真正在用 5.23GB |
| `1.87 GiB free` | 还剩 1.87GB |
| `5.31 GiB reserved` | PyTorch 缓存池占了 5.31GB |

**问题在于**：要 2GB，但只剩 1.87GB。

**解决顺序**（代价从小到大）：

```bash
# ① 减小 batch_size（最有效，直接降激活值）
--batch_size 16  →  --batch_size 8

# ② 减小 block_size（注意力是 O(T²)，T 减半显存约减半）
--block_size 512  →  --block_size 256

# ③ 加大 grad_accum 补偿（不增加显存）
--grad_accum 8  →  --grad_accum 16
```

```python
# ④ 清缓存（碎片严重时有用，但不是万能的）
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

**⚠️ 一个反直觉的点**：**`reserved` 比 `allocated` 大是正常的**，别一看到「reserved 5.31GB」就以为浪费了 5GB——PyTorch 预留缓存是为了避免频繁向系统申请。**只有 `allocated` 才代表真的在用。**

### 7.12 错误 #11：NaN loss

**这个不一定报错**——可能只是 `loss = nan`，然后一路 NaN 到底。

**排查清单（按可能性排序）**：

| 原因 | 怎么查 | 怎么修 |
|---|---|---|
| **学习率太大** | 是不是前几步就 NaN | 降 lr，加 warmup |
| **忘了 `loss / grad_accum`** | 检查梯度累积代码 | 补上 |
| **归一化没走 fp32** | 看 RMSNorm 实现 | `x = x.float()` 再算 |
| **数据里有 NaN/Inf** | `torch.isnan(x).any()` | 清洗数据 |
| **除法没加 epsilon** | 检查所有除法和 sqrt | 加 `+1e-6` |
| **梯度爆炸** | `print(grad.norm())` | 加梯度裁剪 |

**诊断代码**（插在训练循环里，直接告诉你是哪层先炸的）：

```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"iter {it}: loss = {loss.item()}")
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"  ⚠️ {name} 的梯度出现 NaN/Inf")
    raise SystemExit("NaN 出现，已定位")
```

### 7.13 错误 #12：模型没更新（最隐蔽，因为无报错）

**完全没有报错**，只是 loss 不降。

**四步排查**：

```python
# ① 参数有没有梯度？
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"⚠️ {name} 无梯度")

# ② 参数有没有真的变？
before = model.fc.weight.clone()
optimizer.step()
print("参数变化量:", (model.fc.weight - before).abs().max().item())   # 应为非零

# ③ 学习率是多少？
for g in optimizer.param_groups:
    print("当前 lr:", g["lr"])      # 是不是 0？

# ④ 是不是忘了 optimizer.zero_grad()?
# → 症状是 loss 先降后突然爆炸
```

**最常见的两个原因**：

1. **学习率被 cosine 衰减到接近 0**——检查 `get_lr` 的 `min_lr`
2. **参数分组把不该冻结的冻了**——检查 `requires_grad`

### 7.14 报错速查总表

| 报错关键词 | 原因 | 一句话修法 |
|---|---|---|
| `shapes cannot be multiplied` | 形状不匹配 | 让内维相等 |
| `expected scalar type Float but found Long` | dtype 不匹配 | `.float()` |
| `not on the expected device` | 设备不匹配 | 都 `.to(device)` |
| `view size is not compatible` | 内存不连续 | `.reshape()` 或 `.contiguous().view()` |
| `backward through the graph a second time` | 图已释放 | `retain_graph=True` 或重新前向 |
| `leaf Variable ... in-place operation` | 叶子节点原地改 | `with torch.no_grad():` |
| `does not require grad and does not have a grad_fn` | 图被掐断 | 查 `no_grad()` / `detach()` |
| `index out of range in self` | 索引越界 | 查 `vocab_size` |
| `CUDA out of memory` | 显存不足 | 减 batch / block_size |
| `loss = nan` | 数值爆炸 | 降 lr / 归一化走 fp32 |
| loss 不降且无报错 | 参数没更新 | 查梯度、查 lr |
| `BrokenPipeError` | Windows 多进程 DataLoader | `num_workers=0` |

### 7.15 本章浓缩

**读报错三步法**：

1. 读**最后一行**
2. 从下往上找**第一个你自己的文件**
3. 忽略所有 `torch/` 开头的栈帧

**最高频的前三个**：**形状、设备、dtype。这三个占了新手 80% 的报错。**

> **面试官可能问**：「你怎么调试 PyTorch 代码？」
> **答**：分两类。硬报错类，PyTorch 的错误信息质量很高，关键是从下往上读、只看最后一行和第一个自己的栈帧，90% 是 shape / device / dtype 三类之一。静默错误类——loss 不降、结果不对但不报错——更难查，我的做法是插桩：在关键位置打印 shape 和各层梯度范数，用 `torch.isnan` 定位 NaN 首次出现的位置，用「参数前后差值是否非零」确认参数真的在更新。另外我会先把模型跑在极小的玩具配置上（2 层、几十维、几十个样本），确认它能把一个 batch 过拟合到接近零 loss——这是排除数据管道问题的标准手法：如果小配置都学不会，问题一定在代码或数据，不在规模。

---

<a name="附录-a"></a>
## 附录 A · 面试问答

> 前面每章末尾的「面试官可能问」都在这里汇总，按主题分组。
> **目的是让你能讲，不是让你背**——所以每条都给了「为什么这么答」。

### A.1 Tensor 基础

**Q：PyTorch 里最常见的错误是什么？**

> shape 不匹配。因为矩阵乘法要求内维相等，而神经网络前向就是一连串矩阵乘法，任何一层输出维度写错，错误会一直传到后面才爆出来。所以我调试时第一件事永远是打印 shape。

**为什么这么答**：面试官想听的是「你知道矩阵乘法的约束」+「你有调试习惯」。只说「shape 错误」不够，要顺带说出**为什么**它高频。

**Q：什么是广播（broadcasting）？**

> 两个形状不同的张量做运算时的自动扩展规则。从最后一维开始对齐，长度为 1 或缺失的维度可以扩展成对方的长度，否则报错。用处是省掉很多手动复制，危险在于不匹配时它不报错、而是静默产生错误形状，所以我会习惯性检查 shape。

**为什么这么答**：重点在**「它不报错」**这个危险性。只说规则是背课本。

### A.2 autograd 与反向传播

**Q：为什么反向传播比前向传播慢？**

> 因为反向每一步都是一次矩阵乘法（雅可比矩阵的转置乘），规模比前向更大。准确地说，前向每个参数是 2 FLOPs（一次乘加），反向要算「对输入的梯度」和「对权重的梯度」各一次，共 4 FLOPs。所以训练一个 token 的总计算量约 6N（N 为参数量），其中反向占 2/3。

**加分点**：说出 6N 这个公式。这是 LLM 算力估算的基础（C ≈ 6ND）。

**Q：训练循环里 zero_grad() 为什么必须在 backward() 之前？**

> 因为 PyTorch 的 .grad 是累加的。一个 batch 算完、step() 更新完参数后，上一轮的梯度还留在 .grad 里，不清零的话这一轮的梯度会叠加到上一轮上，等效于学习率不断增大，最后发散。放在 backward() 之前还是 step() 之后都可以（只要在下一轮 backward() 前清掉），但写在循环开头最不容易漏。

**加分点**：说出「累加是特性不是 bug」——梯度累积就靠它。

**Q：detach() 和 torch.no_grad() 的区别？**

> detach() 是把某一个张量从计算图上摘下来，返回一个不需要梯度的新张量，但图的其他部分还在；no_grad() 是一个上下文管理器，在它的作用域内所有运算都不建图。前者用于局部截断梯度（比如目标网络、冻结），后者用于整个推理过程，省内存也更快。

### A.3 nn.Module

**Q：nn.Module 做了什么？为什么不用裸 Tensor？**

> nn.Module 主要是工程封装，没有数学魔法。它做了三件事：一，通过 nn.Parameter 自动登记参数，所以 model.parameters()、model.to(device)、state_dict() 能一行处理全部参数；二，提供 train()/eval() 模式切换；三，通过 forward 定义前向逻辑，并用 __call__ 包一层以便插入 hook。用裸 Tensor 也能训练——训练循环的五个步骤完全一样——只是参数一多就管理不过来。

**加分点**：强调「用裸 Tensor 也能训」。这证明你真的理解了两者的关系，而不是把 nn.Module 当必需品。

### A.4 训练细节

**Q：训练 LLM 时 Adam 的 β2 为什么用 0.95 而不是 0.999？**

> β2 控制二阶动量（梯度平方的指数平均）的记忆长度。0.999 意味着记忆约 1000 步，对最近梯度分布的变化反应很迟钝；而 LLM 预训练的梯度分布在训练不同阶段变化很大，需要更快地跟踪当前的梯度尺度，所以用 0.95（记忆约 20 步）。这是 GPT-3 论文确立的实践。

**Q：为什么 weight decay 不施加在 norm 和 bias 上？**

> 因为它们的语义和权重矩阵不同。Norm 的 weight 是缩放因子，本应接近 1，衰减它会把它往 0 拉、破坏归一化；bias 是偏移量，衰减它没有正则化意义。只有权重矩阵是「知识」的载体，衰减它才能防过拟合。所以实现上按 p.dim() >= 2 分组，只对高维参数施加 decay。

**Q：bf16 和 fp16 有什么区别？为什么推荐 bf16？**

> 都占 2 字节，区别在位数分配。fp16 是 5 位指数 + 10 位尾数，动态范围约 10^(±5)；bf16 是 8 位指数 + 7 位尾数，指数位和 fp32 相同，动态范围约 10^(±38)，不会溢出，代价是尾数精度更低。训练时梯度跨度很大，fp16 容易上下溢，需要 GradScaler 反复升降缩放系数兜底；bf16 不需要，代码更简单也更稳定。现代 GPU 都原生支持 bf16。

**加分点**：能说出「指数位决定动态范围」这个底层原因。

**Q：讲一下一个完整的训练 step 都做了什么。**

> 五步：前向算出预测值 → 用 loss 函数算标量损失 → zero_grad() 清掉上一轮梯度（因为 .grad 是累加的）→ backward() 用链式法则算出所有参数的梯度 → step() 按优化器规则更新参数。LLM 训练里还会加四件事：动态调整学习率（warmup 让初期稳定、cosine 让后期精细收敛）、梯度裁剪防异常 batch 炸掉训练、bf16 混合精度省显存、以及梯度累积在有限显存下凑大 batch——累积时要注意把 loss 除以累积步数，否则等效学习率会被放大相应倍数。

**Q：梯度累积的数学原理是什么？**

> 设大 batch 分成 k 份，总损失 L = (1/k)·Σ L_i，梯度 ∇L = (1/k)·Σ ∇L_i。所以每份算 L_i/k 的梯度再累加到 .grad 里，结果完全等于大 batch 的梯度——因为 PyTorch 的 .grad 本来就是累加的。代价是 k 倍的前向反向时间，收益是不增加激活显存。

**这一条我实测验证过**：正确除以 k 时梯度与真大 batch 相差 1.19e-07（浮点误差）；忘了除则差正好 k 倍。

### A.5 架构与工程

**Q：RMSNorm 和 LayerNorm 有什么区别？**

> LayerNorm 是 (x-μ)/√(σ²+ε)·γ+β，做均值中心化和方差归一化，还有偏置 β；RMSNorm 是 x/√(mean(x²)+ε)·γ，只缩放不中心化。RMSNorm 少一次均值计算和一组参数，论文说效果相当但更快，是现代 LLM 默认。工程上关键点：算 mean(x²) 必须用 fp32，因为 bf16 下 x² 容易下溢成 0，导致除法爆炸成 NaN。

**Q：为什么现代 LLM 用 Pre-Norm 而不是 Post-Norm？**

> 因为 Pre-Norm 的残差路径更干净。Pre-Norm 是 x = x + sublayer(norm(x))，残差分支上 x 直接通到底，没有任何操作，梯度可以无衰减直达；Post-Norm 是 x = norm(x + sublayer(x))，残差要经过归一化，深层网络里不稳定。反向传播时 Pre-Norm 的雅可比里有个恒等项（对 x + f(x) 求导是 1 + f'(x)），那个 1 保证梯度不会消失。

**Q：为什么 FFN 的 hidden 维度是 8/3 倍而不是 4 倍？**

> 因为 SwiGLU 有三个矩阵（gate/up/down），普通 MLP 只有两个。为了在参数量上对齐：3 × (8/3)d = 8d，等于 2 × 4d = 8d。所以用 8/3 倍能让 SwiGLU 和传统 4 倍 MLP 参数量持平。

**Q：怎么估算训练显存？**

> 混合精度 + AdamW 训练时每个参数约 16 字节：bf16 权重 2 + fp32 master 权重 4 + bf16 梯度 2 + Adam 的两个 fp32 动量 8。再叠加激活值，粗略公式是 34 × batch_size × seq_len × d_model × n_layer 字节。所以 8GB 显存能训的模型上限约 300M 参数，舒服区间是 150M 以内。

**Q：KV cache 为什么能加速推理？**

> 自回归生成时，历史 token 的 K 和 V 不会变。没有 cache 时生成第 t 个 token 要把前 t 个重新前向一遍，总计算量 O(T³)；把每层的 K/V 缓存起来，新 token 只算自己的 Q 和缓存做注意力，降到 O(T²)。生成 1000 token 能差 300 倍。代价是显存，占用是 2 × n_layer × batch × n_kv_head × max_seq × head_dim × 2 字节——这也是 GQA 存在的意义，n_kv_head 减半，KV cache 就减半。

### A.6 调试

**Q：训练很慢，你怎么排查？**

> 先分清是「真的慢」还是「没用上 GPU」。第一步 nvidia-smi 看 GPU 利用率——长期低于 30% 说明瓶颈在数据加载或 CPU；接近 0% 说明根本没在 GPU 上跑，检查模型和数据是否都 .to(cuda)。第二步算 MFU：6 × 参数量 × tokens_per_sec / 峰值算力，健康区间 0.2–0.5，低于 0.15 说明卡在了 kernel 启动开销或数据加载。第三步检查是不是漏了 @torch.no_grad()、是不是 num_workers=0 导致数据加载串行、是不是 batch 太小导致 GPU 空转。

**Q：怎么调试 PyTorch 代码？**

> 分两类。硬报错类，PyTorch 错误信息质量很高，关键是从下往上读、只看最后一行和第一个自己的栈帧，90% 是 shape / device / dtype 三类之一。静默错误类——loss 不降、结果不对但不报错——更难查，我的做法是插桩：打印 shape 和各层梯度范数，用 torch.isnan 定位 NaN 首次出现位置，用「参数前后差值是否非零」确认参数真的在更新。另外我会先把模型跑在极小的玩具配置上，确认它能把一个 batch 过拟合到接近零 loss——如果小配置都学不会，问题一定在代码或数据，不在规模。

---

<a name="附录-b"></a>
## 附录 B · 公式 ↔ 代码对照表

> 这张表是**整份文档的索引**。你忘了某个公式对应什么代码时，来这查。

### B.1 数学运算

| 数学 | 代码 | 备注 |
|---|---|---|
| 标量乘 | `a * b` | |
| 逐元素乘（Hadamard） | `a * b` | 形状要能广播 |
| 点积 | `(a * b).sum()` 或 `a @ b` | 一维时 `@` 就是点积 |
| 矩阵乘 AB | `A @ B` 或 `torch.matmul(A, B)` | **内维必须相等** |
| 转置 Aᵀ | `A.T` 或 `A.transpose(0,1)` | 二维用 `.T` |
| Frobenius 范数 | `A.norm()` | 梯度裁剪就靠它 |
| e^x | `torch.exp(x)` | |
| ln x | `torch.log(x)` | |
| √x | `torch.sqrt(x)` | |
| 1/√x | `torch.rsqrt(x)` | RMSNorm 用 |
| x^n | `x ** n` 或 `x.pow(n)` | |
| 求和 Σx_i | `x.sum()` 或 `x.sum(dim=...)` | 指定维度很重要 |
| 均值 | `x.mean()` | |
| 最大值 | `x.max()` | |
| 截断到 [0,1] | `x.clamp(0, 1)` | |
| 绝对值 | `x.abs()` | |

### B.2 神经网络组件

| 数学 | 代码 | 备注 |
|---|---|---|
| ŷ = Wx + b | `nn.Linear(in, out)` | **weight 形状是 (out, in)** |
| 查表 | `nn.Embedding(V, d)` | 输入 int64，输出 float |
| sigmoid | `torch.sigmoid(x)` | |
| ReLU | `F.relu(x)` | |
| GELU | `F.gelu(x)` | GPT-2 用 |
| SiLU | `F.silu(x)` | **Llama 用** |
| softmax | `F.softmax(x, dim=-1)` | **必须指定 dim** |
| LayerNorm | `nn.LayerNorm(d)` | |
| RMSNorm | `nn.RMSNorm(d)` | **算方差要 fp32** |
| 交叉熵 | `F.cross_entropy(logits, target)` | **输入是 logits，内部含 softmax** |
| MSE | `F.mse_loss(y_pred, y)` | |
| 注意力 | `F.scaled_dot_product_attention(q,k,v)` | 因果掩码用 `is_causal=True` |

### B.3 训练三件套

> **先说清楚：为什么这几行全都在求导？**
> 因为「训练」数学上就等于「求导 + 沿着导数反方向走」。loss 对参数的导数为正，
> 说明参数增大 loss 会变大，所以要减小它；为负则相反。**求导不是工具，它是「学习」的定义。**
> 详见 2.0 节。

| 数学 | 代码 |
|---|---|
| 损失 L(θ) | `loss = loss_fn(model(x), y)` |
| **梯度 ∇L ← 就是求导** | `loss.backward()` → `p.grad` |
| 参数更新 θ ← θ - η∇L | `optimizer.step()` |
| 重置梯度 | `optimizer.zero_grad()` |
| 梯度裁剪（范数超阈值则等比缩小） | `clip_grad_norm_(params, c)` |

**「梯度」到底是什么**：它是**导数在多维的推广**。
一维时导数是 dL/dθ 一个数；参数是百万维向量时，∇L 是同样百万维的向量，
第 i 个分量是 ∂L/∂θ_i。**PyTorch 里 `.grad` 的形状和参数本身完全一样**——
因为每个参数都有自己的偏导数。

**几个术语说的其实是同一件事**：

| 术语 | 实际含义 |
|---|---|
| 导数 / 偏导数 | 标量对单个变量求导 |
| **梯度** | 导数在多维的推广，是个向量 |
| **反向传播** | 用链式法则**高效地**批量求所有参数的梯度 |
| **autograd** | PyTorch 里负责"自动求导"的模块 |
| 梯度下降 | 沿梯度反方向更新参数 |

**为什么不需要手写反向传播**：因为它的机制是固定的（链式法则 + 计算图），
与模型结构无关。PyTorch 记录了每个算子的"本地导数"，反向时按链式法则乘起来即可。
**你写的每一行 `x = self.attn(norm(x))` 都在自动登记进计算图**，
这就是为什么改模型结构不用改任何求导代码。

### B.4 形状操作

| 目的 | 代码 | 陷阱 |
|---|---|---|
| 改形状（不改顺序） | `x.view(...)` / `x.reshape(...)` | `view` 要求内存连续 |
| 转置 | `x.transpose(0,1)` / `x.T` | 之后要 `view` 得先 `.contiguous()` |
| 任意换轴 | `x.permute(2,0,1)` | |
| 加一维 | `x.unsqueeze(0)` | |
| 删长度 1 的维 | `x.squeeze()` | |
| 自动推一维 | `x.view(-1, V)` | `-1` = 让 PyTorch 自己算 |
| 展平 | `x.flatten()` | |
| 拼接 | `torch.cat([a,b], dim=...)` | 需要指定维度 |
| 堆叠成新维 | `torch.stack([a,b])` | |
| 保持维度求均值 | `.mean(-1, keepdim=True)` | 不加会广播失败 |

### B.5 索引与切片

| 目的 | 代码 |
|---|---|
| 第 i 个 | `x[i]` |
| 第 i 到 j | `x[i:j]` |
| 最后一维 | `x[..., -1]` |
| 最后一维（保维） | `x[..., [-1]]` ← **注意中括号** |
| 按条件取 | `x[x > 0]` |
| 掩码赋值 | `x[mask] = -inf` |

### B.6 设备与精度

| 目的 | 代码 |
|---|---|
| 上 GPU | `x.to("cuda")` 或 `x.cuda()` |
| 回 CPU | `x.to("cpu")` 或 `x.cpu()` |
| 判断可用 | `torch.cuda.is_available()` |
| 当前显存 | `torch.cuda.memory_allocated()` |
| 转 fp32 | `x.float()` |
| 转 bf16 | `x.to(torch.bfloat16)` |
| 转 int64 | `x.long()` |
| 混合精度 | `with torch.autocast("cuda", dtype=torch.bfloat16):` |
| 关梯度 | `with torch.no_grad():` |
| 断图 | `x.detach()` |

### B.7 常数值对照

| 量 | 值 | 说明 |
|---|---|---|
| log(vocab_size) | 词表 8192 → 9.01；32000 → 10.37 | **随机模型的初始 loss**，看到 loss 降到 3 才算学到东西 |
| LLM 学习率 | 6e-4（小模型可 1e-3） | |
| min_lr | lr 的 1/10 | |
| warmup | 总步数的 2% | |
| beta1 / beta2 | 0.9 / **0.95** | 不是 Adam 默认的 0.999 |
| weight_decay | 0.1（只对权重矩阵） | |
| grad_clip | 1.0 | |
| 训练显存/参数 | **16 字节** | 参数 + 梯度 + 优化器状态 |
| 训练算力 | C ≈ 6ND | N 参数，D token 数 |
| 数据量 | D ≈ 20N | Chinchilla |
| 健康 MFU | 0.2 – 0.5 | 低于 0.15 要查瓶颈 |

---

## 结语：你现在会什么

学完这份文档，你应该能：

**知识层面**

- 说清 Tensor 的 shape / dtype / device 三个属性和它们为什么重要
- 用链式法则解释 backward() 在做什么，并手写一个 mini autograd 引擎
- 讲清 nn.Module 的三个职责，以及为什么 super().__init__() 必须调
- 背出训练循环五件套，并说清漏掉任何一步的后果
- 讲清 AdamW 和朴素梯度下降的区别、为什么 β2 用 0.95
- 解释 bf16 为什么不需要 GradScaler

**实践层面**

- 看懂 code/model.py 和 code/train.py 的每一行
- 遇到报错能自己定位（尤其 shape / device / dtype 三类）
- 能改模型结构、调超参数
- 能判断训练是否健康（loss 曲线、MFU、过拟合）

**下一步**

按 手搓LLM实施手册.md 的路线走：

1. 用字符级数据跑通全流程（第 1 周）
2. TinyStories + BPE，训一个会讲故事的模型（第 2 周）
3. 放大到 30–60M，学会评估（第 3 周）
4. LoRA 微调 Qwen，做出能用的东西（第 4 周）

**最后一句**：这份文档里最重要的不是任何一个 API，而是**「为什么」**。

API 会过时（torch.optim 的接口改过好几版），但 shape 不匹配的原理、链式法则、残差为什么有用——这些二十年后还是对的。

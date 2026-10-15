"""
从零实现 byte-level BPE(Byte Pair Encoding) 分词器。
零依赖：只用 Python 标准库。教学用，对应手册第 6 章。

原理（一句话）：
  把文本先按 utf-8 拆成 256 个"字节"作为初始词表，
  然后反复统计"哪两个相邻 id 一起出现得最多"，把这一对合并成一个新 id。
  重复 vocab_size-256 次，词表就从 256 涨到 vocab_size。

为什么用 byte-level：
  任何字符串都能编码成 utf-8 字节，所以永远不会出现 OOV(未登录词)。
  代价是中文一个字通常占 3 个字节，所以中文要用更大的词表。

用法:
    python bpe.py            # 自带 demo，跑一遍训练+编码+解码

注意：这是朴素实现，时间 O(语料字节数 × 合并次数)。
      10MB 语料 train 到 4096 词表大约 1~3 分钟。
      真实项目请用 HuggingFace 的 tokenizers 库（C++ 实现，快 100 倍），见手册 6.3。
"""

import json
import re
from collections import Counter

# ---------------------------------------------------------------------------
# 预切分(pre-tokenization)
# ---------------------------------------------------------------------------
# 为什么不直接在整篇文本上做 BPE？
#   因为 BPE 只统计"相邻共现"，会在 "the" 和 " there" 之间把 'e'+' ' 合并掉，
#   导致 "the" 和 " there" 共享一个奇怪的 token，泛化很差。
#   GPT-2 的做法是先用正则把文本切成"词块"，BPE 只在词块内部合并。
#
# 这是 GPT-2 正则的 stdlib 版（原版用 \p{L}/\p{N}，需要 regex 三方库）：
#   [^\W\d_]+   == 字母（unicode 感知，不含数字/下划线）
#   \d+         == 数字
#   \s+(?!\S)   == 后面不跟非空白的空白（即行尾空白）
SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+|[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _get_stats(ids, counts=None):
    """统计 ids 列表中所有相邻 pair 的出现次数。可以累加到外部 counts 上。"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids, pair, new_id):
    """把 ids 中所有出现的 pair 替换成 new_id。"""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}   # {(int,int): int}  pair -> 合并后的新 id
        self.vocab = {}    # {int: bytes}      id  -> 对应的字节串

    # -- 训练 --------------------------------------------------------------
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "词表至少 256（基础字节）"
        num_merges = vocab_size - 256
        if num_merges == 0:
            self._build_base_vocab()
            return

        # 1) 预切分成词块，每个词块编码成 utf-8 字节 id 列表
        chunks = SPLIT_PATTERN.findall(text)
        ids = [list(c.encode("utf-8")) for c in chunks]
        ids = [c for c in ids if c]           # 丢掉空块
        print(f"[bpe] 预切分出 {len(ids)} 个词块, "
              f"{sum(len(c) for c in ids)} 个初始字节")

        self._build_base_vocab()

        # 2) 迭代合并
        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                _get_stats(chunk, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            # 每一对只在一个词块内合并，但统计是全局的
            ids = [_merge(chunk, pair, new_id) for chunk in ids]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and num_merges >= 10 and (i + 1) % max(1, num_merges // 10) == 0:
                print(f"[bpe] merge {i + 1}/{num_merges}  "
                      f"{pair} -> {new_id}  频率={stats[pair]}")

        if len(self.merges) < num_merges:
            print(f"[bpe] 注意：第 {len(self.merges)} 次合并后就没有新的相邻 pair 了。")
            print(f"      这是正常的 —— 说明语料里的词已经被合并成单个 token 了。")
            print(f"      想让词表长到 {vocab_size}，需要更丰富（更多样化）的语料。")

        print(f"[bpe] 训练完成: vocab_size={len(self.vocab)}, "
              f"merges={len(self.merges)}")

    def _build_base_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}

    # -- 编码 --------------------------------------------------------------
    def encode(self, text):
        """文本 -> id 列表。"""
        out = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids = list(chunk.encode("utf-8"))
            out.extend(self._encode_chunk(ids))
        return out

    def _encode_chunk(self, ids):
        """在一个词块内部，按训练时学到的顺序（rank 小的先合并）贪心合并。"""
        if len(ids) < 2:
            return ids
        # 每次找 rank 最小(最先学到)的可合并 pair
        while len(ids) >= 2:
            stats = _get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                          # 没有可合并的了
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    # -- 解码 --------------------------------------------------------------
    def decode(self, ids):
        """id 列表 -> 文本。用 errors='replace' 兜底，绝不会抛异常。"""
        data = b"".join(self.vocab.get(i, b"") for i in ids)
        return data.decode("utf-8", errors="replace")

    # -- 存取 --------------------------------------------------------------
    def save(self, path):
        payload = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab_size": len(self.vocab),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls()
        tok._build_base_vocab()
        for k, v in payload["merges"]:
            pair = (k[0], k[1])
            tok.merges[pair] = v
            tok.vocab[v] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok

    @property
    def vocab_size(self):
        return len(self.vocab)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 语料要"够丰富"词表才长得大：只有多样化的词，才会不断有新的 pair 可合并。
    # 如果只是把同一段话复制 100 遍，合并几十次就没事可干了（见下面的提示）。
    corpus = ("""
Once upon a time, there was a little girl named Lily.
Lily loved to play in the garden with her dog, Max.
One sunny day, Lily found a tiny bird with a broken wing.
She took the bird home and made a soft bed for it in a shoebox.
Every morning, Lily gave the bird seeds and fresh water.
Soon the bird got better and could fly again.
The bird sang a happy song to say thank you to Lily.
The garden was full of red roses, yellow sunflowers, and purple tulips.
Max chased the butterflies while Lily picked ripe strawberries.
Her mother baked a warm apple pie, and the whole kitchen smelled sweet.
Winter arrived, and the pond froze into a smooth, silver mirror.
Lily put on her woolen coat, knitted mittens, and a bright green scarf.
They built a snowman with a carrot nose and two coal-black eyes.
In spring, the birds returned and the cherry trees blossomed pink.
Lily learned to read, to write, and to count the stars at night.
每次翻开这本书，她都会想起那个安静的下午和窗外的阳光。
山脚下的小村庄里，住着一位会讲故事的老人。
他讲的故事里有会飞的小鸟，有会唱歌的小溪，还有一座古老的石桥。
孩子们围坐在火堆旁，听得入了迷，谁也不愿意回家。
夜深了，星星一颗一颗亮起来，风轻轻吹过金黄色的麦田。
""") * 30

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=512, verbose=True)

    for s in ["Once upon a time, there was a little girl.",
              "从前有座山，山里有座庙。",
              "hello world 123 !!!"]:
        ids = tok.encode(s)
        print(f"\n原文: {s}")
        print(f"ids : {ids}")
        print(f"还原: {tok.decode(ids)}")
        print(f"压缩率: {len(s)} 字符 -> {len(ids)} token")

    # 往返一致性检查：encode 之后 decode 必须还原（byte-level 的理论保证）
    test = "Hello, 世界! It's 2026 — emoji 🚀 too."
    assert tok.decode(tok.encode(test)) == test, "往返编码不一致!"
    print("\n[ok] 往返编码一致性验证通过")

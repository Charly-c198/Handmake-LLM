"""
唐诗语料清洗：剔除 chinese-poetry 原始数据里混入的校勘注。

为什么要做（这是"数据管道决定成败"的活例子）：
  chinese-poetry 的唐诗 JSON 里，除了诗句本身，还混有大量古籍校勘注释，例如：
      （「四夷」，劉云校作「黃子」。）
      （一作「青鸞舞鏡」）
      （見《四部叢刊》影印本《詩淵》第七冊第四七九頁）。
  如果不洗掉，模型会把它们当成"诗的一部分"一起学，生成时也会冒出来。

更麻烦的是：有些注释的左右括号被切到了不同段落里，例如
      ["或遭巴兒爭折取，或恐人工何足言。", "（「四夷」，劉云校作「黃子」。", "）身死無此頭，斷人須斷處。"]
  所以要跨段落跟踪"未闭合的括号"。

用法:
    python clean_poetry.py --src D:/llm-work/corpus/cp-repo/全唐诗 \
        --out D:/llm-work/corpus/tang_poems_clean.txt
"""

import argparse
import glob
import json
import os
import re

CJK = re.compile(r"[\u4e00-\u9fff]")          # 只要还剩汉字就算有效诗句
PAIR = re.compile(r"[（(][^）)]*[）)]")         # 成对的括号注释


def clean_paragraphs(paras):
    """逐段清洗，跨段跟踪未闭合的括号。返回清洗后的诗句列表。"""
    out, pending = [], False
    for x in paras:
        # 上一段留下了未闭合的「（」，先跳到本段的「）」
        if pending:
            idx = min([i for i in (x.find("）"), x.find(")")) if i != -1] or [-1])
            if idx == -1:
                continue                      # 整段都在注释里，丢弃
            x = x[idx + 1:]
            pending = False
        # 去掉成对的括号注释
        x = PAIR.sub("", x)
        # 出现孤立的「（」→ 从这里截断，标记 pending
        for op in "（(":
            i = x.find(op)
            if i != -1:
                x, pending = x[:i], True
                break
        x = x.strip()
        if x and CJK.search(x):               # 还有汉字才保留（纯标点/空白丢弃）
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="chinese-poetry 的「全唐诗」目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="poet.tang.*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, args.glob)))
    if not files:
        raise SystemExit(f"没找到文件: {os.path.join(args.src, args.glob)}")

    poems, raw_paras, clean_paras = [], 0, 0
    for fp in files:
        for p in json.load(open(fp, encoding="utf-8")):
            raw = [x.strip() for x in p.get("paragraphs", []) if x.strip()]
            raw_paras += len(raw)
            body = clean_paragraphs(raw)
            clean_paras += len(body)
            if not body:
                continue
            title = (p.get("title") or "").strip()
            author = (p.get("author") or "").strip()
            head = f"《{title}》{author}".strip()
            poems.append(head + "\n" + "\n".join(body))

    text = "\n\n".join(poems)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[clean] 文件 {len(files)} 个，诗 {len(poems):,} 首")
    print(f"[clean] 段落 {raw_paras:,} → {clean_paras:,}"
          f"（剔除 {raw_paras - clean_paras:,}，{(raw_paras-clean_paras)/max(1,raw_paras)*100:.1f}%）")
    print(f"[clean] 字符数 {len(text):,}，输出 {args.out}")


if __name__ == "__main__":
    main()

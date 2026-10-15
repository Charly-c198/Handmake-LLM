"""
P6-1：LoRA 微调 Qwen2.5-0.5B-Instruct。对应手册第 9 章。

为什么用 LoRA 而不是全参微调：
  冻结原权重，只在旁边并联两个小矩阵 A(r×d) 和 B(d×r)，只训这两个。
  可训练参数从 5 亿降到几百万，显存和时间都大降，效果接近全参微调。

为什么不用 4-bit（bitsandbytes）：
  本机是 sm_120 (Blackwell)，bitsandbytes 在 Windows + sm_120 上不稳（手册 3.2）。
  Qwen2.5-0.5B 用 fp16 全精度加载只需约 1GB，8GB 显存完全够，所以绕开量化。

关键：loss mask
  只在 assistant 的回答部分算 loss，问题和 system 部分设成 -100。
  否则模型会学着去"生成用户的提问"——这是 SFT 最常见的错误（手册 9.4）。

用法:
    python finetune_lora.py --model D:/llm-work/models/Qwen2.5-0.5B-Instruct \
        --data D:/llm-work/data/sft_data.jsonl --out D:/llm-work/out/lora
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)


# ---------------------------------------------------------------------------
def build_example(tok, item, max_len):
    """把一条 {instruction, output} 变成带 loss mask 的训练样本。

    prompt 部分（system+user+assistant 前缀）不参与 loss，
    只有 assistant 的回复和结尾的 <|im_end|> 参与。
    """
    system = item.get("system", "你是一个乐于助人的助手。")
    user, answer = item["instruction"], item["output"]

    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\n{user}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    full = prompt + answer + "<|im_end|>\n"

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]

    full_ids = full_ids[:max_len]
    labels = list(full_ids)
    # 关键：把 prompt 部分（含 system/user）的 label 置为 -100，不算 loss
    n_mask = min(len(prompt_ids), len(labels))
    for i in range(n_mask):
        labels[i] = -100

    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/llm-work/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help="SFT 数据 jsonl，每行 {instruction, output}")
    ap.add_argument("--out", default="D:/llm-work/out/lora")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  model={args.model}")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 模型：fp16 加载（绕开 bitsandbytes）----
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---- LoRA ----
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # 全挂上，效果最好
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- 数据 ----
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda x: build_example(tok, x, args.max_len),
                remove_columns=ds.column_names, desc="tokenize")
    print(f"[data] {len(ds)} 条样本")
    assert len(ds) > 10, f"样本太少: {len(ds)}"
    split = ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[data] train {len(train_ds)} / val {len(eval_ds)}")

    # ---- 训练 ----
    os.makedirs(args.out, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator)

    print("[train] 开始 LoRA 微调 ...")
    trainer.train()

    # ---- 保存 LoRA adapter（很小，通常几 MB~几十 MB）----
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    print(f"[done] LoRA adapter 已保存 -> {adapter_dir}")
    print(f"       接着跑: python app.py --adapter {adapter_dir}")


if __name__ == "__main__":
    main()
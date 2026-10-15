"""
P6-2：Gradio 交互界面。对应手册第 9/10 章。

加载 Qwen2.5-0.5B 底座 + LoRA adapter，在浏览器里对话。
--no_lora 可以对比"微调前 vs 微调后"，这是 P6 验收标准的核心证据。

用法:
    python app.py --model D:/llm-work/models/Qwen2.5-0.5B-Instruct \
        --adapter D:/llm-work/out/lora/adapter
    python app.py --model ... --no_lora      # 对比：不用 adapter
"""

import argparse
import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = None
TOK = None
CFG = None


def load(args):
    global MODEL, TOK, CFG
    CFG = args
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    tag = "base（未微调）"
    if not args.no_lora and args.adapter and os.path.isdir(args.adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        tag = f"LoRA adapter: {os.path.basename(os.path.dirname(args.adapter))}"
    model.eval()
    TOK, MODEL = tok, model
    print(f"[load] {args.model} + {tag}")
    return tag


def _text_of(content):
    """从 Gradio 的 content 里取出纯文本。

    Gradio 6 会把消息内容规范化成多种形态，必须在喂给 chat template 前统一成字符串：
      - 第一次对话：str                -> "讲个故事"
      - 多轮对话：  list[TextMessage]  -> [TextMessage(text="讲个故事", type="text")]
      - 也可能是 dict                  -> [{"type":"text","text":"讲个故事"}]
    直接把 list 传进 chat template 会报
      TypeError: can only concatenate str (not "list") to str
    这正是"第一轮正常、第二轮就崩"的原因。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, (list, tuple)):
        return "".join(_text_of(x) for x in content)
    # Message/TextMessage 这类对象：优先取 text 属性
    return getattr(content, "text", None) or str(content)


@torch.no_grad()
def chat(message, history, temperature, top_p, max_new_tokens, system):
    """history 是 Gradio 6 的 messages 格式，元素可能是 dict 也可能是 Message 对象。"""
    msgs = [{"role": "system", "content": system or "你是一个乐于助人的助手。"}]
    for m in history or []:
        if isinstance(m, dict):
            role, content = m.get("role"), m.get("content")
        else:
            role, content = getattr(m, "role", None), getattr(m, "content", None)
        if role in ("user", "assistant"):
            msgs.append({"role": role, "content": _text_of(content)})
    msgs.append({"role": "user", "content": message})

    text = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = TOK(text, return_tensors="pt").to(MODEL.device)
    out = MODEL.generate(
        **ids,
        max_new_tokens=int(max_new_tokens),
        temperature=max(temperature, 1e-4),
        top_p=top_p,
        do_sample=temperature > 0,
        repetition_penalty=1.1,
        pad_token_id=TOK.pad_token_id or TOK.eos_token_id,
    )
    reply = TOK.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/llm-work/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="D:/llm-work/out/lora/adapter")
    ap.add_argument("--no_lora", action="store_true", help="不加载 adapter，用于对比")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    tag = load(args)

    from theme import build_theme, CSS

    with gr.Blocks(title="手搓 LLM · LORA 助手") as demo:
        # 标题区：小字宽字距的 eyebrow + 轻字重大标题（设计语言的做法）
        gr.HTML(
            '<div style="padding:2.5rem 0 1.5rem;">'
            '<div style="font-size:12px;letter-spacing:0.3em;text-transform:uppercase;'
            'color:var(--h-muted);margin-bottom:1.25rem;">HANDMADE LLM · P6</div>'
            '<div style="font-size:2rem;font-weight:300;letter-spacing:-0.02em;'
            'color:var(--h-ink);line-height:1.2;">手搓 LLM · LORA 助手</div>'
            '<div style="font-size:13px;letter-spacing:0.08em;color:var(--h-muted);'
            'margin-top:1rem;padding-top:1rem;border-top:1px solid var(--h-rule);">'
            f'底座 Qwen2.5-0.5B-Instruct &nbsp;·&nbsp; {tag}</div>'
            '</div>'
        )
        chatbot = gr.Chatbot(
            height=460, show_label=False,
            value=[{"role": "assistant",
                    "content": "在这里输入问题开始对话。\n\n"
                               "可以试试：什么是五言绝句？ / 写一首关于秋天的诗。"}],
        )
        with gr.Row():
            msg = gr.Textbox(placeholder="输入你的问题，回车发送。", scale=4,
                             show_label=False, container=False)
            send = gr.Button("发送", variant="primary", scale=1)
        with gr.Accordion("生成参数", open=False):
            system = gr.Textbox(value="你是一个乐于助人的助手。", label="System")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
            max_new = gr.Slider(32, 1024, value=256, step=32, label="max_new_tokens")
        # 清空按钮靠右、窄一些，不要占满整行（避免视觉笨重）
        with gr.Row():
            gr.Markdown("")
            clear = gr.Button("清空对话", variant="secondary", size="sm", scale=1)

        def respond(message, history, t, p, m, s):
            if not message.strip():
                return history, ""
            reply = chat(message, history, t, p, m, s)
            # 先把 history 归一化成纯 dict 列表，再追加。
            # 不能直接 `history + [...]`：Gradio 6 传进来的可能是
            # ChatbotDataMessages 对象或 Message 对象列表，相加后结构会坏掉。
            hist = []
            for item in history or []:
                if isinstance(item, dict):
                    hist.append({"role": item.get("role"),
                                 "content": _text_of(item.get("content"))})
                else:
                    hist.append({"role": getattr(item, "role", None),
                                 "content": _text_of(getattr(item, "content", None))})
            hist.append({"role": "user", "content": message})
            hist.append({"role": "assistant", "content": reply})
            return hist, ""

        msg.submit(respond, [msg, chatbot, temperature, top_p, max_new, system], [chatbot, msg])
        send.click(respond, [msg, chatbot, temperature, top_p, max_new, system], [chatbot, msg])
        clear.click(lambda: [], None, chatbot)

    # Gradio 6 把 theme/css 从 Blocks() 挪到了 launch()；传在 Blocks 里会告警且未来会失效
    demo.launch(server_port=args.port, share=args.share,
                theme=build_theme(), css=CSS)


if __name__ == "__main__":
    main()
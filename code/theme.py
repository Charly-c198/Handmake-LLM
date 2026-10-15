"""
Hald Atelier 风格主题（适配 Gradio）。

来源：hald-atelier-frontend-skill 的设计语言。
但原始 skill 面向 React + Tailwind 网站，本项目的界面是 Gradio（Python 生成），
所以这里只移植【能落地的部分】，不照搬不适用的。

移植过来的（设计语言的本质）：
  - 五色克制：暖纸 #f7f5f0 / 墨 #141414 / 灰 #6f6c64 / 发丝线 rgb(20 20 20/0.10)
  - 零圆角、零阴影、零渐变
  - 无强调色：Gradio 默认的蓝/橙色一律映射到墨色灰阶
  - 小字宽字距（eyebrow / label 的做法），大字轻字重
  - 强调靠字号、字重、留白，不靠颜色

没有移植的（不适用，见文件末"未移植清单"）：
  theme.css / Reveal.tsx / link-sweep / hero-image / Tailwind 类名 / verify_port.py
  —— 这些依赖可编辑的 CSS/JSX，Gradio 没有对应物。
"""

import gradio as gr
from gradio.themes.utils.colors import Color

# ---------------------------------------------------------------------------
# 五色盘：暖纸 / 墨 / 灰 / 发丝线 / 表面（表面只用于必要的气泡区分）
# ---------------------------------------------------------------------------
PAPER = "#f7f5f0"   # 页面底（浅色）
INK = "#141414"     # 主文字（浅色）
MUTED = "#6f6c64"   # 次要文字、元信息（浅色）
RULE = "rgb(20 20 20 / 0.10)"
WASH = "rgb(20 20 20 / 0.025)"

# 暗色模式：把同一套语言反相（暖墨底 + 纸色字），而不是另一套配色
D_BG = "#141414"          # 页面底（墨）
D_SURFACE = "#1e1d1b"     # 块 / 输入框 / bot 气泡
D_SURFACE_2 = "#262523"   # user 气泡（比底亮一档）
D_TEXT = "#f7f5f0"        # 主文字（纸）
D_MUTED = "#a8a39a"       # 次要文字
D_RULE = "rgb(247 245 240 / 0.14)"
D_WASH = "rgb(247 245 240 / 0.05)"


def _mono_palette(name, dark, light):
    """造一条从浅到深的单色阶，用作 primary/secondary，代替 Gradio 的蓝橙色。

    设计语言里没有强调色——所有"高亮"都用墨色深浅表达，所以这里
    把 primary_hue 也设成墨阶，默认的蓝色按钮自然消失。
    """
    return Color(
        name=name,
        c50=light, c100="#efede8", c200="#e0ddd6", c300="#c7c3ba",
        c400="#a8a39a", c500=MUTED, c600="#57544e", c700="#3d3b37",
        c800="#262523", c900=dark, c950="#0a0a0a",
    )


_INK = _mono_palette("ink", INK, PAPER)


def build_theme():
    """返回配好的 Gradio 主题对象。"""
    theme = gr.themes.Base(
        primary_hue=_INK,       # 去掉默认蓝 → 墨阶
        secondary_hue=_INK,     # 去掉默认橙 → 墨阶
        neutral_hue=_INK,
        radius_size=gr.themes.sizes.radius_none,   # 零圆角
        text_size=gr.themes.sizes.text_md,
        font=[
            gr.themes.GoogleFont("Inter"),
            "Source Han Sans SC", "PingFang SC", "Microsoft YaHei",
            "ui-sans-serif", "system-ui", "sans-serif",
        ],
    )
    # 主题级 token 覆盖（对应 skill 里的 tokens-first）
    # 设计语言只有一套配色，暗色模式是把它反相（暖纸 ↔ 暖墨）。
    _set_both(theme)
    return theme


def _set_both(theme):
    """一次 set() 设置全部 token（亮色值 + 暗色值）。

    ⚠️ Gradio 6.30 的 theme.set() 每次调用都会重新解析全部变量：分多次调用时，
    后面的调用会把前面刚设的覆盖冲掉（实测 background_fill_primary 会被打回默认
    white）。所以所有值必须攒进一个 dict、只调用一次 set()。

    设计语言只有一种配色（暖纸 ↔ 暖墨），暗色模式是把它反相：深底 + 浅字。
    这里显式给出两套值；未列出的 token 交给 Gradio 从墨阶 palette 推导，
    因为 palette 本身就是暖灰阶，推导出来的暗色也协调。
    """
    light = dict(
        body_background_fill=PAPER,
        body_text_color=INK,
        body_text_color_subdued=MUTED,
        block_background_fill=PAPER,
        block_border_color=RULE,
        block_label_background_fill=PAPER,
        block_title_background_fill=PAPER,
        panel_background_fill=PAPER,
        background_fill_primary=PAPER,
        background_fill_secondary=WASH,
        input_background_fill=PAPER,
        input_border_color=RULE,
        border_color_primary=RULE,
        border_color_accent=RULE,
        link_text_color=INK,
        link_text_color_hover=MUTED,
        link_text_color_active=INK,
        link_text_color_visited=MUTED,
        accordion_text_color=INK,
        table_text_color=INK,
        code_background_fill=WASH,
        table_even_background_fill=PAPER,
        table_odd_background_fill=WASH,
    )
    dark = dict(
        body_background_fill=D_BG,
        body_text_color=D_TEXT,
        body_text_color_subdued=D_MUTED,
        block_background_fill=D_SURFACE,
        block_border_color=D_RULE,
        block_label_background_fill=D_SURFACE,
        block_title_background_fill=D_SURFACE,
        panel_background_fill=D_SURFACE,
        background_fill_primary=D_BG,
        background_fill_secondary=D_WASH,
        input_background_fill=D_SURFACE,
        input_border_color=D_RULE,
        border_color_primary=D_RULE,
        border_color_accent=D_RULE,
        link_text_color=D_TEXT,
        link_text_color_hover=D_MUTED,
        link_text_color_active=D_TEXT,
        link_text_color_visited=D_MUTED,
        accordion_text_color=D_TEXT,
        table_text_color=D_TEXT,
        code_background_fill=D_WASH,
        table_even_background_fill=D_SURFACE,
        table_odd_background_fill=D_WASH,
    )
    # 与模式无关的 token（圆角、阴影），两种模式同值
    both = dict(
        block_radius="0px",
        block_shadow="none",
        button_large_radius="0px",
        button_small_radius="0px",
        checkbox_border_radius="0px",
        shadow_spread="0px",
    )
    values = dict(light)
    values.update(both)
    # 凡是带 *_dark 变体的 token：先默认暗色=亮色（圆角/阴影/以及 light 里未给暗值的）
    for k, v in list(values.items()):
        dk = f"{k}_dark"
        if dk in vars(theme):
            values[dk] = v
    # 显式暗色值覆盖上面的默认
    for k, v in dark.items():
        values[f"{k}_dark"] = v
    theme.set(**values)   # 唯一一次调用，避免互相覆盖


# ---------------------------------------------------------------------------
# 无法通过主题参数表达的部分，用一段 CSS 补足。
# 只做主题 API 做不到的事：字距、字重、以及聊天区的微调。
# ---------------------------------------------------------------------------
CSS = """
/* ---- Palette as CSS vars so every rule below adapts to light/dark ----------
   Light = warm paper; dark = the same language inverted (warm ink). */
.gradio-container {
  --h-bg: #f7f5f0;
  --h-surface: #ffffff;
  --h-surface-2: #ebe8e1;
  --h-ink: #141414;
  --h-muted: #6f6c64;
  --h-rule: rgb(20 20 20 / 0.10);
  --h-rule-soft: rgb(20 20 20 / 0.06);
  --h-rule-strong: rgb(20 20 20 / 0.16);
  --h-rule-focus: rgb(20 20 20 / 0.30);
  --h-wash: rgb(20 20 20 / 0.025);
  --h-chip: rgb(20 20 20 / 0.05);
}
.dark .gradio-container,
.dark.gradio-container {
  --h-bg: #141414;
  --h-surface: #1e1d1b;
  --h-surface-2: #262523;
  --h-ink: #f7f5f0;
  --h-muted: #a8a39a;
  --h-rule: rgb(247 245 240 / 0.14);
  --h-rule-soft: rgb(247 245 240 / 0.10);
  --h-rule-strong: rgb(247 245 240 / 0.26);
  --h-rule-focus: rgb(247 245 240 / 0.45);
  --h-wash: rgb(247 245 240 / 0.05);
  --h-chip: rgb(247 245 240 / 0.07);
}

/* Chinese-friendly stack: Inter for Latin, fallback for CJK */
.gradio-container, .gradio-container * {
  font-feature-settings: 'ss01', 'cv11';
  font-family: 'Inter', 'Source Han Sans SC', 'PingFang SC',
               'Microsoft YaHei', ui-sans-serif, system-ui, sans-serif;
}

/* Headings: light weight + negative tracking. No negative tracking for CJK (would cramp). */
.gradio-container h1, .gradio-container h2 {
  font-weight: 300;
  letter-spacing: -0.02em;
}
.gradio-container h3, .gradio-container h4 {
  font-weight: 400;
  letter-spacing: -0.01em;
}

/* Wide tracking for small caps type (skill's eyebrow/label), applied to labels and buttons */
.gradio-container label,
.gradio-container .label-wrap > span,
.gradio-container button {
  letter-spacing: 0.08em;
}

/* Buttons: the language has no filled buttons; emphasis comes from weight and space.
   So the primary button is ink-outlined, filled on hover, instead of solid black. */
.gradio-container button.primary {
  background: transparent;
  color: var(--h-ink);
  border: 1px solid var(--h-ink);
  box-shadow: none;
  font-weight: 400;
  letter-spacing: 0.08em;
  border-radius: 0;
  min-width: 88px;          /* narrower so it does not look heavy stretched full-width */
}
.gradio-container button.primary:hover {
  background: var(--h-ink);
  color: var(--h-bg);
  border-color: var(--h-ink);
}
.gradio-container button.secondary {
  background: transparent;
  color: var(--h-muted);
  border: 1px solid var(--h-rule-strong);
  box-shadow: none;
  border-radius: 0;
}
.gradio-container button.secondary:hover {
  background: var(--h-wash);
  color: var(--h-ink);
}/* Clear button: small, not full-width */
.gradio-container button.sm {
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  padding: 6px 18px;
  max-width: 140px;
}

/* ---- Chat area: must target Gradio 6's real DOM structure ---------------
   Measured via headless Edge DOM dump, the chat area structure is:
     <div class="block svelte-1plpy97">        ← outer shell
       <div class="wrapper svelte-1wizwbi">
         <div role="log" class="bubble-wrap svelte-kpz1">
   Svelte class names carry a build hash (svelte-xxxxx) that changes across Gradio versions,
   so we rely on semantic classes + attribute selectors instead.

/* Chat shell: force our bg (.block is Gradio's generic block class, covered too) */
.gradio-container [data-testid='chatbot'],
.gradio-container .bubble-wrap,
.gradio-container [role='log'][aria-label*='chat' i],
.gradio-container > .block,
.gradio-container .block.auto-margin {
  background: var(--h-bg) !important;
  border-color: var(--h-rule) !important;
}
/* Chat scroll container (the .block with a fixed height) */
.gradio-container .block[style*='height'] {
  background: var(--h-bg) !important;
}

/* Message bubbles: our tones only, zero radius, no shadows.
   Gradio nests the bubble as
     .message.bot > .message.panel-full-width > [data-testid='bot']
   and paints the OUTER .message.bot with its own colour — which showed up as a
   thick frame around the inner surface. So the colour/border goes on the outer
   container, and the inner surface stays transparent. */
.gradio-container .bubble-wrap .message,
.gradio-container .message-row .message,
.gradio-container [data-testid='user'],
.gradio-container [data-testid='bot'] {
  border-radius: 0 !important;
  box-shadow: none !important;
}
.gradio-container .message.bot {
  background: var(--h-surface) !important;     /* assistant: surface */
  border: 1px solid var(--h-rule) !important;
}
.gradio-container .message.user {
  background: var(--h-surface-2) !important;   /* user: one step off the bg */
  border: 1px solid var(--h-rule-soft) !important;
}
.gradio-container [data-testid='bot'],
.gradio-container [data-testid='user'] {
  background: transparent !important;
  border: none !important;
  color: var(--h-ink) !important;
}
/* Fallback: any message block still rendering with Gradio's own colour */
.gradio-container .bubble-wrap > * {
  color: var(--h-ink);
}

/* Empty-state / bot message text: keep it readable on the surface. */
.gradio-container [data-testid='bot'] .message-content,
.gradio-container [data-testid='bot'] .md,
.gradio-container .placeholder-content,
.gradio-container .placeholder-content * {
  color: var(--h-ink) !important;
  background: transparent !important;
}
/* Message body copy sits at the language's 15px / 1.9 rhythm */
.gradio-container [data-testid='bot'] .md p,
.gradio-container [data-testid='user'] .md p {
  font-size: 15px;
  line-height: 1.9;
}

/* Code chips & API-name pills (API docs drawer, settings) */
.gradio-container .post,
.gradio-container .inline-code {
  background: var(--h-chip) !important;
  color: var(--h-ink) !important;
  border: 1px solid var(--h-rule) !important;
  border-radius: 0 !important;
}

/* Icon buttons (chat header share/clear/copy, per-message copy): their colour is
   var(--block-label-text-color); pin them to our muted ink so they read on both. */
.gradio-container .icon-button {
  color: var(--h-muted) !important;
}
.gradio-container .icon-button:hover {
  color: var(--h-ink) !important;
}

/* Inputs: hairline border, our bg fill */
.gradio-container textarea,
.gradio-container input[type='text'] {
  border: 1px solid var(--h-rule) !important;
  box-shadow: none !important;
  background: var(--h-bg) !important;
  color: var(--h-ink) !important;
}
.gradio-container textarea:focus,
.gradio-container input[type='text']:focus {
  border-color: var(--h-rule-focus) !important;
  box-shadow: none !important;
  outline: none !important;
}
/* Input placeholder: quiet grey */
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: var(--h-muted) !important;
  opacity: 1;
}

/* Accordion / other blocks: hairlines instead of shadows */
.gradio-container .panel,
.gradio-container .block {
  box-shadow: none !important;
  border-color: var(--h-rule);
}

/* Gradio footer links: fade to muted, not default blue */
.gradio-container footer a,
.gradio-container .built-with a {
  color: var(--h-muted) !important;
  text-decoration: none;
}

/* Visible focus (required by the skill, and an a11y baseline) */
.gradio-container :focus-visible {
  outline: 1px solid var(--h-ink);
  outline-offset: 3px;
}

/* (see theme.py for rationale) */
@media (prefers-reduced-motion: reduce) {
  .gradio-container *, .gradio-container *::before, .gradio-container *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
"""


# ---------------------------------------------------------------------------
# 未移植清单（说明为什么不搬，而不是忘了）
# ---------------------------------------------------------------------------
# theme.css            -> Gradio 无全局样式表可 append；其 token 已用 Python 表达
# Reveal.tsx           -> Gradio 组件不支持挂自定义 JS
# .link-sweep          -> 需要自定义 class，Gradio 链接组件不可控
# .hero-image          -> 同 Reveal，且聊天界面无 hero 图
# Tailwind 类名        -> 项目不用 Tailwind
# verify_port.py       -> 它扫 CSS/JSX 源码，对 Gradio 生成的界面无效
# 索引列表/Header/Hero 蓝图 -> 那些是网站布局语法，聊天界面没有对应区域
# 零圆角是否应用到聊天气泡 -> 已应用（radius_size=none 全局生效）
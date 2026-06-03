# =============================================================================
# FinExplain-LLM — Premium Streamlit Dashboard
# Run: streamlit run dashboard/app.py
# =============================================================================
import json
import os
import streamlit as st
from plots import (
    plot_agreement_matrix, plot_top_tokens, compute_top_tokens, plot_probs,
    METHOD_LABELS, METHOD_SHORT,
)

st.set_page_config(page_title="FinExplain-LLM", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")

# ── Premium CSS (shared design system with Projects 1 & 2) ────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; background:#0a0e27; }
    #MainMenu, footer, header { visibility:hidden; }

    .hero-header {
        background:linear-gradient(135deg,#0a0e27 0%,#1a2238 50%,#151b3d 100%);
        border:1px solid #2d3561; border-radius:16px; padding:2.5rem 3rem;
        margin-bottom:2rem; position:relative; overflow:hidden;
    }
    .hero-header::before { content:''; position:absolute; inset:0;
        background:radial-gradient(circle at 30% 50%, rgba(199,146,234,0.10) 0%, transparent 60%); }
    .hero-header h1 { font-size:2.5rem; font-weight:700; color:#c792ea; margin:0 0 .5rem 0;
        letter-spacing:-.02em; position:relative; z-index:1; }
    .hero-header p { color:#8892b0; font-size:1.1rem; margin:0; position:relative; z-index:1; }

    .metric-card { background:rgba(26,34,56,.6); backdrop-filter:blur(10px);
        border:1px solid #2d3561; border-radius:12px; padding:1.5rem; text-align:center;
        transition:all .3s ease; min-height:150px; display:flex; flex-direction:column;
        justify-content:center; align-items:center; }
    .metric-card:hover { transform:translateY(-2px); border-color:#c792ea;
        box-shadow:0 8px 24px rgba(199,146,234,.12); }
    .metric-value { font-family:'JetBrains Mono',monospace; font-size:2.6rem; font-weight:600;
        margin:.4rem 0; line-height:1; }
    .metric-label { color:#8892b0; font-size:.72rem; text-transform:uppercase;
        letter-spacing:1.2px; font-weight:600; }
    .metric-delta { font-family:'JetBrains Mono',monospace; font-size:.8rem; margin-top:.4rem; }
    .m-primary{color:#64ffda;} .m-secondary{color:#c792ea;} .m-blue{color:#82aaff;}
    .m-success{color:#c3e88d;} .m-warning{color:#ffcb6b;} .m-danger{color:#ff5370;}

    .section-header { font-size:1.05rem; font-weight:600; text-transform:uppercase;
        letter-spacing:1.5px; color:#8892b0; border-left:4px solid #c792ea;
        padding-left:1rem; margin:2.2rem 0 1.3rem 0; }

    .info-card { background:linear-gradient(135deg,rgba(199,146,234,.08),rgba(130,170,255,.06));
        border:1px solid rgba(199,146,234,.2); border-radius:12px; padding:1.3rem; margin:1rem 0; }
    .info-card h4 { color:#c792ea; font-size:1rem; margin:0 0 .4rem 0; }
    .info-card p { color:#e4e9f7; font-size:.9rem; line-height:1.6; margin:0; }

    .token-box { background:#151b3d; border:1px solid #2d3561; border-radius:12px;
        padding:1.4rem 1.5rem; margin:.6rem 0; line-height:2.4; }
    .tok { display:inline-block; padding:.18rem .45rem; margin:.12rem .12rem;
        border-radius:6px; font-family:'JetBrains Mono',monospace; font-size:.95rem;
        color:#e4e9f7; }
    .method-tag { font-family:'JetBrains Mono',monospace; font-size:.75rem; color:#c792ea;
        text-transform:uppercase; letter-spacing:1px; margin-bottom:.3rem; display:block; }

    .badge { display:inline-block; padding:.3rem .7rem; border-radius:6px; font-size:.7rem;
        font-family:'JetBrains Mono',monospace; font-weight:600; text-transform:uppercase;
        letter-spacing:.5px; }
    .b-positive{background:rgba(195,232,141,.15);color:#c3e88d;}
    .b-negative{background:rgba(255,83,112,.15);color:#ff5370;}
    .b-neutral{background:rgba(255,203,107,.15);color:#ffcb6b;}
    .b-ok{background:rgba(195,232,141,.15);color:#c3e88d;}
    .b-no{background:rgba(255,83,112,.15);color:#ff5370;}

    .stTabs [data-baseweb="tab-list"] { gap:8px; background:#151b3d; padding:.5rem; border-radius:12px; }
    .stTabs [data-baseweb="tab"] { border-radius:8px; color:#8892b0; padding:.5rem 1.4rem; font-weight:500; }
    .stTabs [aria-selected="true"] { background:#c792ea; color:#0a0e27; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "results.json")


@st.cache_data
def load_results(path):
    with open(path) as f:
        return json.load(f)


# ── Token coloring ────────────────────────────────────────────────────────────

def tok_color(value, is_signed):
    """Return an rgba background for a normalized attribution value in [-1,1]."""
    v = max(-1.0, min(1.0, float(value)))
    if is_signed:
        if v >= 0:  # supports the prediction -> green
            a = min(0.85, abs(v) * 0.8 + 0.06)
            return f"rgba(120,230,140,{a:.3f})"
        else:       # argues against -> coral
            a = min(0.85, abs(v) * 0.8 + 0.06)
            return f"rgba(255,83,112,{a:.3f})"
    # unsigned (attention) -> teal intensity
    a = min(0.85, abs(v) * 0.8 + 0.06)
    return f"rgba(100,255,218,{a:.3f})"


def render_tokens(tokens, scores, is_signed):
    spans = []
    for tok, sc in zip(tokens, scores):
        bg = tok_color(sc, is_signed)
        safe = (tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        spans.append(f'<span class="tok" style="background-color:{bg};" '
                     f'title="{sc:+.2f}">{safe}</span>')
    return '<div class="token-box">' + " ".join(spans) + "</div>"


def main():
    if not os.path.exists(DATA_PATH):
        st.error("⚠️ **results.json not found.** Run colab_inference.py, then place it in dashboard/data/.")
        st.stop()

    R = load_results(DATA_PATH)
    meta = R["metadata"]
    classes = meta["class_names"]
    methods = meta["methods"]
    signed = meta.get("signed", {})
    samples = R["samples"]

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>🔍 FinExplain-LLM</h1>
        <p>Token-Level Explainability for Financial Sentiment Decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Experiment Configuration")
        st.markdown(f"**Model:** `{meta['model']}`")
        st.markdown(f"**Dataset:** `{meta['dataset']}`")
        st.markdown(f"**Classes:** {', '.join(classes)}")
        st.markdown(f"**Sentences:** {meta['num_samples']}")
        st.markdown(f"**Methods:** {', '.join(METHOD_SHORT.get(m, m) for m in methods)}")
        st.divider()
        st.markdown("### 🧭 How to read attributions")
        st.markdown(
            "<span style='color:#78e68c'>Green</span> = token pushed the model "
            "**toward** its prediction. <span style='color:#ff5370'>Coral</span> = "
            "pushed **against**. <span style='color:#64ffda'>Teal</span> (attention) = "
            "importance, unsigned. Darker = stronger.", unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    acc = sum(s["correct"] for s in samples) / max(len(samples), 1)
    agree_vals = [v for v in R["agreement"].values() if v is not None]
    mean_agree = sum(agree_vals) / len(agree_vals) if agree_vals else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card"><div class="metric-label">Sentences Explained</div>
        <div class="metric-value m-secondary">{meta['num_samples']}</div>
        <div class="metric-delta m-blue">financial sentiment</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card"><div class="metric-label">Model Accuracy</div>
        <div class="metric-value m-success">{acc:.0%}</div>
        <div class="metric-delta m-blue">on explained set</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card"><div class="metric-label">Attribution Methods</div>
        <div class="metric-value m-primary">{len(methods)}</div>
        <div class="metric-delta m-blue">IG · Attn · LOO</div></div>""", unsafe_allow_html=True)
    am = "—" if mean_agree != mean_agree else f"{mean_agree:.2f}"
    c4.markdown(f"""<div class="metric-card"><div class="metric-label">Mean Method Agreement</div>
        <div class="metric-value m-warning">{am}</div>
        <div class="metric-delta m-blue">Spearman ρ</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Explain a Sentence", "⚖️ Method Comparison",
        "🧩 Method Agreement", "📊 Top Tokens by Class",
    ])

    # ── Tab 1: single sentence, single method ─────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Token Attribution</div>', unsafe_allow_html=True)
        idx = st.slider("Select sentence", 0, len(samples) - 1, 0, key="t1")
        s = samples[idx]
        method = st.selectbox("Attribution method", methods,
                              format_func=lambda m: METHOD_LABELS.get(m, m), key="m1")

        true_b = "b-" + classes[s["true_label"]].lower()
        pred_b = "b-" + classes[s["pred_label"]].lower()
        ok = "b-ok" if s["correct"] else "b-no"
        st.markdown(
            f'<span class="badge {pred_b}">Predicted: {classes[s["pred_label"]]}</span> '
            f'<span class="badge {true_b}">True: {classes[s["true_label"]]}</span> '
            f'<span class="badge {ok}">{"✓ correct" if s["correct"] else "✗ wrong"}</span>',
            unsafe_allow_html=True)

        st.markdown(render_tokens(s["tokens"], s["attributions"][method],
                                  signed.get(method, True)), unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"""<div class="info-card"><h4>{METHOD_LABELS.get(method, method)}</h4>
                <p>{'Signed: green supports the prediction, coral argues against it.'
                    if signed.get(method, True)
                    else 'Unsigned importance (attention flow) — teal intensity shows how much attention reached each token.'}</p>
                </div>""", unsafe_allow_html=True)
        with right:
            st.plotly_chart(plot_probs(s["probs"], classes), use_container_width=True)

    # ── Tab 2: all methods stacked for one sentence ──────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Same Sentence, Every Method</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card"><h4>Do the methods agree?</h4>
            <p>The same sentence is shown under each attribution method. Where the highlighted
            words differ, the methods disagree about <em>why</em> the model decided as it did —
            the core question of trustworthy interpretability.</p></div>""", unsafe_allow_html=True)
        idx2 = st.slider("Select sentence", 0, len(samples) - 1, 0, key="t2")
        s2 = samples[idx2]
        st.markdown(f'<p style="color:#e4e9f7;font-size:1.05rem;">"{s2["text"]}"</p>',
                    unsafe_allow_html=True)
        for m in methods:
            st.markdown(f'<span class="method-tag">{METHOD_LABELS.get(m, m)}'
                        f'{" (signed)" if signed.get(m, True) else " (unsigned)"}</span>',
                        unsafe_allow_html=True)
            st.markdown(render_tokens(s2["tokens"], s2["attributions"][m], signed.get(m, True)),
                        unsafe_allow_html=True)

    # ── Tab 3: agreement matrix ───────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">How Much Do Explanations Agree?</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_agreement_matrix(R), use_container_width=True)
        rows = "".join(
            f"<tr><td style='padding:.4rem 1rem;color:#e4e9f7'>"
            f"{METHOD_SHORT.get(k.split('__')[0])} vs {METHOD_SHORT.get(k.split('__')[1])}</td>"
            f"<td style='padding:.4rem 1rem;font-family:JetBrains Mono;color:#ffcb6b'>"
            f"{'—' if v is None else f'{v:.3f}'}</td></tr>"
            for k, v in R["agreement"].items())
        st.markdown(f"""<div class="info-card"><h4>Interpretation</h4>
            <p>Low correlations mean the methods identify <em>different</em> tokens as important.
            This is a known and important finding — attention in particular often does not reflect
            causal feature importance. For trustworthy financial AI, this argues against relying on
            any single explanation method.</p>
            <table style="margin-top:.6rem;">{rows}</table></div>""", unsafe_allow_html=True)

    # ── Tab 4: top tokens per class ───────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">Tokens That Drive Each Sentiment</div>',
                    unsafe_allow_html=True)
        if "integrated_gradients" not in methods:
            st.info("Top-token aggregation uses Integrated Gradients, which was unavailable in this run.")
        else:
            cls = st.radio("Class", classes, horizontal=True)
            rows = compute_top_tokens(R, cls, min_count=2, top_k=12)
            if not rows:
                st.info(f"No token recurs at least twice in sentences predicted **{cls}** "
                        f"on this {meta['num_samples']}-sentence sample. Run more sentences "
                        f"(increase N_SAMPLES) for richer aggregate signal.")
            else:
                st.plotly_chart(plot_top_tokens(rows, cls), use_container_width=True)
                st.markdown("""<div class="info-card"><h4>How to read this</h4>
                    <p>Tokens that appear in at least two sentences the model labelled this class,
                    ranked by their <strong>mean</strong> Integrated-Gradients attribution — the words
                    that most consistently pushed FinBERT toward this sentiment. Watch for
                    non-semantic tokens (e.g. <code>https</code>): when a URL fragment ranks highly,
                    it signals the model is partly keying on a <em>data artifact</em> rather than
                    meaning — exactly the kind of spurious correlation an explainability audit should
                    surface before deployment.</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#8892b0;font-size:.85rem;'>"
        f"FinExplain-LLM | {meta['num_samples']} sentences | "
        f"<code style='color:#c792ea;'>{meta['timestamp']}</code></div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()

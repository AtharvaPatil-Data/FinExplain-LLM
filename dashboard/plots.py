# =============================================================================
# FinExplain-LLM — Plotly visualizations (premium dark theme)
# =============================================================================
import numpy as np
import plotly.graph_objects as go

BG = "#0a0e27"
PAPER_BG = "#151b3d"
CARD_BG = "#1a2238"
GRID = "#2d3561"
TEXT = "#e4e9f7"
MUTED = "#8892b0"
PRIMARY = "#64ffda"     # teal
SECONDARY = "#c792ea"   # purple
DANGER = "#ff5370"      # coral
WARNING = "#ffcb6b"     # gold
SUCCESS = "#c3e88d"     # green
BLUE = "#82aaff"
FONT_SANS = "Inter, -apple-system, system-ui, sans-serif"
FONT_MONO = "JetBrains Mono, Consolas, monospace"

METHOD_LABELS = {
    "integrated_gradients": "Integrated Gradients",
    "attention": "Attention Rollout",
    "leave_one_out": "Leave-One-Out",
}
METHOD_SHORT = {"integrated_gradients": "IG", "attention": "Attn", "leave_one_out": "LOO"}


def _base(**kw):
    layout = dict(
        plot_bgcolor=BG, paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_SANS, color=TEXT, size=12),
        margin=dict(l=60, r=40, t=60, b=50),
        hoverlabel=dict(bgcolor=CARD_BG, font_family=FONT_MONO, bordercolor=PRIMARY),
    )
    layout.update(kw)
    return layout


def plot_agreement_matrix(results: dict) -> go.Figure:
    """Symmetric Spearman agreement matrix between attribution methods."""
    methods = results["metadata"]["methods"]
    agree = results["agreement"]
    n = len(methods)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            key1 = f"{methods[i]}__{methods[j]}"
            key2 = f"{methods[j]}__{methods[i]}"
            val = agree.get(key1, agree.get(key2))
            v = float(val) if val is not None else np.nan
            M[i, j] = M[j, i] = v

    labels = [METHOD_SHORT.get(m, m) for m in methods]
    text = [[("—" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}") for j in range(n)] for i in range(n)]
    hover = [[(f"{labels[i]} (same method)" if i == j
               else (f"{labels[i]} vs {labels[j]}: "
                     + ("n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.3f}")))
              for j in range(n)] for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=M, x=labels, y=labels,
        zmin=0, zmax=1,
        colorscale=[[0.0, DANGER], [0.35, WARNING], [0.7, BLUE], [1.0, PRIMARY]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=18, family=FONT_MONO, color=BG),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title="Spearman", tickfont=dict(color=MUTED, family=FONT_MONO),
                      title_font=dict(color=TEXT)),
    ))
    fig.update_layout(**_base(
        title=dict(text="Method Agreement (Spearman of |importance|)",
                   font=dict(size=18, color=TEXT), x=0.5, xanchor="center"),
        xaxis=dict(side="bottom", tickfont=dict(color=TEXT, size=13, family=FONT_MONO)),
        yaxis=dict(autorange="reversed", tickfont=dict(color=TEXT, size=13, family=FONT_MONO)),
        height=380,
    ))
    return fig


# Common English stopwords + tokens that carry no sentiment signal.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "being", "at", "by", "with", "from", "as", "that",
    "this", "these", "those", "it", "its", "has", "have", "had", "will", "would",
    "after", "before", "but", "not", "no", "than", "then", "so", "if", "into",
    "out", "up", "down", "over", "under", "about", "their", "they", "them", "he",
    "she", "his", "her", "we", "our", "you", "your", "i", "s", "t", "re", "ll",
    "ve", "m", "d", "co", "amp", "via", "per", "inc", "ltd", "corp",
}


def compute_top_tokens(results: dict, class_name: str, min_count: int = 2, top_k: int = 12):
    """Recompute class-driving tokens from per-sentence Integrated-Gradients data.

    A token must appear at least `min_count` times in sentences predicted as
    `class_name`; tokens are ranked by MEAN attribution. This removes the
    single-occurrence ties produced by per-sentence normalisation, while keeping
    recurring tokens (including data artifacts like 'https') visible.
    Returns a list of (token, mean_score, count).
    """
    meta = results["metadata"]
    if "integrated_gradients" not in meta["methods"]:
        return []
    classes = meta["class_names"]
    if class_name not in classes:
        return []
    cls_idx = classes.index(class_name)

    agg = {}  # token -> [sum_score, count]
    for s in results["samples"]:
        if s["pred_label"] != cls_idx:
            continue
        ig = s["attributions"]["integrated_gradients"]
        for tok, sc in zip(s["tokens"], ig):
            t = tok.lower()
            if len(t) < 2:               # drop single chars
                continue
            if not any(c.isalpha() for c in t):  # drop pure numbers/punctuation
                continue
            if t in STOPWORDS:
                continue
            slot = agg.setdefault(t, [0.0, 0])
            slot[0] += float(sc)
            slot[1] += 1

    rows = [(t, v[0] / v[1], v[1]) for t, v in agg.items() if v[1] >= min_count]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_k]


def plot_top_tokens(rows, class_name: str) -> go.Figure:
    """Horizontal bar of recurring tokens that drive a class (mean IG attribution).

    `rows` is the output of compute_top_tokens: list of (token, mean_score, count).
    """
    color = {"Positive": SUCCESS, "Negative": DANGER, "Neutral": WARNING}.get(class_name, PRIMARY)
    if not rows:
        fig = go.Figure()
        fig.update_layout(**_base(height=360, title=dict(
            text=f"No tokens recur \u2265 2× in sentences predicted {class_name}",
            font=dict(color=MUTED, size=15), x=0.5, xanchor="center")))
        return fig

    toks = [f"{r[0]} (\u00d7{r[2]})" for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    counts = [r[2] for r in rows][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=toks, orientation="h",
        marker=dict(color=color, opacity=0.85, line=dict(color=color, width=1)),
        text=[f"{v:.2f}" for v in vals], textposition="outside",
        textfont=dict(family=FONT_MONO, color=color, size=12),
        customdata=counts,
        hovertemplate="<b>%{y}</b><br>mean attribution: %{x:.3f}"
                      "<br>occurrences: %{customdata}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title=dict(text=f"Recurring Tokens Driving \u2192 {class_name}",
                   font=dict(size=18, color=TEXT), x=0.5, xanchor="center"),
        xaxis=dict(title="Mean attribution (Integrated Gradients)", showgrid=True,
                   gridcolor=GRID, tickfont=dict(color=MUTED, family=FONT_MONO),
                   title_font=dict(color=TEXT, size=12)),
        yaxis=dict(tickfont=dict(color=TEXT, family=FONT_MONO, size=13)),
        height=420,
    ))
    return fig


def plot_probs(probs, class_names) -> go.Figure:
    """Small bar chart of class probabilities for the selected sentence."""
    colors = [{"Positive": SUCCESS, "Negative": DANGER, "Neutral": WARNING}.get(c, PRIMARY)
              for c in class_names]
    fig = go.Figure(go.Bar(
        x=class_names, y=probs,
        marker=dict(color=colors, opacity=0.85),
        text=[f"{p:.1%}" for p in probs], textposition="outside",
        textfont=dict(family=FONT_MONO, color=TEXT, size=12),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(**_base(
        title=dict(text="Predicted Probabilities", font=dict(size=15, color=TEXT),
                   x=0.5, xanchor="center"),
        xaxis=dict(tickfont=dict(color=TEXT, family=FONT_SANS, size=12)),
        yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor=GRID,
                   tickfont=dict(color=MUTED, family=FONT_MONO)),
        height=300, showlegend=False,
    ))
    return fig

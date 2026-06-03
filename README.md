# 🔍 FinExplain-LLM

> **Token-Level Explainability for Financial Sentiment Decisions — and a study of whether explanation methods agree**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B.svg)](REPLACE_WITH_YOUR_STREAMLIT_URL)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57+-FF4B4B.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-FinBERT-2ea44f.svg)](https://huggingface.co/ProsusAI/finbert)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**▶️ Try the live dashboard:** _REPLACE_WITH_YOUR_STREAMLIT_URL_

---

## 🔍 Overview

**FinExplain-LLM** explains *why* a financial language model makes each sentiment decision, at the level of individual words — and then asks a harder question: **do different explanation methods actually agree?**

It applies **three complementary token-attribution methods** to **FinBERT** (`ProsusAI/finbert`) on a financial-sentiment task, renders the results as colored token heatmaps, and quantifies how much the methods agree using rank correlation. Where Projects 1–2 in this portfolio *measured* model behaviour (adversarial safety, calibration), this project *opens the box* and inspects the reasoning — a direct step toward **interpretable, auditable** financial AI.

**Task:** Financial sentiment classification (Positive / Negative / Neutral).

---

## 🧠 The Three Attribution Methods

| Method | Family | Signed? | What it measures |
|--------|--------|---------|------------------|
| **Integrated Gradients** | Gradient-based (Captum) | Yes | Causal contribution of each token to the predicted probability |
| **Attention Rollout** | Architecture-native | No | How much attention flows to each token through all layers |
| **Leave-One-Out (LOO)** | Occlusion, model-agnostic | Yes | Drop in predicted-class probability when a token is masked |

Using three methods is deliberate: it lets us test **explanation faithfulness** — if independent methods disagree about which words matter, no single explanation can be taken at face value.

---

## 📊 Key Results

> Produced by `colab_inference.py` on 40 sentences from the Twitter Financial News Sentiment set; the dashboard renders everything live.

| Metric | Value |
|--------|-------|
| Sentences explained | 40 |
| Model accuracy (on explained set) | 67.5% |
| Attribution methods | 3 (IG · Attention · LOO) |
| Mean method agreement (Spearman) | **0.21** |

**Pairwise method agreement (Spearman of \|importance\|):**

| Method pair | Agreement |
|-------------|-----------|
| Integrated Gradients vs Attention | **0.10** (very low) |
| Integrated Gradients vs Leave-One-Out | **0.31** (low–moderate) |
| Attention vs Leave-One-Out | **0.21** (low) |

### Three findings worth discussing

1. **Individual explanations are often legible.** For *"Urban Outfitters stock down 9% after Q3 results"* (predicted Negative, 97.2%), Integrated Gradients attributes the decision almost entirely to the word **"down"** — the model's reasoning matches human intuition.

2. **The methods largely disagree.** Pairwise agreement is only 0.10–0.31. Integrated Gradients and Leave-One-Out (both measure effect on the prediction) agree most; attention (which measures information flow, not causal effect) agrees least — consistent with the *"Attention is not Explanation"* literature. **Implication: trustworthy financial AI should not rely on a single explanation method.**

3. **The audit surfaced a data artifact.** Aggregating attributions across sentences, the token **`https`** ranks among the strongest "Positive" drivers — the model partly keys on URL fragments from the tweet data rather than financial meaning. Detecting this kind of **spurious correlation** is exactly what an explainability audit is for, and it's a data-quality red flag you'd want *before* deployment.

---

## 🚀 Quick Start

### 1. Generate explanations on Google Colab

1. Open [Google Colab](https://colab.research.google.com) → **Runtime → Change runtime type → T4 GPU**
2. Upload `colab_inference.py`
3. Install dependencies (**do not** upgrade torch/transformers — Colab's builds work):
   ```python
   !pip install -q captum datasets scikit-learn scipy
   ```
4. Run:
   ```python
   !python colab_inference.py
   ```
5. Download results:
   ```python
   from google.colab import files
   files.download("results.json")
   ```
6. Place `results.json` in `dashboard/data/`.

**Runtime:** ~2–4 minutes on a T4.

### 2. Launch the dashboard locally

```bash
git clone https://github.com/AtharvaPatil-Data/FinExplain-LLM.git
cd FinExplain-LLM
pip install -r requirements.txt
# ensure results.json is in dashboard/data/
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## 🧪 Methodology

### Model & Data
- **Model:** `ProsusAI/finbert` — BERT fine-tuned on financial text (3-class sentiment).
- **Data:** A financial-sentiment benchmark. The script tries the **Financial PhraseBank** first and falls back to the Parquet-native **Twitter Financial News Sentiment** dataset (used here), mapping Bearish/Bullish/Neutral → negative/positive/neutral.
- Loaded with **eager attention** (`attn_implementation="eager"`) so attention weights are exposed for rollout.

### Attribution details
- **Integrated Gradients:** Captum `LayerIntegratedGradients` on the embedding layer, 50 interpolation steps, PAD-token baseline; attributions summed over the embedding dimension to a per-token signed score.
- **Attention Rollout:** following Abnar & Zuidema (2020) — head-averaged attention, augmented with the identity and re-normalised, multiplied across layers; token importance read from the `[CLS]` row.
- **Leave-One-Out:** each token masked in turn; importance = drop in the predicted-class probability.

### Token handling
WordPiece sub-tokens (e.g. `un` + `##profit` + `##able`) are **merged into whole words** with their attributions **summed**, so the heatmap reads in natural language. Scores are normalised per sentence (strongest token = ±1) for display only.

### Method agreement
For each sentence with ≥3 tokens, we compute the **Spearman rank correlation** between the absolute token importances of each method pair, then average across sentences. Ranking by absolute importance makes the signed (IG, LOO) and unsigned (attention) methods comparable.

### Aggregate token analysis
For each class, tokens appearing in **≥2** sentences predicted as that class are ranked by **mean** Integrated-Gradients attribution (stopwords and punctuation removed). This reveals consistent class drivers — and exposes artifacts like `https`.

---

## 📸 Dashboard

- **Explain a Sentence** — colored token heatmap for any sentence and method (green = supports the prediction, coral = argues against, teal = attention importance), with prediction/true/correct badges and class probabilities.
- **Method Comparison** — the same sentence under all three methods, stacked, so disagreement is visible at a glance.
- **Method Agreement** — the Spearman agreement matrix and interpretation.
- **Top Tokens by Class** — recurring tokens that most drive each sentiment, with occurrence counts.

*(Add screenshots to `screenshots/` and reference them here.)*

---

## 🎓 PhD Research Connection

This project targets the **interpretability** pillar of the Central Bank of Ireland PhD — *"Framework for Interpretable and Behavioural Risk Assessment of Intelligent Language Models."*

- **Interpretability:** token attributions make a financial model's decisions inspectable word-by-word.
- **Trustworthiness / faithfulness:** quantifying method (dis)agreement is evidence about how much any single explanation can be trusted — a prerequisite for using explanations in regulated settings.
- **Auditability:** the pipeline detected a spurious correlation (URL fragments driving sentiment), demonstrating explainability as a data-quality and risk-control tool.
- **Portfolio arc:** complements Project 1 (*what fails* — adversarial robustness) and Project 2 (*how confident to be* — calibration) with *why the model decides* — interpretability.

---

## 🔧 Technical Details

- **Attribution:** Captum (Integrated Gradients); attention rollout and leave-one-out implemented from scratch.
- **Compute:** Colab T4 GPU for inference; Streamlit (CPU, no matplotlib) for the dashboard — deploys cleanly to Streamlit Cloud.
- **Robustness:** the script pre-flight-tests Integrated Gradients and degrades gracefully to attention + LOO if it is unavailable, so a run never aborts midway.

---

## 📚 References

1. **Sundararajan, M., Taly, A., & Yan, Q. (2017).** *Axiomatic Attribution for Deep Networks.* ICML. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365) — Integrated Gradients
2. **Abnar, S., & Zuidema, W. (2020).** *Quantifying Attention Flow in Transformers.* ACL. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928) — Attention Rollout
3. **Jain, S., & Wallace, B. C. (2019).** *Attention is not Explanation.* NAACL. [arXiv:1902.10186](https://arxiv.org/abs/1902.10186)
4. **Wiegreffe, S., & Pinter, Y. (2019).** *Attention is not not Explanation.* EMNLP. [arXiv:1908.04626](https://arxiv.org/abs/1908.04626)
5. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* KDD — occlusion / local explanations
6. **Araci, D. (2019).** *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)
7. **Kokhlikyan, N. et al. (2020).** *Captum: A unified and generic model interpretability library for PyTorch.* [arXiv:2009.07896](https://arxiv.org/abs/2009.07896)

---

## 📂 Project Structure

```
FinExplain-LLM/
├── colab_inference.py          # Attribution engine: IG + attention rollout + LOO (run on Colab)
├── dashboard/
│   ├── app.py                  # Streamlit dashboard (token heatmaps, comparison, agreement)
│   ├── plots.py                # Plotly visualizations + token aggregation
│   └── data/
│       └── results.json        # Output of colab_inference.py
├── screenshots/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👤 Author

**Atharva Patil** — MSc Computing (Data Analytics), Dublin City University
PhD Applicant, Central Bank of Ireland PhD Programme
GitHub: [github.com/AtharvaPatil-Data](https://github.com/AtharvaPatil-Data)

---

## 🙏 Acknowledgments

- **Supervisors:** Dr. Lili Zhang (DCU), Prof. Tomás Ward (DCU), Prof. Robert Whelan (TCD)
- **Funding:** Central Bank of Ireland + Insight Centre for Data Analytics

---

## 📄 License

MIT License.

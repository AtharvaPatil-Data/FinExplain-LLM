# =============================================================================
# FinExplain-LLM: Token-Level Explainability for Financial Sentiment Decisions
# =============================================================================
# Run on Google Colab with T4 GPU (Runtime > Change runtime type > T4 GPU)
#
# Install dependencies first (run in a Colab cell):
#   !pip install -q captum datasets scikit-learn scipy
#
#   IMPORTANT: do NOT use `-U` / `--upgrade` and do NOT reinstall torch,
#   torchvision, or transformers. Colab ships working, CUDA-matched builds;
#   upgrading them breaks the GPU stack. captum is pure-Python and safe.
#
# This script explains WHY FinBERT makes each financial-sentiment prediction,
# using three complementary token-attribution methods:
#   1. Integrated Gradients  (gradient-based, Captum)        — signed
#   2. Attention Rollout     (architecture-native)           — unsigned
#   3. Leave-One-Out (LOO)   (occlusion, model-agnostic)     — signed
#
# It also measures how much the methods AGREE (Spearman rank correlation of
# token importances) — a faithfulness/robustness question central to trustworthy
# interpretability. Results are saved to results.json for the Streamlit dashboard.
# =============================================================================

import json
import numpy as np
import torch
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ProsusAI/finbert"
DATASET_CONFIG = "sentences_50agree"
OUTPUT_FILE = "results.json"
N_SAMPLES = 40            # sentences to explain (kept small: LOO + IG are per-token)
IG_STEPS = 50             # Integrated Gradients interpolation steps
MAX_LEN = 64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"[INFO] Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# ── Dataset (robust loader: PhraseBank -> Twitter fallback) ───────────────────

def load_financial_sentiment():
    """Return (texts, labels_int, dataset_label_names, dataset_id)."""
    for repo in ("financial_phrasebank", "takala/financial_phrasebank"):
        try:
            ds = load_dataset(repo, DATASET_CONFIG, trust_remote_code=True)["train"]
            names = ds.features["label"].names
            print(f"[INFO] Loaded Financial PhraseBank from '{repo}'")
            return list(ds["sentence"]), np.array(ds["label"]), names, f"{repo}/{DATASET_CONFIG}"
        except Exception as e:
            print(f"[WARN] PhraseBank via '{repo}' unavailable: {str(e)[:110]}")
    repo = "zeroshot/twitter-financial-news-sentiment"
    ds = load_dataset(repo)["train"]
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    names = ["Bearish", "Bullish", "Neutral"]
    print("[INFO] Loaded Twitter Financial News Sentiment (Parquet fallback)")
    return list(ds[text_col]), np.array(ds["label"]), names, repo

texts_all, labels_all, dataset_label_names, DATASET_ID = load_financial_sentiment()

# Stratified sample of sentences to explain
idx, _ = train_test_split(
    np.arange(len(texts_all)), train_size=min(N_SAMPLES, len(texts_all)),
    random_state=RANDOM_SEED, stratify=labels_all,
)
texts = [texts_all[i] for i in idx]
labels_ds = [int(labels_all[i]) for i in idx]
print(f"[INFO] Explaining {len(texts)} sentences")

# ── Model ─────────────────────────────────────────────────────────────────────
print("[INFO] Loading FinBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# eager attention is required for output_attentions=True (the default SDPA/flash
# implementations do not return attention weights, which breaks attention rollout)
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, attn_implementation="eager"
    ).to(DEVICE)
except Exception as e:
    print(f"[WARN] eager attn_implementation not accepted ({str(e)[:80]}); using default + config flag")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
model.eval()

model_id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
model_label_order = [model_id2label[i] for i in range(len(model_id2label))]
CLASS_NAMES = [n.capitalize() for n in model_label_order]
print(f"[INFO] Model labels: {model_label_order}")

# Align dataset labels (PhraseBank or Twitter) into FinBERT's index space
SENTIMENT_SYNONYMS = {
    "bearish": "negative", "bullish": "positive",
    "negative": "negative", "positive": "positive", "neutral": "neutral",
}
def canon(name):
    return SENTIMENT_SYNONYMS.get(name.lower(), name.lower())
dataset_to_model = {i: model_label_order.index(canon(n)) for i, n in enumerate(dataset_label_names)}
labels_aligned = [dataset_to_model[l] for l in labels_ds]

# Robustly locate the embedding module for Integrated Gradients
try:
    EMBEDDING_LAYER = model.base_model.embeddings
except AttributeError:
    EMBEDDING_LAYER = model.get_input_embeddings()

PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id
SPECIAL_IDS = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}

# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax_logits(input_ids, attention_mask):
    return torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)

def merge_subwords(tokens, scores):
    """Merge BERT '##' continuation tokens into whole words, summing their scores."""
    out_tok, out_sc = [], []
    for tok, sc in zip(tokens, scores):
        if tok.startswith("##") and out_tok:
            out_tok[-1] += tok[2:]
            out_sc[-1] += sc
        else:
            out_tok.append(tok)
            out_sc.append(float(sc))
    return out_tok, out_sc

def normalize(scores):
    """Scale by max absolute value so the strongest token is ±1 (display only)."""
    arr = np.asarray(scores, dtype=float)
    m = np.abs(arr).max()
    return (arr / m).tolist() if m > 1e-12 else arr.tolist()

# ── Attribution methods ───────────────────────────────────────────────────────

def attr_integrated_gradients(input_ids, attention_mask, target):
    """Captum LayerIntegratedGradients on the embedding layer. Signed per token."""
    from captum.attr import LayerIntegratedGradients

    def fwd(inp, mask):
        return softmax_logits(inp, mask)

    lig = LayerIntegratedGradients(fwd, EMBEDDING_LAYER)
    baseline = input_ids.clone()
    # baseline: replace non-special tokens with PAD
    for j in range(baseline.size(1)):
        if int(input_ids[0, j]) not in SPECIAL_IDS:
            baseline[0, j] = PAD_ID
    atts = lig.attribute(
        inputs=input_ids, baselines=baseline, target=int(target),
        additional_forward_args=(attention_mask,), n_steps=IG_STEPS,
        internal_batch_size=IG_STEPS,
    )
    # sum across hidden dim -> per-token signed scalar
    return atts.sum(dim=-1).squeeze(0).detach().cpu().numpy()

def attr_attention_rollout(input_ids, attention_mask):
    """Attention rollout (Abnar & Zuidema, 2020). Unsigned per token."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    if not out.attentions:
        raise RuntimeError(
            "Model returned no attentions. Load with attn_implementation='eager'."
        )
    mats = [a.mean(dim=1).squeeze(0) for a in out.attentions]  # avg heads -> [S,S] each
    S = mats[0].size(0)
    eye = torch.eye(S, device=mats[0].device)
    result = eye.clone()
    for A in mats:
        A_aug = A + eye
        A_aug = A_aug / A_aug.sum(dim=-1, keepdim=True)
        result = A_aug @ result
    return result[0].detach().cpu().numpy()  # importance flowing from [CLS]

def attr_leave_one_out(input_ids, attention_mask, target, base_prob):
    """Occlusion: drop in target-class prob when each token is masked. Signed."""
    scores = np.zeros(input_ids.size(1))
    with torch.no_grad():
        for j in range(input_ids.size(1)):
            if int(input_ids[0, j]) in SPECIAL_IDS:
                continue
            masked = input_ids.clone()
            masked[0, j] = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else PAD_ID
            p = softmax_logits(masked, attention_mask)[0, int(target)].item()
            scores[j] = base_prob - p  # positive => token supported the prediction
    return scores

# ── Pre-flight: confirm Integrated Gradients works before the full loop ───────
IG_OK = True
try:
    _enc = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    _ = attr_integrated_gradients(_enc["input_ids"], _enc["attention_mask"], 0)
    print("[INFO] Integrated Gradients: OK")
except Exception as e:
    IG_OK = False
    print(f"[WARN] Integrated Gradients unavailable ({str(e)[:120]}); continuing with attention + LOO")

METHODS = (["integrated_gradients"] if IG_OK else []) + ["attention", "leave_one_out"]
SIGNED = {"integrated_gradients": True, "attention": False, "leave_one_out": True}

# ── Main loop ─────────────────────────────────────────────────────────────────
samples = []
agree_pairs = {f"{a}__{b}": [] for i, a in enumerate(METHODS) for b in METHODS[i + 1:]}
class_token_attr = {name: {} for name in CLASS_NAMES}  # aggregate IG by class

for n, text in enumerate(texts):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]

    with torch.no_grad():
        probs = softmax_logits(input_ids, attn).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    base_prob = float(probs[pred])

    toks_full = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    real_pos = [j for j, t in enumerate(input_ids.squeeze(0).tolist()) if t not in SPECIAL_IDS]

    raw = {}
    if IG_OK:
        raw["integrated_gradients"] = attr_integrated_gradients(input_ids, attn, pred)
    raw["attention"] = attr_attention_rollout(input_ids, attn)
    raw["leave_one_out"] = attr_leave_one_out(input_ids, attn, pred, base_prob)

    # Slice to real tokens, then merge subwords (consistent token list across methods)
    real_tokens = [toks_full[j] for j in real_pos]
    merged_tokens = None
    attributions = {}
    for m in METHODS:
        sliced = [float(raw[m][j]) for j in real_pos]
        mt, ms = merge_subwords(real_tokens, sliced)
        merged_tokens = mt  # identical across methods
        attributions[m] = normalize(ms)

    # Method agreement: Spearman on absolute importances (need >=3 tokens)
    if merged_tokens and len(merged_tokens) >= 3:
        for key in agree_pairs:
            a, b = key.split("__")
            rho, _ = spearmanr(np.abs(attributions[a]), np.abs(attributions[b]))
            if not np.isnan(rho):
                agree_pairs[key].append(rho)

    # Aggregate: which tokens push toward the predicted class (IG, signed)
    if IG_OK:
        for tok, sc in zip(merged_tokens, attributions["integrated_gradients"]):
            key = tok.lower()
            class_token_attr[CLASS_NAMES[pred]][key] = class_token_attr[CLASS_NAMES[pred]].get(key, 0.0) + sc

    samples.append({
        "text": text,
        "tokens": merged_tokens,
        "true_label": int(labels_aligned[n]),
        "pred_label": pred,
        "correct": bool(pred == labels_aligned[n]),
        "probs": probs.tolist(),
        "attributions": attributions,
    })
    if (n + 1) % 10 == 0:
        print(f"[INFO] Explained {n + 1}/{len(texts)}")

# ── Aggregate agreement + top tokens ──────────────────────────────────────────
agreement = {k: (float(np.mean(v)) if v else None) for k, v in agree_pairs.items()}

top_tokens = {}
for name, d in class_token_attr.items():
    ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:12]
    top_tokens[name] = [[t, round(s, 4)] for t, s in ranked]

results = {
    "metadata": {
        "model": MODEL_NAME,
        "dataset": DATASET_ID,
        "class_names": CLASS_NAMES,
        "num_samples": len(samples),
        "methods": METHODS,
        "signed": SIGNED,
        "ig_steps": IG_STEPS,
        "timestamp": datetime.now().isoformat() + "Z",
    },
    "agreement": agreement,
    "top_tokens": top_tokens,
    "samples": samples,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[INFO] Saved {OUTPUT_FILE}")
print("=" * 60)
print("EXPLAINABILITY SUMMARY")
print("=" * 60)
print(f"  Model:        {MODEL_NAME}")
print(f"  Dataset:      {DATASET_ID}")
print(f"  Sentences:    {len(samples)}")
print(f"  Methods:      {METHODS}")
acc = np.mean([s['correct'] for s in samples])
print(f"  Accuracy:     {acc:.1%}")
print("  Method agreement (Spearman, |importance|):")
for k, v in agreement.items():
    print(f"    {k.replace('__',' vs '):40} {v if v is None else round(v,3)}")
print("=" * 60)
print("[INFO] Download results.json -> place in dashboard/data/results.json")

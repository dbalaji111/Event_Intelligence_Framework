"""
theme_cluster_pipeline.py
==========================
Dual topic modelling on a JSONL memory file:
  - LDA      : word-frequency topics (gensim)
  - BERTopic : text-driven semantic topics using TF-IDF embeddings
               (BERT embeddings are too domain-similar for this corpus)

Usage
-----
    python theme_cluster_pipeline.py \
        --jsonl  memory_2024_2026/df_major_memory_2024_2026.jsonl \
        --output clustered_output \
        --n_topics 30 \
        --min_topic_size 5

CLI flags
---------
    --jsonl           path to JSONL memory file (required)
    --output          output directory (default: clustered_output/)
    --n_topics        number of LDA topics (default: 30)
    --min_topic_size  BERTopic min events per topic (default: 5)

Dependencies
------------
    pip install numpy pandas tqdm gensim nltk bertopic umap-learn hdbscan scikit-learn
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg, imp in [
        ("gensim",       "gensim"),
        ("nltk",         "nltk"),
        ("bertopic",     "bertopic"),
        ("umap-learn",   "umap"),
        ("hdbscan",      "hdbscan"),
        ("scikit-learn", "sklearn"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(f"ERROR: missing packages — run:\n  pip install {' '.join(missing)}")

check_deps()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaMulticore
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STOPWORDS
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_STOP = {
    # generic article noise
    "said", "would", "could", "also", "one", "two", "three", "year", "new",
    "last", "first", "may", "use", "say", "make", "including", "according",
    "well", "still", "even", "much", "many", "since", "nan", "none", "time",
    "week", "month", "day", "today", "thursday", "monday", "friday",
    "however", "while", "after", "before", "among", "within", "without",
    "will", "more", "than", "that", "this", "have", "been", "from", "with",
    "there", "their", "which", "about", "were", "they", "when", "what",
    # oil terms that appear in every article — not useful for separation
    "price", "prices", "market", "markets", "oil", "energy", "global",
    "company", "companies", "barrel", "barrels", "production", "supply",
    "demand", "output", "crude", "brent", "percent", "billion", "million",
    "report", "forecast", "analyst", "analysts", "growth", "level", "rate",
    "higher", "lower", "average", "future", "industry", "trade", "world",
    "international", "increase", "decrease", "change", "continue", "remain",
    "quarter", "expected", "bbl", "approximately",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD JSONL
# ─────────────────────────────────────────────────────────────────────────────

def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", s)


def parse_embedding(raw) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=np.float32)
    if isinstance(raw, str):
        clean = strip_ansi(raw).replace("[", "").replace("]", "").replace("\n", " ")
        nums = []
        for t in clean.split():
            try:
                nums.append(float(t.rstrip(",")))
            except ValueError:
                pass
        if nums:
            return np.array(nums, dtype=np.float32)
    return None


def load_jsonl(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    records, embeddings = [], []
    bad_emb = 0

    print(f"[load] reading {path.name} ...")
    with path.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="  rows", unit=" events"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            emb = parse_embedding(row.pop("bert_embeddings", None))
            records.append(row)
            if emb is not None and len(emb) >= 100:
                embeddings.append(emb)
            else:
                embeddings.append(None)
                bad_emb += 1

    df = pd.DataFrame(records)
    print(f"[load] {len(df)} events  |  {bad_emb} missing/bad embeddings")

    valid = [e for e in embeddings if e is not None]
    if not valid:
        sys.exit("ERROR: no valid embeddings found")

    dim = int(np.median([len(e) for e in valid]))
    X = np.zeros((len(embeddings), dim), dtype=np.float32)
    for i, e in enumerate(embeddings):
        if e is not None:
            l = min(len(e), dim)
            X[i, :l] = e[:l]

    print(f"[load] embedding dim: {dim}")
    return df, X


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LDA
# ─────────────────────────────────────────────────────────────────────────────

def ensure_nltk():
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_for_lda(text: str, stop_words: set, lemmatizer) -> List[str]:
    text   = strip_ansi(str(text))
    text   = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if len(t) > 3
        and t not in stop_words
        and t not in _DOMAIN_STOP
    ]


def run_lda(texts: List[str], n_topics: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ensure_nltk()
    stop_words = set(stopwords.words("english")) | _DOMAIN_STOP
    lemmatizer = WordNetLemmatizer()

    print("[lda] tokenising ...")
    tokenised  = [clean_for_lda(t, stop_words, lemmatizer) for t in tqdm(texts, desc="  tokenise")]
    tokenised  = [t if len(t) >= 3 else ["unknown"] for t in tokenised]

    dictionary = corpora.Dictionary(tokenised)
    dictionary.filter_extremes(no_below=3, no_above=0.70, keep_n=15_000)
    bow_corpus = [dictionary.doc2bow(t) for t in tokenised]

    print(f"[lda] training  n_topics={n_topics} passes=15 ...")
    lda = LdaMulticore(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=15,
        workers=4,
        random_state=42,
        alpha="asymmetric",
        eta="auto",
    )

    topic_words  = [
        " · ".join(w for w, _ in lda.show_topic(i, topn=6))
        for i in range(n_topics)
    ]

    topic_matrix = np.zeros((len(bow_corpus), n_topics), dtype=np.float32)
    dominant     = np.zeros(len(bow_corpus), dtype=np.int32)
    for i, bow in enumerate(bow_corpus):
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        for k, v in dist.items():
            topic_matrix[i, k] = v
        dominant[i] = int(topic_matrix[i].argmax())

    print(f"[lda] done")
    return topic_matrix, dominant, topic_words


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BERTopic — TF-IDF based (correct approach for domain-specific corpora)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """
    Build TF-IDF embeddings from cleaned text.
    These capture word-level distinctions far better than BERT embeddings
    for a corpus where all articles are in the same domain (oil/geopolitics).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print("[bertopic] building TF-IDF embeddings from text ...")

    extra_stop = list(_DOMAIN_STOP) + list(stopwords.words("english"))

    tfidf = TfidfVectorizer(
        stop_words=extra_stop,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.75,
        max_features=10_000,
        sublinear_tf=True,      # log(1+tf) — reduces impact of very frequent terms
    )

    ensure_nltk()
    X_tfidf = tfidf.fit_transform(texts)
    print(f"[bertopic] TF-IDF shape: {X_tfidf.shape}")

    # Reduce to 200 dims with SVD (like LSA) before UMAP
    # This removes noise and speeds up UMAP significantly
    n_components = min(200, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    X_reduced = normalize(X_reduced, norm="l2")
    print(f"[bertopic] SVD reduced to {X_reduced.shape[1]} dims  "
          f"(variance explained: {svd.explained_variance_ratio_.sum():.1%})")

    return X_reduced.astype(np.float32)


def run_bertopic(
    texts: List[str],
    min_topic_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:

    n = len(texts)
    print(f"[bertopic] corpus={n} events  min_topic_size={min_topic_size}")

    # Build TF-IDF embeddings instead of using BERT
    X_tfidf = build_tfidf_embeddings(texts)

    # UMAP on TF-IDF space — much more separable than BERT for this corpus
    n_neighbors = min(15, max(5, n // 30))
    print(f"[bertopic] UMAP n_neighbors={n_neighbors} n_components=15 ...")
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    extra_stop = list(_DOMAIN_STOP) + list(stopwords.words("english"))
    vectorizer = CountVectorizer(
        stop_words=extra_stop,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.75,
    )

    # c-TF-IDF boosts topic-distinctive words vs whole corpus
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )

    topic_ids, probs = topic_model.fit_transform(texts, embeddings=X_tfidf)
    topic_ids = np.array(topic_ids, dtype=np.int32)
    probs     = np.array(probs, dtype=np.float32)
    if probs.ndim == 2:
        probs = probs.max(axis=1)

    topic_info = topic_model.get_topic_info()
    label_map  = {}
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            label_map[tid] = "outlier"
        else:
            words = topic_model.get_topic(tid)
            label_map[tid] = " · ".join(w for w, _ in words[:5]) if words else f"topic_{tid}"

    n_real = len([t for t in topic_info["Topic"] if int(t) != -1])
    n_out  = int((topic_ids == -1).sum())
    print(f"[bertopic] {n_real} topics found  |  {n_out} outlier events")

    topic_labels = [label_map.get(int(tid), f"topic_{tid}") for tid in topic_ids]
    return topic_ids, probs, topic_labels, label_map


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MERGE & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save(
    df: pd.DataFrame,
    lda_dominant: np.ndarray,
    lda_topic_words: List[str],
    lda_matrix: np.ndarray,
    bert_topic_ids: np.ndarray,
    bert_probs: np.ndarray,
    bert_topic_labels: List[str],
    out_path: Path,
) -> pd.DataFrame:

    out = df.copy()
    out["lda_topic_id"]     = lda_dominant
    out["lda_topic_label"]  = [lda_topic_words[i] for i in lda_dominant]
    out["lda_topic_prob"]   = lda_matrix[np.arange(len(lda_dominant)), lda_dominant].round(4)
    out["bert_topic_id"]    = bert_topic_ids
    out["bert_topic_label"] = bert_topic_labels
    out["bert_topic_prob"]  = bert_probs.round(4)
    out["bert_is_outlier"]  = (bert_topic_ids == -1).astype(int)
    out["topics_agree"]     = (
        (out["lda_topic_prob"]  >= 0.15) &
        (out["bert_topic_prob"] >= 0.15) &
        (out["bert_is_outlier"] == 0)
    ).astype(int)
    out = out.drop(columns=["bert_embeddings"], errors="ignore")
    out.to_csv(out_path, index=False)
    print(f"[save] {out_path}  ({len(out)} rows, {len(out.columns)} cols)")
    return out


def save_topic_legend(lda_words: List[str], bert_label_map: dict, out_path: Path):
    rows = []
    for i, label in enumerate(lda_words):
        rows.append({"model": "LDA", "topic_id": i, "topic_label": label})
    for tid, label in sorted(bert_label_map.items()):
        rows.append({"model": "BERTopic", "topic_id": tid, "topic_label": label})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[save] topic legend -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 6.  POST-PROCESSING — merge duplicates, split oversized topics
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_topics(
    df: pd.DataFrame,
    bert_label_map: dict,
    oversized_threshold: float = 0.30,   # topic with >30% of corpus gets split
    min_topic_size: int = 5,
) -> pd.DataFrame:
    """
    Two fixes applied to bert_topic_id column:
    1. Merge near-duplicate topics (cosine similarity > 0.85 on label words)
    2. Flag oversized topics (>30% of corpus) — mark bert_topic_oversized=1
       so downstream you know to treat them as catch-alls
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    out = df.copy()
    n   = len(out)

    # ── 1. Detect near-duplicate BERTopic topics by label similarity ─────────
    tids   = [tid for tid in bert_label_map if tid != -1]
    labels = [bert_label_map[tid] for tid in tids]

    if len(labels) >= 2:
        vec = TfidfVectorizer(ngram_range=(1,1)).fit_transform(labels)
        sim = cosine_similarity(vec)

        merge_map = {}   # old_tid -> canonical_tid
        merged    = set()
        for i in range(len(tids)):
            if tids[i] in merged:
                continue
            for j in range(i + 1, len(tids)):
                if tids[j] in merged:
                    continue
                if sim[i, j] > 0.75:   # labels share >75% vocabulary → merge
                    merge_map[tids[j]] = tids[i]
                    merged.add(tids[j])
                    print(f"[postprocess] merge topic {tids[j]} "
                          f"('{labels[j]}') → topic {tids[i]} ('{labels[i]}')")

        if merge_map:
            out["bert_topic_id"] = out["bert_topic_id"].map(
                lambda x: merge_map.get(int(x), int(x))
            )
            # Rebuild label column
            reverse = {v: k for k, v in merge_map.items()}
            all_map  = {**bert_label_map, **{merge_map[k]: bert_label_map[merge_map[k]]
                                              for k in merge_map}}
            out["bert_topic_label"] = out["bert_topic_id"].map(
                lambda x: bert_label_map.get(int(x), f"topic_{x}")
            )
            print(f"[postprocess] {len(merge_map)} duplicate topics merged")
        else:
            print("[postprocess] no duplicate topics found")

    # ── 2. Flag oversized topics ──────────────────────────────────────────────
    counts = out["bert_topic_id"].value_counts()
    oversized = counts[counts / n > oversized_threshold].index.tolist()
    out["bert_topic_oversized"] = out["bert_topic_id"].isin(oversized).astype(int)
    if oversized:
        for tid in oversized:
            pct = counts[tid] / n * 100
            lbl = bert_label_map.get(int(tid), f"topic_{tid}")
            print(f"[postprocess] topic {tid} flagged oversized "
                  f"({counts[tid]} events, {pct:.0f}% of corpus): '{lbl}'")
    else:
        print("[postprocess] no oversized topics")

    # ── 3. Flag LDA singleton topics (< 3 events) ────────────────────────────
    lda_counts = out["lda_topic_id"].value_counts()
    singleton_lda = lda_counts[lda_counts < 3].index.tolist()
    out["lda_topic_singleton"] = out["lda_topic_id"].isin(singleton_lda).astype(int)
    if singleton_lda:
        print(f"[postprocess] {len(singleton_lda)} LDA singleton topics flagged "
              f"(ids: {sorted(singleton_lda)})")

    return out

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--jsonl",  required=True)
    ap.add_argument("--output", default="clustered_output")
    ap.add_argument("--n_topics", type=int, default=30,
                    help="number of LDA topics (default: 30)")
    ap.add_argument("--min_topic_size", type=int, default=5,
                    help="BERTopic min events per topic (default: 5)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    out_dir    = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        sys.exit(f"ERROR: file not found -- {jsonl_path}")

    df, X  = load_jsonl(jsonl_path)
    texts  = df["text"].fillna("").astype(str).tolist()

    print(f"\n{'─'*60}\n[1/2] LDA  n_topics={args.n_topics}\n{'─'*60}")
    lda_matrix, lda_dominant, lda_topic_words = run_lda(texts, args.n_topics)

    print(f"\n{'─'*60}\n[2/2] BERTopic  min_topic_size={args.min_topic_size}\n{'─'*60}")
    bert_ids, bert_probs, bert_labels, bert_label_map = run_bertopic(
        texts, min_topic_size=args.min_topic_size
    )

    print(f"\n{'─'*60}\n[3/3] Post-processing + Saving\n{'─'*60}")
    df_out = merge_and_save(
        df, lda_dominant, lda_topic_words, lda_matrix,
        bert_ids, bert_probs, bert_labels,
        out_dir / "df_topics.csv",
    )
    # Post-process: merge duplicate BERTopic topics, flag oversized + LDA singletons
    df_out = postprocess_topics(df_out, bert_label_map,
                                 oversized_threshold=0.30,
                                 min_topic_size=args.min_topic_size)
    df_out.to_csv(out_dir / "df_topics.csv", index=False)
    print(f"[save] updated df_topics.csv with post-processing flags")
    save_topic_legend(lda_topic_words, bert_label_map, out_dir / "topic_legend.csv")

    n_bert = len([t for t in bert_label_map if t != -1])
    n_oversized  = int(df_out.get("bert_topic_oversized", pd.Series([0])).sum()) if "bert_topic_oversized" in df_out else 0
    n_lda_single = int(df_out.get("lda_topic_singleton", pd.Series([0])).sum()) if "lda_topic_singleton" in df_out else 0
    print(f"\n{'='*60}")
    print(f"  events                   : {len(df)}")
    print(f"  LDA topics               : {args.n_topics}")
    print(f"  LDA singleton events     : {n_lda_single}  (lda_topic_singleton=1)")
    print(f"  BERTopic topics          : {n_bert}")
    print(f"  BERTopic outliers        : {int((bert_ids == -1).sum())}  (bert_is_outlier=1)")
    print(f"  BERTopic oversized events: {n_oversized}  (bert_topic_oversized=1)")
    print(f"  topics_agree=1           : {int(df_out['topics_agree'].sum())} events")
    print(f"\n  new columns added to df_topics.csv:")
    print(f"    lda_topic_id / lda_topic_label / lda_topic_prob")
    print(f"    lda_topic_singleton    <- 1 if topic has <3 events (noise)")
    print(f"    bert_topic_id / bert_topic_label / bert_topic_prob")
    print(f"    bert_is_outlier        <- 1 if BERTopic assigned to noise")
    print(f"    bert_topic_oversized   <- 1 if topic has >30% of corpus (catch-all)")
    print(f"    topics_agree           <- 1 if both models confident")
    print(f"\n  outputs in: {out_dir}")
    print(f"    df_topics.csv      <- one row per event")
    print(f"    topic_legend.csv   <- topic id to label reference")
    print(f"{'='*60}\n")

    print("LDA topics:")
    for i, label in enumerate(lda_topic_words):
        count = int((lda_dominant == i).sum())
        bar   = "█" * min(count, 40)
        print(f"  {i:>3} ({count:>3} events)  {bar}  {label}")

    print("\nBERTopic topics:")
    for tid, label in sorted(bert_label_map.items()):
        if tid == -1:
            continue
        count = int((bert_ids == tid).sum())
        bar   = "█" * min(count, 40)
        print(f"  {tid:>3} ({count:>3} events)  {bar}  {label}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAlex Topic Classification Pipeline (TXT abstracts)
=====================================================

This script performs end-to-end topic classification of scientific abstracts
using the OpenAlex Topics dataset.

Workflow:
  1) Download all OpenAlex Topics (if not present locally) to topics.jsonl
  2) Build or load topic embeddings (cached .npy)
  3) Read local .txt abstracts (one abstract per file)
  4) Classify with three methods:
       - Whole-abstract cosine similarity
       - Sentence-level aggregated cosine similarity
       - (Optional) LLM-generated topics, then mapped to OpenAlex via cosine similarity
  5) Fuse method scores into final_keywords
  6) Validate final_keywords against official OpenAlex topic names
  7) Export:
       - doi_keyword_ricgraph.csv (doi, keyword)
       - doi_keyword_ricgraph_with_ids.csv (doi, topic_display_name, topic_id)
       - classified_abstracts_openalex.xlsx (full table)
       - classified_abstracts_openalex_extended.xlsx (multi-sheet report)

Requirements:
  pip install sentence-transformers pandas numpy scikit-learn requests
  optional: pip install openpyxl llama-cpp-python
"""

import os
import re
import time
import json
import atexit
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== Configuration =====================
TEXTS_FOLDER     = "../texts"                      # Folder containing .txt abstracts
TOPICS_JSONL     = "topics.jsonl"                  # Local OpenAlex topics file
EMBEDDINGS_NPY   = "../topics_embeddings.npy"      # Cached topic embeddings
RESULT_XLSX      = "../classified_abstracts_openalex.xlsx"
RICGRAPH_CSV     = "results/doi_keyword_ricgraph.csv"
RICGRAPH_CSV_WITH_IDS = "results/doi_keyword_ricgraph_with_ids.csv"
RESULT_XLSX_EXT  = "results/classified_abstracts_openalex_extended.xlsx"

MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"  # consider "sentence-transformers/all-mpnet-base-v2"
TEXT_MODE        = "name+desc+labels"  # "name" | "name+desc" | "name+desc+labels"

API_URL          = "https://api.openalex.org/topics"
MAILTO           = "d.grotebeverborg@uu.nl"
PER_PAGE         = 200
SELECT_FIELDS    = "id,display_name,description,works_count,cited_by_count,updated_date,domain,subfield,field"
SLEEP_SEC        = 0.1

USE_LLM          = True                              # enable if you want LLM topics
LLM_PATH = os.path.join(BASE_DIR, "models", "llama-pro-8b-instruct.Q4_K_M.gguf")
LLM_MAX_TOK      = 150


# ===================== OpenAlex topics download =====================
def download_all_topics_jsonl(out_path: str) -> int:
    """
    Download all OpenAlex Topics to a JSONL file using cursor paging.

    Args:
        out_path: path to topics.jsonl

    Returns:
        Number of topics downloaded.
    """
    cursor = "*"
    total = 0
    pages = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "wb") as out:
        while True:
            params = {"per-page": PER_PAGE, "cursor": cursor, "select": SELECT_FIELDS, "mailto": MAILTO}
            print(f"Downloading topics page {pages+1} (cursor={cursor})")
            r = requests.get(API_URL, params=params, timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"OpenAlex API error HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for obj in results:
                out.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            total += len(results)
            pages += 1
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor
            time.sleep(SLEEP_SEC)

    print(f"Downloaded {total} topics → {out_path}")
    return total


# ===================== TXT abstracts loader =====================
_DOI_RX = re.compile(r"\b(10\.\d{4,9}/\S+)\b")

def _extract_doi_from_text(text: str) -> str | None:
    """
    Extract a DOI from free text, if present.

    Args:
        text: text content

    Returns:
        DOI string or None.
    """
    m = _DOI_RX.search(text)
    return m.group(1).lower() if m else None

def _doi_from_filename(fname: str) -> str:
    """
    Construct a DOI-like identifier from filename (best-effort).

    Args:
        fname: file name

    Returns:
        Lowercased identifier (may be DOI-like if filename encodes it).
    """
    base = os.path.splitext(fname)[0]
    base = re.sub(r"(_abstract|_abs|_txt)$", "", base, flags=re.IGNORECASE)
    parts = base.split("_")
    if len(parts) >= 2 and parts[0].startswith("10"):
        return (parts[0] + "/" + "".join(parts[1:])).lower()
    return base.lower()

def _strip_metadata_lines(lines: list[str]) -> str:
    """
    Remove common metadata lines and return the cleaned abstract body.

    Args:
        lines: list of raw lines

    Returns:
        Cleaned abstract string.
    """
    body = []
    for line in lines:
        ls = line.strip().lower()
        if ls.startswith("title:") or ls.startswith("doi:"):
            continue
        body.append(line.rstrip())
    text = "\n".join(body).strip()
    return re.sub(r"\n{3,}", "\n\n", text)

def load_abstracts_from_txt(folder: str, limit: int | None = None) -> list[dict]:
    """
    Load abstracts from .txt files in a folder.

    Args:
        folder: directory containing .txt files
        limit: optional max number of files to read

    Returns:
        List of dicts: [{"doi": str, "abstract": str}, ...]
    """
    records = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith(".txt"):
            continue
        full = os.path.join(folder, filename)
        try:
            with open(full, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue

        raw = "".join(lines)
        doi = _extract_doi_from_text(raw) or _doi_from_filename(filename)
        abstract = _strip_metadata_lines(lines).strip()
        if doi and abstract:
            records.append({
                "doi": doi.lower().replace("https://doi.org/", "").strip(),
                "abstract": abstract
            })
            if limit and len(records) >= limit:
                break
    print(f"Loaded {len(records)} abstracts from TXT files")
    return records


# ===================== Topics + embeddings =====================
def load_or_create_embeddings_topics(topics_path: str,
                                     model_name: str,
                                     embeddings_npy: str,
                                     text_mode: str = "name+desc+labels") -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """
    Load OpenAlex topics and build or load embeddings.

    Args:
        topics_path: path to topics.jsonl or CSV/TSV
        model_name: SentenceTransformer model name
        embeddings_npy: cache path for embeddings
        text_mode: "name", "name+desc", or "name+desc+labels"

    Returns:
        (topics_df, topic_names, topic_embeddings)
    """
    ext = os.path.splitext(topics_path)[1].lower()
    if ext == ".jsonl":
        df = pd.read_json(topics_path, lines=True, dtype=False)
    else:
        sep = "\t" if topics_path.endswith(".tsv") else ","
        df = pd.read_csv(topics_path, sep=sep, dtype=False)

    if "id" not in df.columns or "display_name" not in df.columns:
        raise ValueError("Expected 'id' and 'display_name' columns in topics.")

    if df["id"].duplicated().any():
        df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    for col in ["description", "domain", "field", "subfield"]:
        if col not in df.columns:
            df[col] = None

    def build_text(row: pd.Series) -> str:
        name = str(row.get("display_name", "") or "")
        desc = str(row.get("description", "") or "")
        dom  = str(row.get("domain", "") or "")
        fld  = str(row.get("field", "") or "")
        sub  = str(row.get("subfield", "") or "")
        if text_mode == "name":
            parts = [name]
        elif text_mode == "name+desc":
            parts = [name, desc]
        else:
            labels = " · ".join([x for x in [dom, fld, sub] if x])
            parts = [name, desc, labels] if labels else [name, desc]
        return " ".join(p.strip() for p in parts if p and str(p).strip())

    topic_texts = df.apply(build_text, axis=1).tolist()
    topic_names = df["display_name"].astype(str).tolist()

    if os.path.exists(embeddings_npy):
        X = np.load(embeddings_npy)
        if X.shape[0] != len(topic_texts):
            print("Embedding cache mismatch, regenerating...")
            X = _encode_and_save(topic_texts, model_name, embeddings_npy)
        else:
            print(f"Loaded topic embeddings from {embeddings_npy}")
    else:
        X = _encode_and_save(topic_texts, model_name, embeddings_npy)

    return df, topic_names, X

def _encode_and_save(texts: list[str], model_name: str, embeddings_npy: str) -> np.ndarray:
    """
    Encode a list of texts using a SentenceTransformer and save to .npy.

    Args:
        texts: list of strings
        model_name: model name
        embeddings_npy: output .npy path

    Returns:
        Embedding matrix (N, D).
    """
    print(f"Encoding {len(texts)} topics using '{model_name}'")
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    np.save(embeddings_npy, X)
    print(f"Saved embeddings to {embeddings_npy}")
    return X


# ===================== Cosine-based classifiers =====================
def classify_abstract_whole(abstract: str,
                            topic_names: list[str],
                            topic_embeddings: np.ndarray,
                            model: SentenceTransformer,
                            top_k: int = 5) -> list[tuple[str, float]]:
    """
    Compute cosine similarity using the full abstract text.

    Returns:
        List of (topic_name, score) pairs.
    """
    emb = model.encode([abstract], convert_to_numpy=True)
    sims = cosine_similarity(emb, topic_embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [(topic_names[i], float(sims[i])) for i in top_idx]

def simple_sentence_split(text: str) -> list[str]:
    """
    Split text into sentences with a simple regex heuristic.

    Returns:
        List of sentence strings.
    """
    return re.split(r'(?<=[.!?])\s+', text.strip())

def classify_abstract_by_sentences(abstract: str,
                                   topic_names: list[str],
                                   topic_embeddings: np.ndarray,
                                   model: SentenceTransformer,
                                   top_k: int = 5) -> list[tuple[str, float]]:
    """
    Aggregate cosine similarities over individual sentences in the abstract.

    Returns:
        Top-K topics with highest aggregated scores.
    """
    sentences = simple_sentence_split(abstract)
    topic_scores: dict[str, float] = {}
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        emb = model.encode([s], convert_to_numpy=True)
        sims = cosine_similarity(emb, topic_embeddings)[0]
        top_idx = sims.argsort()[-top_k:]
        for i in top_idx:
            topic = topic_names[i]
            topic_scores[topic] = topic_scores.get(topic, 0.0) + float(sims[i])
    return sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ===================== LLM topic extraction (optional) =====================
_llm = None

def _get_llm():
    """
    Load a local llama.cpp model once (if USE_LLM=True). Returns a cached instance or None.
    Prints diagnostics if loading fails.
    """
    global _llm
    if not USE_LLM:
        return None
    if _llm is not None:
        return _llm
    try:
        from llama_cpp import Llama
    except Exception as e:
        print(f"[LLM] llama_cpp import failed: {e}")
        return None
    try:
        print(f"[LLM] Loading model from: {LLM_PATH}")
        _llm = Llama(model_path=LLM_PATH, n_ctx=2048)
        print("[LLM] Model loaded.")
        return _llm
    except Exception as e:
        print(f"[LLM] Model load failed: {e}")
        _llm = None
        return None

def _close_llm():
    """
    Explicitly close the Llama instance to avoid destructor errors at interpreter shutdown.
    """
    global _llm
    try:
        if _llm is not None and hasattr(_llm, "close"):
            _llm.close()
    except Exception:
        pass
    finally:
        _llm = None

atexit.register(_close_llm)

def _parse_llm_csv(text: str) -> list[str]:
    """
    Parse a comma-separated line of topics into a clean list.

    Returns:
        List of non-empty topic strings.
    """
    if not isinstance(text, str):
        return []
    items = [t.strip().strip(",.;:") for t in text.split(",")]
    return [t for t in items if t]

def generate_llm_topics(abstract: str) -> list[str]:
    """
    Ask the local LLM for 1–10 concise research topics.
    Robust with tuned sampling, safer stops, and a retry.

    Returns:
        List of topic strings (possibly empty if LLM disabled/unavailable).
    """
    llm = _get_llm()
    if llm is None:
        return []

    prompt = f"""You are a scientific classification assistant.
Return ONLY a comma-separated list of 1 to 10 concise research topics (no extra words).
Use standardized academic phrasing (e.g., Medical informatics, Data privacy, Bayesian statistics).
Do not repeat or number items. Do not add explanations.

Abstract:
{abstract}

Topics:"""

    def _ask(stops):
        try:
            out = llm(
                prompt,
                max_tokens=LLM_MAX_TOK,
                stop=stops,
                temperature=0.2,
                top_p=0.9,
                top_k=50,
                repeat_penalty=1.1,
            )
            text = (out.get("choices", [{}])[0].get("text") or "").strip()
            return _parse_llm_csv(text)
        except Exception as e:
            print(f"[LLM] Inference error: {e}")
            return []

    topics = _ask(stops=["\n\n", "\nAbstract:", "\nTopics:"])
    if not topics:
        topics = _ask(stops=None)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in topics:
        tl = t.lower()
        if tl and tl not in seen:
            seen.add(tl)
            deduped.append(t)

    return deduped[:10]

def match_llm_topics_to_openalex(llm_topics: list[str],
                                 topic_names: list[str],
                                 topic_embeddings: np.ndarray,
                                 model: SentenceTransformer,
                                 top_k: int = 3) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Map LLM-generated topics to official OpenAlex topics using cosine similarity.

    Args:
        llm_topics: list of topic strings from the LLM
        topic_names: official OpenAlex topic names
        topic_embeddings: embedding matrix for topics
        model: SentenceTransformer for encoding queries
        top_k: number of nearest topics to return

    Returns:
        (top1_matches, topk_matches) where each is a list of (topic_name, score).
    """
    results_1: list[tuple[str, float]] = []
    results_3: list[tuple[str, float]] = []
    if not llm_topics:
        return results_1, results_3
    for topic in llm_topics:
        emb = model.encode([topic], convert_to_numpy=True)
        sims = cosine_similarity(emb, topic_embeddings)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        top_matches = [(topic_names[i], float(sims[i])) for i in top_idx]
        if top_matches:
            results_1.append(top_matches[0])
        results_3.extend(top_matches)
    return results_1, results_3


# ===================== Utilities =====================
def format_topic_matches(matches: list[tuple[str, float]]) -> str:
    """
    Format a list of (topic, score) pairs into a semicolon-separated string.
    """
    return "; ".join([f"{name} ({score:.2f})" for name, score in matches])

def truncate_text(text: str, max_words: int = 300) -> str:
    """
    Shorten a text to a maximum number of words.
    """
    return " ".join((text or "").split()[:max_words])

def parse_topic_score_column(col: str | None) -> list[tuple[str, float]]:
    """
    Parse a string like 'Topic (0.83); Topic2 (0.72)' into a list of pairs.

    Args:
        col: scored column string

    Returns:
        List of (topic, score) tuples.
    """
    out: list[tuple[str, float]] = []
    if isinstance(col, str):
        for part in col.split(";"):
            part = part.strip()
            if "(" in part and ")" in part:
                try:
                    topic, score = part.rsplit("(", 1)
                    out.append((topic.strip(), float(score.replace(")", ""))))
                except:
                    pass
    return out

def add_final_keywords_column(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Combine multiple scoring methods into a single final keyword list.

    Args:
        df: results dataframe with scored columns
        top_n: number of final keywords to keep

    Returns:
        DataFrame with a new 'final_keywords' column.
    """
    weights = {
        "top_topics_llm_matched_1a": 1.0,
        "top_topics_whole": 0.8,
        "top_topics_sentences": 0.6,
    }
    finals = []
    for _, row in df.iterrows():
        scores = defaultdict(float)
        for col, w in weights.items():
            for topic, sc in parse_topic_score_column(row.get(col, "")):
                scores[topic] += w * sc
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        finals.append("; ".join([t for t, _ in top]))
    df["final_keywords"] = finals
    return df

def validate_final_keywords(df: pd.DataFrame,
                            column: str,
                            official_names_set: set[str]) -> pd.DataFrame:
    """
    Keep only official OpenAlex topic names in final_keywords.

    Args:
        df: results dataframe
        column: column name to validate
        official_names_set: set of allowed names (lowercased)

    Returns:
        DataFrame with validated column.
    """
    cleaned = []
    for _, row in df.iterrows():
        names = [x.strip() for x in str(row.get(column, "")).split(";") if x.strip()]
        names_ok = [n for n in names if n.lower() in official_names_set]
        cleaned.append("; ".join(names_ok))
    df[column] = cleaned
    return df

def expand_final_keywords_with_ids(df: pd.DataFrame,
                                   topics_jsonl_path: str,
                                   keyword_column: str = "final_keywords",
                                   output_path: str = "doi_keyword_ricgraph_with_ids.csv") -> None:
    """
    Export DOI–topic pairs with OpenAlex topic IDs.

    Args:
        df: results dataframe
        topics_jsonl_path: path to topics.jsonl
        keyword_column: column containing topic names
        output_path: output csv path
    """
    topics_df = pd.read_json(topics_jsonl_path, lines=True)
    name_to_id = {str(n).lower(): str(i) for n, i in zip(topics_df["display_name"], topics_df["id"])}

    rows = []
    for _, row in df.iterrows():
        doi = str(row.get("doi") or "").strip()
        for name in str(row.get(keyword_column, "")).split(";"):
            nm = name.strip()
            if not nm:
                continue
            tid = name_to_id.get(nm.lower(), "")
            rows.append({"doi": doi, "topic_display_name": nm, "topic_id": tid})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote extended mapping CSV: {output_path}")

def export_doi_keywords_long_format(df: pd.DataFrame,
                                    keyword_column: str = "final_keywords",
                                    output_path: str = "doi_keyword_ricgraph.csv") -> None:
    """
    Export a simple DOI–keyword CSV (one keyword per row).

    Args:
        df: results dataframe
        keyword_column: column with semicolon-separated topic names
        output_path: csv path
    """
    rows = []
    for _, row in df.iterrows():
        doi = (row.get("doi") or "").strip()
        for kw in str(row.get(keyword_column, "")).split(";"):
            if kw.strip():
                rows.append({"doi": doi, "keyword": kw.strip()})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote CSV: {output_path}")

def _name_to_id_map(topics_df: pd.DataFrame) -> dict:
    """
    Build a lowercase name -> OpenAlex topic_id map.

    Args:
        topics_df: topics dataframe

    Returns:
        Dict mapping display_name.lower() to id.
    """
    if not {"id", "display_name"}.issubset(topics_df.columns):
        raise ValueError("topics_df must contain 'id' and 'display_name'")
    return {str(n).lower(): str(i) for n, i in zip(topics_df["display_name"], topics_df["id"])}

def _column_to_long_rows(df: pd.DataFrame,
                         column: str,
                         method_label: str,
                         name2id: dict) -> list[dict]:
    """
    Convert a scored column like 'Topic (0.83); Topic2 (0.72)' into long rows.

    Args:
        df: results dataframe
        column: column name to expand
        method_label: method identifier
        name2id: mapping name.lower() -> topic_id

    Returns:
        List of row dicts for long-format export.
    """
    rows = []
    for _, row in df.iterrows():
        doi = str(row.get("doi") or "").strip()
        scored = parse_topic_score_column(row.get(column, ""))
        for rank, (name, score) in enumerate(scored, start=1):
            topic_id = name2id.get(name.lower(), "")
            rows.append({
                "doi": doi,
                "method": method_label,
                "rank": rank,
                "topic_display_name": name,
                "topic_id": topic_id,
                "score": float(score),
            })
    return rows

def export_extended_excel(df_results: pd.DataFrame,
                          topics_df: pd.DataFrame,
                          out_path: str = "classified_abstracts_openalex_extended.xlsx",
                          include_llm_raw_sheet: bool = True) -> None:
    """
    Write a multi-sheet Excel report:
      - Summary: per-abstract overview
      - Candidates_long: all candidates from all methods with topic_id and scores
      - LLM_raw: raw LLM topics (optional)
      - Topics_snapshot: subset of topics metadata
    """
    name2id = _name_to_id_map(topics_df)

    long_rows = []
    if "top_topics_whole" in df_results.columns:
        long_rows += _column_to_long_rows(df_results, "top_topics_whole", "whole_abstract", name2id)
    if "top_topics_sentences" in df_results.columns:
        long_rows += _column_to_long_rows(df_results, "top_topics_sentences", "sentence_aggregate", name2id)
    if "top_topics_llm_matched_1a" in df_results.columns:
        long_rows += _column_to_long_rows(df_results, "top_topics_llm_matched_1a", "llm_top1", name2id)
    if "top_topics_llm_matched_3b" in df_results.columns:
        long_rows += _column_to_long_rows(df_results, "top_topics_llm_matched_3b", "llm_top3", name2id)
    df_long = pd.DataFrame(long_rows, columns=["doi", "method", "rank", "topic_display_name", "topic_id", "score"])

    summary_cols = [
        "doi",
        "final_keywords",
        "top_topics_whole",
        "top_topics_sentences",
        "top_topics_llm_raw",
        "top_topics_llm_matched_1a",
        "top_topics_llm_matched_3b",
    ]
    summary_cols = [c for c in summary_cols if c in df_results.columns]
    df_summary = df_results.loc[:, summary_cols].copy()

    if include_llm_raw_sheet and "top_topics_llm_raw" in df_results.columns:
        df_llm_raw = df_results.loc[:, ["doi", "top_topics_llm_raw"]].copy()
    else:
        df_llm_raw = None

    topic_cols_pref = ["id", "display_name", "domain", "field", "subfield", "works_count", "cited_by_count", "updated_date"]
    topic_cols = [c for c in topic_cols_pref if c in topics_df.columns]
    df_topics_snap = topics_df.loc[:, topic_cols].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df_summary.to_excel(xw, sheet_name="Summary", index=False)
        if not df_long.empty:
            df_long.sort_values(by=["doi", "method", "rank"], inplace=True)
            df_long.to_excel(xw, sheet_name="Candidates_long", index=False)
        if df_llm_raw is not None:
            df_llm_raw.to_excel(xw, sheet_name="LLM_raw", index=False)
        df_topics_snap.to_excel(xw, sheet_name="Topics_snapshot", index=False)

    print(f"Wrote extended Excel: {out_path}")


# ===================== Orchestration =====================
def classify_all(records: list[dict],
                 topic_names: list[str],
                 topic_embeddings: np.ndarray,
                 model: SentenceTransformer) -> pd.DataFrame:
    """
    Run all classification methods for a list of abstracts and return a results DataFrame.

    Args:
        records: [{"doi": str, "abstract": str}, ...]
        topic_names: official OpenAlex topic names
        topic_embeddings: embedding matrix for topics
        model: SentenceTransformer used for encoding abstracts/queries

    Returns:
        DataFrame with per-abstract results across methods.
    """
    output = []
    for record in records:
        abstract = record["abstract"]

        # Method 1: whole abstract similarity
        top_whole = classify_abstract_whole(abstract, topic_names, topic_embeddings, model)

        # Method 2: sentence-level aggregated similarity
        top_sent = classify_abstract_by_sentences(abstract, topic_names, topic_embeddings, model)

        # Method 3: LLM topics → mapped to OpenAlex
        short_abs = truncate_text(abstract)
        llm_topics = generate_llm_topics(short_abs)

        # Fallback: synthesize topic strings from embedding results if LLM returns nothing
        if not llm_topics:
            fallback = []
            fallback += [t for t, _ in top_whole[:3]]
            fallback += [t for t, _ in top_sent[:2]]
            seen = set()
            tmp = []
            for t in fallback:
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    tmp.append(t)
            llm_topics = tmp[:5]

        llm_match_1, llm_match_3 = match_llm_topics_to_openalex(llm_topics, topic_names, topic_embeddings, model)

        output.append({
            "doi": record["doi"],
            "abstract": abstract,
            "top_topics_whole": "; ".join([f"{t[0]} ({t[1]:.2f})" for t in top_whole]),
            "top_topics_sentences": "; ".join([f"{t[0]} ({t[1]:.2f})" for t in top_sent]),
            "top_topics_llm_raw": "; ".join(llm_topics),
            "top_topics_llm_matched_1a": format_topic_matches(llm_match_1),
            "top_topics_llm_matched_3b": format_topic_matches(llm_match_3),
        })
    return pd.DataFrame(output)


# ===================== Main =====================
if __name__ == "__main__":
    # Ensure topics.jsonl exists
    if not os.path.exists(TOPICS_JSONL) or os.path.getsize(TOPICS_JSONL) < 1000:
        print("Downloading OpenAlex topics (API)...")
        download_all_topics_jsonl(TOPICS_JSONL)
    else:
        print(f"{TOPICS_JSONL} already exists, skipping download.")

    # Load abstracts
    print("Loading TXT abstracts...")
    abstracts = load_abstracts_from_txt(TEXTS_FOLDER)

    # Load topics and embeddings
    print("Loading topics and embeddings...")
    embedder = SentenceTransformer(MODEL_NAME)
    topics_df, topic_names, topic_embeddings = load_or_create_embeddings_topics(
        TOPICS_JSONL, MODEL_NAME, EMBEDDINGS_NPY, text_mode=TEXT_MODE
    )
    print(f"Topics loaded: {len(topics_df)} | Embeddings shape: {topic_embeddings.shape}")

    # Classify
    print("Classifying abstracts...")
    df_results = classify_all(abstracts, topic_names, topic_embeddings, embedder)

    # Build final keywords
    print("Computing final keywords...")
    df_results = add_final_keywords_column(df_results)

    # Validate final keywords against official OpenAlex names
    official_names = set(n.lower() for n in topics_df["display_name"].astype(str).tolist())
    df_results = validate_final_keywords(df_results, "final_keywords", official_names)

    # Export doi-keyword CSV
    export_doi_keywords_long_format(df_results, output_path=RICGRAPH_CSV)

    # Extended export with topic_id
    expand_final_keywords_with_ids(
        df_results,
        topics_jsonl_path=TOPICS_JSONL,
        keyword_column="final_keywords",
        output_path=RICGRAPH_CSV_WITH_IDS
    )

    # Simple Excel export (single sheet)
    try:
        df_results.to_excel(RESULT_XLSX, index=False)
        print(f"Wrote Excel: {RESULT_XLSX}")
    except Exception as e:
        print(f"Excel export failed: {e}. You still have CSV exports.")

    # Extended Excel export (multi-sheet)
    try:
        export_extended_excel(
            df_results,
            topics_df,
            out_path=RESULT_XLSX_EXT,
            include_llm_raw_sheet=True
        )
    except Exception as e:
        print(f"Extended Excel export failed: {e}")

    print("Done.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stap-voor-stap:
1) Download alle OpenAlex Topics -> topics.jsonl
2) Embed teksten (naam + beschrijving + labels) -> topics_embeddings.npy
3) Kleine demo: top-5 meest gelijkende topics voor een voorbeeldquery

Maak dit bestand gewoon Run in PyCharm. Geen argparse nodig.
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# =======================
# Config (pas aan indien gewenst)
# =======================
API_URL      = "https://api.openalex.org/topics"
MAILTO       = "d.grotebeverborg@uu.nl"   # <-- jouw e-mail (aanbevolen door OpenAlex)
OUT_JSONL    = "topics.jsonl"
EMB_PATH     = "../topics_embeddings.npy"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
PER_PAGE     = 200
SELECT       = "id,display_name,description,works_count,cited_by_count,updated_date,domain,subfield,field"
SLEEP_SEC    = 0.1
TEXT_MODE    = "name+desc+labels"  # of: "name" / "name+desc"


# =======================
# 1) Download alle Topics naar JSONL
# =======================
def download_all_topics_jsonl(out_path: str) -> int:
    """
    Haalt alle topics via cursor-paging op en schrijft naar JSONL.
    Returns: aantal ontvangen records.
    """
    cursor = "*"
    total = 0
    pages = 0

    # ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "wb") as out:
        while True:
            params = {
                "per-page": PER_PAGE,
                "cursor": cursor,
                "select": SELECT,
                "mailto": MAILTO
            }
            print(f"📄 Ophalen pagina {pages+1}… (cursor={cursor})")
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

    print(f"✅ Download klaar: {total} topics (in {pages} pagina's) -> {out_path}")
    return total


# =======================
# 2) Embeddings maken of laden
# =======================
def load_or_create_embeddings_topics(topics_path: str,
                                     model_name: str,
                                     embeddings_npy: str,
                                     text_mode: str = "name+desc+labels"):
    """
    Laadt topics (JSONL of CSV/TSV) en maakt/lekt embeddings.
    Retourneert: df, topic_ids, texts, embeddings (np.ndarray)
    """
    # --- inlezen ---
    ext = os.path.splitext(topics_path)[1].lower()
    if ext == ".jsonl":
        df = pd.read_json(topics_path, lines=True, dtype=False)
    else:
        sep = "\t" if topics_path.endswith(".tsv") else ","
        df = pd.read_csv(topics_path, sep=sep, dtype=False)

    if "id" not in df.columns or "display_name" not in df.columns:
        raise ValueError("Verwacht minstens kolommen 'id' en 'display_name'.")

    if df["id"].duplicated().any():
        df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)

    # kolommen die we kunnen gebruiken
    for col in ["description", "domain", "field", "subfield"]:
        if col not in df.columns:
            df[col] = None

    def build_text(row: Dict) -> str:
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

    texts = df.apply(build_text, axis=1).tolist()
    topic_ids = df["id"].astype(str).tolist()

    # --- embeddings ---
    if os.path.exists(embeddings_npy):
        X = np.load(embeddings_npy)
        if X.shape[0] != len(texts):
            print(f"⚠️ Cache mismatch: {X.shape[0]} != {len(texts)} → opnieuw encoden.")
            X = _encode_and_save(texts, model_name, embeddings_npy)
        else:
            print(f"✅ Embeddings geladen uit cache: {embeddings_npy}")
    else:
        X = _encode_and_save(texts, model_name, embeddings_npy)

    return df, topic_ids, texts, X


def _encode_and_save(texts: List[str], model_name: str, embeddings_npy: str) -> np.ndarray:
    print(f"🔄 Embeddings genereren voor {len(texts)} topics met '{model_name}'…")
    model = SentenceTransformer(model_name)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    np.save(embeddings_npy, X)
    print(f"💾 Opgeslagen: {embeddings_npy}  (shape={X.shape})")
    return X


# =======================
# 3) Demo: cosine top-k zoeken
# =======================
def search_topics(query: str, model_name: str, texts: List[str], X: np.ndarray, top_k: int = 5):
    """
    Encode query en toon top-k meest gelijkende topics.
    """
    model = SentenceTransformer(model_name)
    q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

    # cosine similarity: (X·q) / (||X|| * ||q||)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    sims = X_norm @ q_norm

    idx = np.argsort(-sims)[:top_k]
    scores = sims[idx]
    return idx, scores


# =======================
# main
# =======================
def main():
    # 1) Download topics indien niet aanwezig (of leeg)
    need_download = (not os.path.exists(OUT_JSONL)) or (os.path.getsize(OUT_JSONL) < 1000)
    if need_download:
        print("🌐 Topics downloaden (API)…")
        download_all_topics_jsonl(OUT_JSONL)
    else:
        print(f"📦 {OUT_JSONL} bestaat al — overslaan download.")

    # 2) Embeddings bouwen / laden
    df, topic_ids, texts, X = load_or_create_embeddings_topics(
        topics_path=OUT_JSONL,
        model_name=MODEL_NAME,
        embeddings_npy=EMB_PATH,
        text_mode=TEXT_MODE
    )
    print(f"✅ Topics geladen: {len(df)} | Embeddings shape: {X.shape}")

    # 3) Mini-demo: top-5 matches voor een voorbeeldvraag
    example_query = "co2 history of netherlands"
    print(f"\n🔎 Query demo: “{example_query}”")
    idx, scores = search_topics(example_query, MODEL_NAME, texts, X, top_k=5)
    for rank, (i, sc) in enumerate(zip(idx, scores), start=1):
        print(f"{rank:>2}. {df.iloc[i]['display_name']}  (score={sc:.4f})  id={df.iloc[i]['id']}")

    print("\n🎉 Klaar.")

if __name__ == "__main__":
    main()

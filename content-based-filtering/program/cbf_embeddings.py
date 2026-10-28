#!/usr/bin/env python3
"""
Content-Based Filtering dengan Word Embedding dan Sentence Embedding
=====================================================================

Program demonstrasi untuk laporan kajian. Program ini menunjukkan:

1. Word Embedding sederhana berbasis matriks co-occurrence + PPMI + SVD.
   Setiap kata menjadi vektor padat; kedekatan vektor = kedekatan makna.
2. Sentence Embedding sederhana dengan mean pooling vektor kata.
3. Perbandingan dengan TF-IDF (baseline klasik) dan Sentence-BERT
   (sentence embedding modern, opsional jika paket tersedia).
4. Content-based filtering: item yang mirip dengan query diperingkat
   menggunakan cosine similarity.

Jalankan:
    python3 cbf_embeddings.py

Output:
    - ringkasan proses dan hasil di terminal
    - ../assets/demo-data.json  (data untuk halaman web)
    - ../assets/demo-data.js    (data yang dimuat langsung oleh halaman)
    - output.txt                (salinan ringkasan output)

Catatan:
    - Metode Word Embedding di sini sengaja kecil dan transparan agar mudah
      dijelaskan. Ia bukan Word2Vec produksi, tetapi memakai prinsip yang sama:
      kata yang muncul dalam konteks serupa mendapat vektor yang mirip.
    - Sentence-BERT memakai model all-MiniLM-L6-v2. Jika paket/model tidak
      tersedia, program tetap berjalan dengan TF-IDF dan mean pooling.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# 1. Dataset kecil: 12 film dan deskripsinya
# ---------------------------------------------------------------------------

MOVIES = [
    {
        "id": "interstellar",
        "title": "Interstellar",
        "year": 2014,
        "genres": ["Sci-Fi", "Adventure", "Drama"],
        "description": "A team of astronauts travels through a wormhole in space to find a new home for humanity.",
    },
    {
        "id": "the-martian",
        "title": "The Martian",
        "year": 2015,
        "genres": ["Sci-Fi", "Adventure", "Drama"],
        "description": "An astronaut stranded on Mars must survive and find a way back to Earth.",
    },
    {
        "id": "arrival",
        "title": "Arrival",
        "year": 2016,
        "genres": ["Sci-Fi", "Drama", "Mystery"],
        "description": "A linguist works with aliens who arrive on Earth to understand their language and prevent a global war.",
    },
    {
        "id": "gravity",
        "title": "Gravity",
        "year": 2013,
        "genres": ["Sci-Fi", "Thriller", "Drama"],
        "description": "Two astronauts fight to survive after debris destroys their space shuttle in orbit.",
    },
    {
        "id": "the-notebook",
        "title": "The Notebook",
        "year": 2004,
        "genres": ["Romance", "Drama"],
        "description": "A young couple falls in love in the 1940s, but war and illness test their romance.",
    },
    {
        "id": "titanic",
        "title": "Titanic",
        "year": 1997,
        "genres": ["Romance", "Drama"],
        "description": "A young woman falls in love with an artist aboard the doomed ocean liner Titanic.",
    },
    {
        "id": "la-la-land",
        "title": "La La Land",
        "year": 2016,
        "genres": ["Romance", "Drama", "Music"],
        "description": "A jazz pianist and an aspiring actress fall in love while pursuing their dreams in Los Angeles.",
    },
    {
        "id": "the-godfather",
        "title": "The Godfather",
        "year": 1972,
        "genres": ["Crime", "Drama"],
        "description": "The aging patriarch of a crime family transfers control to his reluctant son.",
    },
    {
        "id": "toy-story",
        "title": "Toy Story",
        "year": 1995,
        "genres": ["Animation", "Adventure", "Comedy"],
        "description": "A cowboy doll feels threatened when a new space ranger toy becomes a boy's favorite.",
    },
    {
        "id": "inception",
        "title": "Inception",
        "year": 2010,
        "genres": ["Sci-Fi", "Action", "Thriller"],
        "description": "A thief who steals secrets through dream-sharing is hired to plant an idea in a target's mind.",
    },
    {
        "id": "pride-and-prejudice",
        "title": "Pride & Prejudice",
        "year": 2005,
        "genres": ["Romance", "Drama"],
        "description": "A young woman navigates family expectations and a complicated romance in 19th-century England.",
    },
    {
        "id": "the-dark-knight",
        "title": "The Dark Knight",
        "year": 2008,
        "genres": ["Action", "Crime", "Drama"],
        "description": "A vigilante faces a criminal mastermind who wants to plunge Gotham into chaos.",
    },
]

QUERY_PRESETS = [
    {
        "id": "q-mars",
        "label": "Astronaut stranded on Mars",
        "text": "A film about an astronaut who is stranded on Mars and must survive alone.",
    },
    {
        "id": "q-love",
        "label": "Tragic love on a ship",
        "text": "A tragic love story about a young couple on a doomed ship.",
    },
    {
        "id": "q-crime",
        "label": "Crime family",
        "text": "A crime family and a reluctant son who takes control of the business.",
    },
]

STOPWORDS = set(
    """
    a an the and or but if while when where who whom which what why how
    in on at to of for from with by as is are was were be been being
    that this these those it its their his her they he she we you i
    not no do does did have has had will would can could may might must
    into over after before under above about than then there here
    """.split()
)


def normalize_token(word: str) -> str:
    """Normalisasi ringan agar bentuk jamak dan tunggal tidak terpisah."""
    if word in {"mars", "news", "series", "species", "debris", "angeles"}:
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Tokenisasi sederhana: huruf kecil, buang tanda baca dan stopword."""
    words = re.findall(r"[a-z]+", text.lower())
    return [normalize_token(w) for w in words if w not in STOPWORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# 2. Word embedding: co-occurrence -> PPMI -> SVD
# ---------------------------------------------------------------------------


def build_word_vectors(
    documents: Iterable[str], dim: int = 48, window: int = 2
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """
    Membangun word embedding statis dari korpus kecil.

    Langkah:
      1. Hitung co-occurrence dalam jendela `window`.
      2. Ubah counts menjadi Positive PMI (PPMI).
      3. Reduksi dimensi dengan TruncatedSVD.
      4. Normalisasi L2 agar cosine similarity mudah dihitung.

    PPMI menonjolkan pasangan kata yang muncul lebih sering daripada
    yang diharapkan secara acak. SVD memampatkan matriks renggang menjadi
    vektor padat yang menangkap pola semantik utama.
    """
    tokenized = [tokenize(doc) for doc in documents]
    vocab = sorted({word for doc in tokenized for word in doc})
    word_to_id = {word: i for i, word in enumerate(vocab)}
    size = len(vocab)

    counts = np.zeros((size, size), dtype=np.float64)
    for doc in tokenized:
        for i, word in enumerate(doc):
            left = max(0, i - window)
            right = min(len(doc), i + window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                counts[word_to_id[word], word_to_id[doc[j]]] += 1.0

    total = counts.sum()
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((counts * total) / (row_sums * col_sums))
    pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)
    ppmi = np.maximum(pmi, 0.0)

    components = max(2, min(dim, min(ppmi.shape) - 1))
    svd = TruncatedSVD(n_components=components, random_state=42)
    dense = svd.fit_transform(ppmi)

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.clip(norms, 1e-12, None)
    vectors = {word: dense[i] for i, word in enumerate(vocab)}

    info = {
        "vocab_size": size,
        "cooccurrence_shape": list(counts.shape),
        "dim": components,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "nonzero_ppmi": int(np.count_nonzero(ppmi)),
    }
    return vectors, info


def mean_sentence_vector(text: str, word_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Sentence embedding sederhana: rata-rata vektor kata yang dikenal."""
    if not word_vectors:
        raise ValueError("word_vectors kosong")
    dim = len(next(iter(word_vectors.values())))
    words = [word for word in tokenize(text) if word in word_vectors]
    if not words:
        return np.zeros(dim, dtype=np.float64)
    vector = np.mean([word_vectors[word] for word in words], axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else vector


def nearest_words(
    word: str, word_vectors: dict[str, np.ndarray], top_n: int = 8
) -> list[tuple[str, float]]:
    """Kata-kata dengan cosine similarity tertinggi terhadap `word`."""
    if word not in word_vectors:
        return []
    words = list(word_vectors)
    matrix = np.vstack([word_vectors[w] for w in words])
    sims = matrix @ word_vectors[word]
    order = np.argsort(-sims)
    return [(words[i], float(sims[i])) for i in order if words[i] != word][:top_n]


# ---------------------------------------------------------------------------
# 3. Baseline TF-IDF dan Sentence-BERT opsional
# ---------------------------------------------------------------------------


def tfidf_vectors(texts: list[str]) -> tuple[np.ndarray, TfidfVectorizer]:
    """TF-IDF + normalisasi L2, baseline leksikal untuk CBF."""
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, lowercase=False, token_pattern=None
    )
    matrix = vectorizer.fit_transform(texts)
    dense = matrix.toarray().astype(np.float64)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.clip(norms, 1e-12, None)
    return dense, vectorizer


def sbert_vectors(texts: list[str]) -> tuple[np.ndarray | None, str]:
    """
    Sentence embedding modern dengan Sentence-BERT.

    Model all-MiniLM-L6-v2 menghasilkan vektor 384 dimensi. Jika paket
    atau model tidak tersedia (misalnya saat offline), fungsi mengembalikan
    None agar program tetap bisa dijalankan.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None, "Sentence-BERT tidak tersedia (paket tidak terpasang)."

    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float64), "sentence-transformers/all-MiniLM-L6-v2"
    except Exception as exc:  # model belum diunduh / tidak ada koneksi
        return None, f"Sentence-BERT gagal dimuat: {type(exc).__name__}."


# ---------------------------------------------------------------------------
# 4. Content-based filtering: ranking dengan cosine similarity
# ---------------------------------------------------------------------------


def rank_items(
    query_vector: np.ndarray,
    item_vectors: np.ndarray,
    query_index: int | None = None,
    top_n: int = 5,
) -> list[dict[str, object]]:
    """Peringkat item berdasarkan cosine similarity terhadap query."""
    query_norm = np.linalg.norm(query_vector)
    if query_norm < 1e-12:
        return []
    query = query_vector / query_norm
    sims = item_vectors @ query
    order = np.argsort(-sims)
    results: list[dict[str, object]] = []
    for index in order:
        if query_index is not None and int(index) == query_index:
            continue
        results.append(
            {
                "index": int(index),
                "id": MOVIES[int(index)]["id"],
                "title": MOVIES[int(index)]["title"],
                "year": MOVIES[int(index)]["year"],
                "genres": MOVIES[int(index)]["genres"],
                "score": round(float(sims[int(index)]), 4),
            }
        )
        if len(results) >= top_n:
            break
    return results


def format_rankings(name: str, rankings: list[dict[str, object]]) -> str:
    lines = [f"  {name}"]
    for position, item in enumerate(rankings, start=1):
        lines.append(
            f"    {position}. {item['title']:<20} {item['score']:.4f}  "
            f"({', '.join(item['genres'])})"
        )
    return "\n".join(lines)


def cosine_breakdown(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    """Rincian perhitungan cosine untuk satu pasangan vektor."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    return {
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine": dot / (norm_a * norm_b) if norm_a and norm_b else 0.0,
    }


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


class Tee:
    """Tulis ke stdout sekaligus ke file output."""

    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def main() -> int:
    here = Path(__file__).resolve().parent
    assets = here.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    tee = Tee(here / "output.txt")
    sys.stdout = tee  # type: ignore[assignment]

    print("=" * 78)
    print("CONTENT-BASED FILTERING DENGAN WORD & SENTENCE EMBEDDING")
    print("=" * 78)
    print(f"Jumlah item            : {len(MOVIES)} film")
    print(f"Jumlah kata deskripsi  : {sum(len(tokenize(m['description'])) for m in MOVIES)} token")

    descriptions = [movie["description"] for movie in MOVIES]
    query_index = next(i for i, movie in enumerate(MOVIES) if movie["id"] == "interstellar")

    # --- Word embedding -----------------------------------------------------
    print("\n[1] WORD EMBEDDING: co-occurrence -> PPMI -> SVD")
    word_vectors, info = build_word_vectors(descriptions, dim=48, window=2)
    print(f"    Ukuran kosakata        : {info['vocab_size']}")
    print(f"    Matriks co-occurrence  : {info['cooccurrence_shape'][0]} x {info['cooccurrence_shape'][1]}")
    print(f"    Dimensi vektor         : {info['dim']}")
    print(f"    Explained variance SVD : {info['explained_variance']:.2%}")
    print(f"    Sel PPMI tak nol       : {info['nonzero_ppmi']}")

    print("\n    Tetangga terdekat (cosine) untuk beberapa kata:")
    neighbor_words = ["space", "astronaut", "love", "crime", "dream", "family"]
    neighbors_data = {}
    for word in neighbor_words:
        pairs = nearest_words(word, word_vectors, top_n=8)
        neighbors_data[word] = pairs
        rendered = ", ".join(f"{w} ({s:.2f})" for w, s in pairs) if pairs else "(kata tidak ada di kosakata)"
        print(f"      {word:<10} -> {rendered}")

    # --- Sentence vectors ---------------------------------------------------
    print("\n[2] SENTENCE EMBEDDING")
    word_mean_item_vectors = np.vstack(
        [mean_sentence_vector(movie["description"], word_vectors) for movie in MOVIES]
    )
    print("    Mean pooling           : rata-rata vektor kata, lalu normalisasi L2")

    tfidf_item_vectors, tfidf_vectorizer = tfidf_vectors(descriptions)
    print(f"    TF-IDF baseline        : {tfidf_item_vectors.shape[1]} dimensi kosakata")

    sbert_item_vectors, sbert_note = sbert_vectors(descriptions)
    if sbert_item_vectors is not None:
        print(f"    Sentence-BERT          : {sbert_note} ({sbert_item_vectors.shape[1]} dimensi)")
    else:
        print(f"    Sentence-BERT          : {sbert_note}")

    # --- Ranking untuk query Interstellar -----------------------------------
    query_text = MOVIES[query_index]["description"]
    print(f"\n[3] REKOMENDASI UNTUK QUERY: {MOVIES[query_index]['title']}")
    print(f"    Query: \"{query_text}\"")

    tfidf_query = tfidf_item_vectors[query_index]
    word_query = word_mean_item_vectors[query_index]
    tfidf_rankings = rank_items(tfidf_query, tfidf_item_vectors, query_index=query_index, top_n=5)
    word_rankings = rank_items(word_query, word_mean_item_vectors, query_index=query_index, top_n=5)
    print()
    print(format_rankings("TF-IDF (leksikal)", tfidf_rankings))
    print()
    print(format_rankings("Word Embedding + Mean Pooling", word_rankings))

    sbert_rankings: list[dict[str, object]] = []
    if sbert_item_vectors is not None:
        sbert_query = sbert_item_vectors[query_index]
        sbert_rankings = rank_items(
            sbert_query, sbert_item_vectors, query_index=query_index, top_n=5
        )
        print()
        print(format_rankings("Sentence-BERT (semantik)", sbert_rankings))

    # --- Contoh perhitungan cosine -----------------------------------------
    print("\n[4] CONTOH PERHITUNGAN COSINE (Word Embedding + Mean Pooling)")
    compare_index = next(i for i, movie in enumerate(MOVIES) if movie["id"] == "the-martian")
    breakdown = cosine_breakdown(word_query, word_mean_item_vectors[compare_index])
    print(f"    Pasangan: Interstellar vs The Martian")
    print(f"    dot(q, d)      = {breakdown['dot']:.6f}")
    print(f"    ||q||          = {breakdown['norm_a']:.6f}")
    print(f"    ||d||          = {breakdown['norm_b']:.6f}")
    print(f"    cosine         = {breakdown['cosine']:.6f}")

    # --- Data untuk halaman web --------------------------------------------
    item_word_vectors = {
        movie["id"]: [round(float(x), 6) for x in word_mean_item_vectors[i]]
        for i, movie in enumerate(MOVIES)
    }
    demo_data = {
        "meta": {
            "title": "Content-Based Filtering: Word Embedding vs Sentence Embedding",
            "word_method": "PPMI + SVD (co-occurrence, window=2)",
            "sentence_method": "Mean pooling vektor kata + L2 normalization",
            "sbert_model": sbert_note if sbert_item_vectors is not None else None,
            "word_dim": int(info["dim"]),
            "corpus_vocab": int(info["vocab_size"]),
            "explained_variance": round(float(info["explained_variance"]), 4),
        },
        "movies": MOVIES,
        "queryPresets": QUERY_PRESETS,
        "wordVectors": {
            word: [round(float(x), 6) for x in vector]
            for word, vector in word_vectors.items()
        },
        "wordNeighbors": {
            word: [{"word": w, "score": round(score, 4)} for w, score in pairs]
            for word, pairs in neighbors_data.items()
        },
        "wordMean": {
            "dim": int(word_mean_item_vectors.shape[1]),
            "itemVectors": {
                movie["id"]: [round(float(x), 6) for x in word_mean_item_vectors[i]]
                for i, movie in enumerate(MOVIES)
            },
        },
        "tfidf": {
            "vocab": list(tfidf_vectorizer.get_feature_names_out()),
            "idf": {
                word: round(float(value), 6)
                for word, value in zip(
                    tfidf_vectorizer.get_feature_names_out(),
                    tfidf_vectorizer.idf_,
                )
            },
        },
        "sbert": None
        if sbert_item_vectors is None
        else {
            "model": sbert_note,
            "dim": int(sbert_item_vectors.shape[1]),
            "itemVectors": {
                movie["id"]: [round(float(x), 6) for x in sbert_item_vectors[i]]
                for i, movie in enumerate(MOVIES)
            },
            "presetVectors": {},
        },
        "rankings": {
            "interstellar": {
                "tfidf": tfidf_rankings,
                "word_mean": word_rankings,
                "sbert": sbert_rankings,
            }
        },
        "cosineExample": {
            "query": "Interstellar",
            "candidate": "The Martian",
            **{key: round(float(value), 6) for key, value in breakdown.items()},
        },
    }

    if sbert_item_vectors is not None:
        preset_vectors, _ = sbert_vectors([preset["text"] for preset in QUERY_PRESETS])
        if preset_vectors is not None:
            for preset, vector in zip(QUERY_PRESETS, preset_vectors):
                demo_data["sbert"]["presetVectors"][preset["id"]] = [
                    round(float(x), 6) for x in vector
                ]

    json_path = assets / "demo-data.json"
    js_path = assets / "demo-data.js"
    json_text = json.dumps(demo_data, ensure_ascii=False, indent=2)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    js_path.write_text(
        "// Data demo yang dihasilkan oleh program/cbf_embeddings.py\n"
        "// Dibuat otomatis; jangan diedit manual.\n"
        f"window.DEMO_DATA = {json_text};\n",
        encoding="utf-8",
    )

    print("\n[5] FILE YANG DIHASILKAN")
    print(f"    {json_path}")
    print(f"    {js_path}")
    print(f"    {here / 'output.txt'}")
    print("\nSelesai. Buka ../index.html untuk melihat demonstrasi interaktif.")
    print("=" * 78)

    tee.close()
    sys.stdout = sys.__stdout__
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

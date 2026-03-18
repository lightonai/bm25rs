"""
Profiled MS MARCO benchmark — shows per-phase timing breakdown.
Usage: CUDA_VISIBLE_DEVICES=3 python benchmarks/bench_msmarco_profiled.py
"""

import gc
import os
import time

import bm25x


def load_msmarco():
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = os.path.join("benchmarks/data", "msmarco")
    if not os.path.exists(data_path):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        data_path = util.download_and_unzip(url, "benchmarks/data")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[did].get("title") or "") + " " + (corpus[did].get("text") or "")
        for did in corpus_ids
    ]
    return corpus_texts


corpus = load_msmarco()
n = len(corpus)
print(f"GPU available: {bm25x.is_gpu_available()}")
print(f"Corpus: {n:,} docs")
print("=" * 65)

# Warmup
w = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
w.add(corpus[:1000])
del w
gc.collect()

# CPU trial
print("\n--- CPU ---")
idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
t0 = time.perf_counter()
idx.add(corpus)
t_cpu = time.perf_counter() - t0
print(f"  Total: {t_cpu:.3f}s  ({n / t_cpu:,.0f} d/s)")
del idx
gc.collect()

# GPU trial (if available)
if bm25x.is_gpu_available():
    # GPU warmup (kernel compilation)
    w = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    w.add(corpus[:1000])
    del w
    gc.collect()

    print("\n--- GPU ---")
    idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    t0 = time.perf_counter()
    idx.add(corpus)
    t_gpu = time.perf_counter() - t0
    print(f"  Total: {t_gpu:.3f}s  ({n / t_gpu:,.0f} d/s)")
    print(f"\n  Speedup: {t_cpu / t_gpu:.2f}x")

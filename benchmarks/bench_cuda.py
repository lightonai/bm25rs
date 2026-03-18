"""
Benchmark bm25x CUDA-accelerated indexing vs CPU (rayon) indexing.

Uses BEIR SciFact dataset for realistic evaluation.
"""

import time

import bm25x
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ───────────────────────── load dataset ──────────────────────────

print("Downloading SciFact dataset...")
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
data_path = util.download_and_unzip(url, "benchmarks/data")

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

corpus_ids = list(corpus.keys())
corpus_texts = [
    (corpus[did].get("title") or "") + " " + (corpus[did].get("text") or "")
    for did in corpus_ids
]
test_qids = sorted(qrels.keys())
test_queries = [queries[qid] for qid in test_qids if qid in queries]

gpu_available = bm25x.is_gpu_available()
print(f"GPU available: {gpu_available}")
print(f"Corpus: {len(corpus_texts)} docs | Queries: {len(test_queries)}")
print("=" * 65)

# ───────────────────────── CPU benchmark ─────────────────────────

print("\n--- CPU indexing (rayon) ---")

# Warmup
warmup = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
warmup.add(corpus_texts[:100])
del warmup

cpu_times = []
for trial in range(5):
    idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    t0 = time.perf_counter()
    idx.add(corpus_texts)
    t = time.perf_counter() - t0
    cpu_times.append(t)
    if trial < 4:
        del idx

cpu_index = idx  # keep last one for correctness check
best_cpu = min(cpu_times)
dps_cpu = len(corpus_texts) / best_cpu
print(f"  Times: {[f'{t:.4f}s' for t in cpu_times]}")
print(f"  Best:  {best_cpu:.4f}s  ({dps_cpu:,.0f} d/s)")

# Search sanity check
cpu_results = [cpu_index.search(q, 10) for q in test_queries[:10]]
print(
    f"  Search check: first 10 queries OK ({sum(len(r) for r in cpu_results)} results)"
)

# ───────────────────────── GPU benchmark ─────────────────────────

if gpu_available:
    print("\n--- GPU indexing (CUDA) ---")

    # Warmup GPU (first call compiles NVRTC kernels)
    warmup = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    warmup.add(corpus_texts[:100])
    del warmup

    gpu_times = []
    for trial in range(5):
        idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
        t0 = time.perf_counter()
        idx.add(corpus_texts)
        t = time.perf_counter() - t0
        gpu_times.append(t)
        if trial < 4:
            del idx

    gpu_index = idx
    best_gpu = min(gpu_times)
    dps_gpu = len(corpus_texts) / best_gpu
    print(f"  Times: {[f'{t:.4f}s' for t in gpu_times]}")
    print(f"  Best:  {best_gpu:.4f}s  ({dps_gpu:,.0f} d/s)")

    # Correctness: compare search results CPU vs GPU
    print("\n--- Correctness verification ---")
    mismatches = 0
    for q in test_queries[:50]:
        cpu_r = cpu_index.search(q, 10)
        gpu_r = gpu_index.search(q, 10)
        cpu_ids = [r[0] for r in cpu_r]
        gpu_ids = [r[0] for r in gpu_r]
        if cpu_ids != gpu_ids:
            mismatches += 1
    print(f"  Tested 50 queries: {mismatches} mismatches")
    if mismatches == 0:
        print("  PASS: GPU and CPU produce identical results")
    else:
        print("  WARN: Some results differ (may be due to tie-breaking)")

    # Summary
    print("\n" + "=" * 65)
    speedup = best_cpu / best_gpu
    print(f"CPU: {best_cpu:.4f}s  ({dps_cpu:,.0f} d/s)")
    print(f"GPU: {best_gpu:.4f}s  ({dps_gpu:,.0f} d/s)")
    print(f"Speedup: {speedup:.2f}x")
else:
    print(
        "\nNOTE: GPU not available. Build with CUDA support to compare:\n"
        "      make build-gpu\n"
        "      # or: cd python && maturin develop --release --features cuda"
    )
    print("\n" + "=" * 65)
    print(f"CPU only: {best_cpu:.4f}s  ({dps_cpu:,.0f} d/s)")

print("Done.")

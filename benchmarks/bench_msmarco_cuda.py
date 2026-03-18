"""
Benchmark bm25x indexing on MS MARCO passage corpus (~8.8M passages).
Compares CPU (rayon) vs GPU (CUDA) indexing throughput.

Usage:
    CUDA_VISIBLE_DEVICES=3 python benchmarks/bench_msmarco_cuda.py
"""

import gc
import os
import time

import bm25x

# ───────────────────────── load MS MARCO ─────────────────────────


def load_msmarco(max_docs=None):
    """Load MS MARCO passage corpus via BEIR."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    dataset = "msmarco"
    data_dir = "benchmarks/data"
    data_path = os.path.join(data_dir, dataset)

    if not os.path.exists(data_path):
        print("Downloading MS MARCO dataset (this may take a while)...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, data_dir)

    print("Loading corpus...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

    corpus_ids = list(corpus.keys())
    if max_docs:
        corpus_ids = corpus_ids[:max_docs]

    corpus_texts = [
        (corpus[did].get("title") or "") + " " + (corpus[did].get("text") or "")
        for did in corpus_ids
    ]

    test_qids = sorted(qrels.keys())[:200]
    test_queries = [queries[qid] for qid in test_qids if qid in queries]

    return corpus_texts, test_queries


def bench_indexing(corpus_texts, label, trials=3):
    """Benchmark indexing and return (best_time, index)."""
    # Warmup
    w = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    w.add(corpus_texts[:500])
    del w
    gc.collect()

    times = []
    for trial in range(trials):
        idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
        t0 = time.perf_counter()
        idx.add(corpus_texts)
        t = time.perf_counter() - t0
        times.append(t)
        print(
            f"  [{label}] trial {trial + 1}: {t:.3f}s  ({len(corpus_texts) / t:,.0f} d/s)"
        )
        if trial < trials - 1:
            del idx
            gc.collect()

    best = min(times)
    dps = len(corpus_texts) / best
    print(f"  [{label}] best: {best:.3f}s  ({dps:,.0f} d/s)")
    return best, idx


# ───────────────────────── main ──────────────────────────────────


def main():
    gpu_available = bm25x.is_gpu_available()
    print(f"GPU available: {gpu_available}")

    # Load corpus
    corpus_texts, test_queries = load_msmarco()
    n = len(corpus_texts)
    print(f"Corpus: {n:,} docs | Queries: {len(test_queries)}")
    print("=" * 65)

    # ── CPU baseline ──
    print(f"\n--- CPU indexing (rayon) on {n:,} docs ---")
    cpu_time, cpu_index = bench_indexing(corpus_texts, "CPU", trials=3)

    # Quick search sanity check
    t0 = time.perf_counter()
    for q in test_queries:
        cpu_index.search(q, 10)
    t_search = time.perf_counter() - t0
    print(
        f"  [CPU] search {len(test_queries)} queries: {t_search:.3f}s ({len(test_queries) / t_search:,.0f} q/s)"
    )

    if not gpu_available:
        print("\nNo GPU — done.")
        return

    # ── GPU ──
    del cpu_index
    gc.collect()

    print(f"\n--- GPU indexing (CUDA) on {n:,} docs ---")
    gpu_time, gpu_index = bench_indexing(corpus_texts, "GPU", trials=3)

    # Search check
    t0 = time.perf_counter()
    for q in test_queries:
        gpu_index.search(q, 10)
    t_search_gpu = time.perf_counter() - t0
    print(
        f"  [GPU] search {len(test_queries)} queries: {t_search_gpu:.3f}s ({len(test_queries) / t_search_gpu:,.0f} q/s)"
    )

    # ── Summary ──
    print("\n" + "=" * 65)
    speedup = cpu_time / gpu_time
    print(f"{'':>20} {'Time':>10} {'Throughput':>15}")
    print(f"{'CPU (rayon)':>20} {cpu_time:>9.3f}s {n / cpu_time:>14,.0f} d/s")
    print(f"{'GPU (CUDA)':>20} {gpu_time:>9.3f}s {n / gpu_time:>14,.0f} d/s")
    print(f"{'Speedup':>20} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()

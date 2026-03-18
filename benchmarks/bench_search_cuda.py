"""
Benchmark bm25x search throughput on MS MARCO.
Measures single-query latency and batch QPS.

Usage: CUDA_VISIBLE_DEVICES=3 BM25X_PROFILE=1 python benchmarks/bench_search_cuda.py
"""

import os
import statistics
import time

import bm25x

# ───────────────────────── load & index ──────────────────────────


def load_and_index():
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = os.path.join("benchmarks/data", "msmarco")
    if not os.path.exists(data_path):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
        data_path = util.download_and_unzip(url, "benchmarks/data")

    print("Loading corpus...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

    corpus_ids = list(corpus.keys())
    corpus_texts = [
        (corpus[did].get("title") or "") + " " + (corpus[did].get("text") or "")
        for did in corpus_ids
    ]

    test_qids = sorted(qrels.keys())
    test_queries = [queries[qid] for qid in test_qids if qid in queries]

    print(f"Corpus: {len(corpus_texts):,} docs | Queries: {len(test_queries)}")

    # Index
    print("Indexing...")
    idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=True)
    t0 = time.perf_counter()
    idx.add(corpus_texts)
    t_index = time.perf_counter() - t0
    print(f"  Indexed in {t_index:.1f}s ({len(corpus_texts) / t_index:,.0f} d/s)")

    return idx, test_queries, corpus_texts


def bench_search(idx, queries, label, k=10, warmup=50, trials=3):
    """Benchmark search: measure per-query latency and batch throughput."""
    n = len(queries)

    # Warmup
    for q in queries[:warmup]:
        idx.search(q, k)

    # Per-query latency (sample 200 queries)
    sample = queries[:200]
    latencies = []
    for q in sample:
        t0 = time.perf_counter()
        idx.search(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    mean_lat = statistics.mean(latencies)

    # Batch throughput
    batch_times = []
    for trial in range(trials):
        t0 = time.perf_counter()
        for q in queries:
            idx.search(q, k)
        batch_times.append(time.perf_counter() - t0)

    best = min(batch_times)
    qps = n / best

    print(f"\n--- {label} (k={k}, {n} queries, {idx.__len__():,} docs) ---")
    print(
        f"  Latency: mean={mean_lat:.2f}ms  p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms"
    )
    print(f"  Batch:   {[f'{t:.3f}s' for t in batch_times]}")
    print(f"  Best:    {best:.3f}s  ({qps:,.0f} q/s)")
    return {
        "mean_lat_ms": mean_lat,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "qps": qps,
        "best_time": best,
    }


# ───────────────────────── main ──────────────────────────────────


def main():
    print(f"GPU available: {bm25x.is_gpu_available()}")
    idx, queries, corpus = load_and_index()

    # Search benchmarks at different k values
    results = {}
    for k in [10, 100, 1000]:
        r = bench_search(idx, queries, f"Search top-{k}", k=k)
        results[k] = r

    # Summary
    print("\n" + "=" * 65)
    print(f"{'k':>6} {'Mean lat':>10} {'p50':>8} {'p95':>8} {'QPS':>10}")
    for k, r in results.items():
        print(
            f"{k:>6} {r['mean_lat_ms']:>9.2f}ms {r['p50_ms']:>7.2f}ms {r['p95_ms']:>7.2f}ms {r['qps']:>9,.0f}"
        )


if __name__ == "__main__":
    main()

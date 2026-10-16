
"""
Step 6: Corrected vector-index benchmark.

Uses saved Parent-Child, Summary, and RAPTOR indexes.
Builds Flat and IVF-PQ in memory.
Summary and RAPTOR are expected to index the full corpus.
Flat agreement is a proxy, not ground-truth relevance.
"""

import csv
import pickle
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent

CORPUS_EMBEDDINGS = ROOT / "msmarco_corpus_embeddings.npy"
QUERY_EMBEDDINGS = ROOT / "msmarco_query_embeddings.npy"
PASSAGES_FILE = ROOT / "msmarco_passages.txt"

PARENT_INDEX = ROOT / "parent_child_index.faiss"
PARENT_META = ROOT / "parent_child_meta.pkl"

SUMMARY_INDEX = ROOT / "summary_index.faiss"
SUMMARY_META = ROOT / "summary_meta.pkl"

RAPTOR_INDEX = ROOT / "raptor_index.faiss"
RAPTOR_META = ROOT / "raptor_meta.pkl"

OUTPUT_FILE = ROOT / "corrected_benchmark_results.csv"

K = 5
CANDIDATE_DEPTH = 50


def load_pickle(path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_text(text):
    return " ".join(str(text).split()).casefold()


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    return vectors


def unique_valid(values, limit=None):
    result = []
    seen = set()

    for value in values:
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue

        if value < 0 or (limit is not None and value >= limit):
            continue

        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def index_size_mb(index_path=None, index=None):
    if index_path is not None and index_path.exists():
        return index_path.stat().st_size / (1024 * 1024)

    if index is None:
        return 0.0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "index.faiss"
        faiss.write_index(index, str(temp_path))
        return temp_path.stat().st_size / (1024 * 1024)


def timed_search(index, queries, search_k):
    search_k = max(1, min(int(search_k), index.ntotal))

    start = time.perf_counter()
    distances, ids = index.search(queries, search_k)
    latency_ms = (
        (time.perf_counter() - start) * 1000 / max(len(queries), 1)
    )
    return distances, ids, latency_ms


def make_passage_lookup(passages):
    lookup = {}
    for doc_id, passage in enumerate(passages):
        key = normalize_text(passage)
        if key:
            lookup.setdefault(key, doc_id)
    return lookup


def passage_id_from_value(value, passage_lookup, corpus_size):
    if isinstance(value, (int, np.integer)):
        doc_id = int(value)
        return doc_id if 0 <= doc_id < corpus_size else None

    if isinstance(value, str):
        return passage_lookup.get(normalize_text(value))

    return None


def map_parent_child_metadata(metadata, passages, corpus_size):
    if not isinstance(metadata, (tuple, list)) or len(metadata) != 2:
        raise ValueError(
            "Parent-Child metadata must contain sentences and parent mapping."
        )

    sentences, parent_data = metadata
    passage_lookup = make_passage_lookup(passages)
    mapping = []

    if isinstance(parent_data, dict):
        for child_id in range(len(sentences)):
            parent = parent_data.get(
                child_id, parent_data.get(str(child_id))
            )
            doc_id = passage_id_from_value(
                parent, passage_lookup, corpus_size
            )
            mapping.append([] if doc_id is None else [doc_id])

    elif isinstance(parent_data, (list, tuple)):
        if len(parent_data) != len(sentences):
            raise ValueError(
                "Parent-Child sentence and parent arrays differ in length."
            )

        for parent in parent_data:
            doc_id = passage_id_from_value(
                parent, passage_lookup, corpus_size
            )
            mapping.append([] if doc_id is None else [doc_id])

    else:
        raise ValueError(
            f"Unsupported Parent-Child mapping: {type(parent_data).__name__}"
        )

    return mapping


def map_summary_metadata(metadata, passages, corpus_size):
    if not isinstance(metadata, (tuple, list)) or len(metadata) != 2:
        raise ValueError(
            "Summary metadata must be (summaries, source_passages)."
        )

    summaries, source_passages = metadata

    if len(summaries) != len(source_passages):
        raise ValueError("Summary metadata arrays differ in length.")

    lookup = make_passage_lookup(passages)
    mapping = []

    for source_passage in source_passages:
        doc_id = passage_id_from_value(
            source_passage, lookup, corpus_size
        )
        mapping.append([] if doc_id is None else [doc_id])

    return mapping


def map_raptor_metadata(metadata, corpus_size):
    if not isinstance(metadata, dict):
        raise ValueError("RAPTOR metadata must be a schema-v2 dictionary.")

    if metadata.get("schema_version") != 2:
        raise ValueError(
            f"Expected RAPTOR schema v2; got "
            f"{metadata.get('schema_version')!r}."
        )

    mappings = metadata.get("index_source_document_ids")
    if not isinstance(mappings, (list, tuple)):
        raise ValueError(
            "RAPTOR metadata lacks index_source_document_ids."
        )

    result = []
    for source_ids in mappings:
        if not isinstance(source_ids, (list, tuple, set)):
            source_ids = [source_ids]
        result.append(unique_valid(source_ids, corpus_size))

    return result


def map_search_results(search_ids, mapping, top_k=K):
    ranked_lists = []

    for row in search_ids:
        ranked = []
        seen = set()

        for raw_id in row:
            idx = int(raw_id)
            if idx < 0 or idx >= len(mapping):
                continue

            for doc_id in mapping[idx]:
                if doc_id not in seen:
                    seen.add(doc_id)
                    ranked.append(doc_id)

                if len(ranked) >= top_k:
                    break

            if len(ranked) >= top_k:
                break

        ranked_lists.append(ranked)

    return ranked_lists


def proxy_metrics(ranked_lists, flat_ids, k=K):
    recalls = []
    reciprocal_ranks = []
    ndcgs = []

    for retrieved, baseline_row in zip(ranked_lists, flat_ids):
        truth = unique_valid(baseline_row)[:k]
        truth_set = set(truth)
        retrieved = unique_valid(retrieved)[:k]

        hits = [
            rank for rank, doc_id in enumerate(retrieved, start=1)
            if doc_id in truth_set
        ]

        overlap = len(set(retrieved) & truth_set)
        recalls.append(overlap / len(truth_set) if truth_set else 0.0)
        reciprocal_ranks.append(1.0 / hits[0] if hits else 0.0)

        dcg = sum(1.0 / np.log2(rank + 1) for rank in hits)
        ideal_count = min(len(truth_set), k)
        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)

    agreement = float(np.mean(recalls)) if recalls else 0.0

    return {
        "Recall@5 proxy": agreement,
        "Flat agreement@5": agreement,
        "MRR@5 proxy": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "nDCG@5 proxy": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def mapping_coverage(mapping):
    if not mapping:
        return 0.0
    return 100.0 * sum(bool(x) for x in mapping) / len(mapping)


def append_result(
    results, strategy, index, search_ids, latency, flat_ids,
    index_path=None, mapping=None, notes=""
):
    if mapping is None:
        ranked_lists = [unique_valid(row)[:K] for row in search_ids]
        coverage = 100.0
    else:
        ranked_lists = map_search_results(search_ids, mapping, K)
        coverage = mapping_coverage(mapping)

    metrics = proxy_metrics(ranked_lists, flat_ids, K)

    results.append({
        "Strategy": strategy,
        "Avg query latency (ms)": round(float(latency), 4),
        "Index size (MB)": round(
            index_size_mb(index_path=index_path, index=index), 4
        ),
        "Recall@5 proxy": round(metrics["Recall@5 proxy"], 4),
        "Flat agreement@5": round(metrics["Flat agreement@5"], 4),
        "MRR@5 proxy": round(metrics["MRR@5 proxy"], 4),
        "nDCG@5 proxy": round(metrics["nDCG@5 proxy"], 4),
        "Source mapping coverage (%)": round(coverage, 2),
        "Indexed vectors": int(index.ntotal),
        "Eligible queries": int(len(search_ids)),
        "Notes": notes,
    })


def main():
    required = [CORPUS_EMBEDDINGS, QUERY_EMBEDDINGS, PASSAGES_FILE]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {ROOT}: {', '.join(missing)}"
        )

    print("=" * 65)
    print("CORRECTED VECTOR INDEX BENCHMARK")
    print(f"Project: {ROOT}")
    print("=" * 65)

    corpus = normalize_vectors(np.load(CORPUS_EMBEDDINGS))
    queries = normalize_vectors(np.load(QUERY_EMBEDDINGS))

    with PASSAGES_FILE.open("r", encoding="utf-8") as f:
        passages = [line.rstrip("\r\n") for line in f if line.strip()]

    n_docs = min(len(corpus), len(passages))
    corpus = corpus[:n_docs]

    if n_docs == 0 or len(queries) == 0:
        raise ValueError("Corpus or query embeddings are empty.")

    if corpus.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: {corpus.shape} vs {queries.shape}"
        )

    dimension = corpus.shape[1]
    print(f"Corpus passages: {n_docs}")
    print(f"Query embeddings: {len(queries)}")
    print(f"Embedding dimension: {dimension}")
    print("Proxy evaluation: agreement with Flat top-5, not relevance.\n")

    results = []

    # 1. Flat exact baseline
    print("[1/5] Building Flat...")
    flat = faiss.IndexFlatIP(dimension)
    start = time.perf_counter()
    flat.add(corpus)
    flat_build_time = time.perf_counter() - start
    _, flat_ids, flat_latency = timed_search(flat, queries, K)

    append_result(
        results, "Flat", flat, flat_ids, flat_latency, flat_ids,
        notes=(
            f"Exact search over all {n_docs:,} passages. "
            f"Build time this run: {flat_build_time:.3f}s."
        ),
    )

    # 2. IVF-PQ
    print("[2/5] Building IVF-PQ in memory...")
    nlist = min(100, max(4, n_docs // 100))
    m = 48 if dimension % 48 == 0 else 16
    quantizer = faiss.IndexFlatIP(dimension)
    ivfpq = faiss.IndexIVFPQ(
        quantizer, dimension, nlist, m, 4, faiss.METRIC_INNER_PRODUCT
    )

    start = time.perf_counter()
    ivfpq.train(corpus)
    ivfpq.add(corpus)
    pq_build_time = time.perf_counter() - start
    ivfpq.nprobe = min(10, nlist)

    _, pq_ids, pq_latency = timed_search(ivfpq, queries, K)
    append_result(
        results, "IVF-PQ", ivfpq, pq_ids, pq_latency, flat_ids,
        notes=(
            f"Rebuilt in memory; nlist={nlist}, nprobe={ivfpq.nprobe}, "
            f"M={m}, 4-bit PQ. Build time this run: {pq_build_time:.3f}s."
        ),
    )

    # 3. Parent-Child
    print("[3/5] Benchmarking Parent-Child...")
    if PARENT_INDEX.exists() and PARENT_META.exists():
        pc_index = faiss.read_index(str(PARENT_INDEX))
        pc_meta = load_pickle(PARENT_META)
        pc_mapping = map_parent_child_metadata(pc_meta, passages, n_docs)
        _, pc_ids, pc_latency = timed_search(
            pc_index, queries, CANDIDATE_DEPTH
        )
        pc_ranked = map_search_results(pc_ids, pc_mapping, K)
        metrics = proxy_metrics(pc_ranked, flat_ids, K)

        results.append({
            "Strategy": "Parent-Child",
            "Avg query latency (ms)": round(pc_latency, 4),
            "Index size (MB)": round(index_size_mb(PARENT_INDEX), 4),
            **{key: round(value, 4) for key, value in metrics.items()},
            "Source mapping coverage (%)": round(
                mapping_coverage(pc_mapping), 2
            ),
            "Indexed vectors": int(pc_index.ntotal),
            "Eligible queries": len(queries),
            "Notes": (
                "Sentence hits mapped to parent passages; "
                f"candidate depth={CANDIDATE_DEPTH}."
            ),
        })
    else:
        print("Skipping Parent-Child: index or metadata missing.")

    # 4. Full-corpus Summary
    print("[4/5] Benchmarking Summary...")
    if SUMMARY_INDEX.exists() and SUMMARY_META.exists():
        summary_index = faiss.read_index(str(SUMMARY_INDEX))
        summary_meta = load_pickle(SUMMARY_META)
        summary_mapping = map_summary_metadata(
            summary_meta, passages, n_docs
        )

        usable_vectors = min(summary_index.ntotal, len(summary_mapping))
        if usable_vectors:
            _, summary_ids, summary_latency = timed_search(
                summary_index, queries,
                min(CANDIDATE_DEPTH, usable_vectors)
            )
            summary_mapping = summary_mapping[:usable_vectors]
            summary_ranked = map_search_results(
                summary_ids, summary_mapping, K
            )
            metrics = proxy_metrics(summary_ranked, flat_ids, K)
            full_corpus = usable_vectors == n_docs

            results.append({
                "Strategy": "Summary",
                "Avg query latency (ms)": round(summary_latency, 4),
                "Index size (MB)": round(index_size_mb(SUMMARY_INDEX), 4),
                **{key: round(value, 4) for key, value in metrics.items()},
                "Source mapping coverage (%)": round(
                    mapping_coverage(summary_mapping), 2
                ),
                "Indexed vectors": int(summary_index.ntotal),
                "Eligible queries": len(queries),
                "Notes": (
                    f"Full-corpus Summary index: {usable_vectors:,} "
                    f"summary vectors linked to source passages out of "
                    f"{n_docs:,} corpus passages."
                    if full_corpus else
                    f"Partial Summary index: {usable_vectors:,} summary "
                    f"vectors out of {n_docs:,} corpus passages."
                ),
            })
    else:
        print("Skipping Summary: index or metadata missing.")

    # 5. RAPTOR schema v2
    print("[5/5] Benchmarking RAPTOR...")
    if RAPTOR_INDEX.exists() and RAPTOR_META.exists():
        raptor_index = faiss.read_index(str(RAPTOR_INDEX))
        raptor_meta = load_pickle(RAPTOR_META)
        raptor_mapping = map_raptor_metadata(raptor_meta, n_docs)

        usable_vectors = min(raptor_index.ntotal, len(raptor_mapping))
        if usable_vectors:
            _, raptor_ids, raptor_latency = timed_search(
                raptor_index, queries,
                min(CANDIDATE_DEPTH, usable_vectors)
            )
            raptor_mapping = raptor_mapping[:usable_vectors]
            raptor_ranked = map_search_results(
                raptor_ids, raptor_mapping, K
            )
            metrics = proxy_metrics(raptor_ranked, flat_ids, K)

            base_docs = raptor_meta.get("n_base_documents", "unknown")
            cluster_count = len(
                raptor_meta.get("cluster_summary_ids", {})
            )

            results.append({
                "Strategy": "RAPTOR",
                "Avg query latency (ms)": round(raptor_latency, 4),
                "Index size (MB)": round(index_size_mb(RAPTOR_INDEX), 4),
                **{key: round(value, 4) for key, value in metrics.items()},
                "Source mapping coverage (%)": round(
                    mapping_coverage(raptor_mapping), 2
                ),
                "Indexed vectors": int(raptor_index.ntotal),
                "Eligible queries": len(queries),
                "Notes": (
                    f"Full-corpus RAPTOR: {base_docs} base documents plus "
                    f"{cluster_count} cluster-summary vectors. "
                    "Schema-v2 source mappings expand cluster-summary "
                    "hits to source document IDs; proxy score is not "
                    "ground-truth relevance."
                ),
            })
    else:
        print("Skipping RAPTOR: index or metadata missing.")

    if not results:
        raise RuntimeError("No benchmark results were generated.")

    fields = [
        "Strategy",
        "Avg query latency (ms)",
        "Index size (MB)",
        "Recall@5 proxy",
        "Flat agreement@5",
        "MRR@5 proxy",
        "nDCG@5 proxy",
        "Source mapping coverage (%)",
        "Indexed vectors",
        "Eligible queries",
        "Notes",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETE")
    print("=" * 65)

    for row in results:
        print(f"\n{row['Strategy']}")
        print(f"  Latency: {row['Avg query latency (ms)']:.4f} ms/query")
        print(f"  Index size: {row['Index size (MB)']:.4f} MB")
        print(f"  Flat agreement@5: {row['Flat agreement@5']:.4f}")
        print(
            "  Source mapping coverage: "
            f"{row['Source mapping coverage (%)']:.2f}%"
        )
        print(f"  Indexed vectors: {row['Indexed vectors']}")
        print(f"  Notes: {row['Notes']}")

    print(f"\nCSV saved to: {OUTPUT_FILE}")
    print(
        "\nFlat agreement measures overlap with Flat's top-5 results; "
        "it is not ground-truth semantic relevance."
    )


if __name__ == "__main__":
    main()

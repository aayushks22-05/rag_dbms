
import csv
import pickle
import time
from pathlib import Path

import faiss
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

PROJECTS = [
    ("rag_dbms", 5000),
    ("rag_dbms_10K", 10000),
]

TOP_K = 5
RANK_K = 10
SEARCH_K = 50
WARMUP = 5

OUTPUT_FILE = ROOT / "all_metrics_results.csv"


# ============================================================
# FILE HELPERS
# ============================================================

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_text(value):
    return " ".join(str(value).split()).strip()


def load_queries(path):
    """
    Supports TSV files with a header or the usual MS MARCO
    two-column format: query_id<TAB>query_text.
    """
    queries = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.readline()
        f.seek(0)

        if not sample.strip():
            return queries

        first = sample.rstrip("\r\n").split("\t")
        has_header = (
            len(first) >= 2
            and first[0].strip().lower()
            in {"query_id", "qid", "id"}
        )

        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                qid = row.get("query_id") or row.get("qid") or row.get("id")
                text = (
                    row.get("query")
                    or row.get("query_text")
                    or row.get("text")
                )
                if qid is not None and text:
                    queries.append((str(qid).strip(), text.strip()))
        else:
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    queries.append(
                        (parts[0].strip(), parts[1].strip())
                    )

    return queries


def load_qrels(path):
    """
    Standard MS MARCO qrels format:
    query_id  0  passage_id  relevance

    Also accepts the three-column variant:
    query_id  passage_id  relevance
    """
    qrels = {}

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 3:
                continue

            try:
                if len(parts) >= 4:
                    qid = str(parts[0])
                    docid = str(parts[2])
                    relevance = float(parts[3])
                else:
                    qid = str(parts[0])
                    docid = str(parts[1])
                    relevance = float(parts[2])
            except ValueError:
                # Ignore headers or malformed lines.
                continue

            qrels.setdefault(qid, {})[docid] = relevance

    return qrels


def index_size_mb(path):
    return path.stat().st_size / (1024 ** 2)


# ============================================================
# ID AND METADATA MAPPING
# ============================================================

def build_text_lookup(passages):
    lookup = {}
    for local_id, text in enumerate(passages):
        key = normalize_text(text)
        if key and key not in lookup:
            lookup[key] = local_id
    return lookup


def map_value(value, text_lookup, corpus_size):
    """
    Convert metadata values to local corpus row IDs.
    Strings are interpreted as passage text.
    Integers are interpreted as local row IDs.
    """
    if isinstance(value, str):
        local_id = text_lookup.get(normalize_text(value))
        return [] if local_id is None else [local_id]

    if isinstance(value, (int, np.integer)):
        local_id = int(value)
        return [local_id] if 0 <= local_id < corpus_size else []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        result = set()
        for item in value:
            result.update(map_value(item, text_lookup, corpus_size))
        return sorted(result)

    return []


def load_strategy_mappings(project, text_lookup, corpus_size):
    mappings = {}

    # Parent-Child metadata contains two arrays per index vector.
    path = project / "parent_child_meta.pkl"
    if path.exists():
        metadata = load_pickle(path)

        if not isinstance(metadata, (tuple, list)) or len(metadata) < 2:
            raise ValueError("Unexpected Parent-Child metadata format.")

        first, second = metadata[0], metadata[1]

        if len(first) != len(second):
            raise ValueError("Parent-Child metadata lengths differ.")

        mappings["Parent-Child"] = [
            sorted(set(
                map_value(a, text_lookup, corpus_size)
                + map_value(b, text_lookup, corpus_size)
            ))
            for a, b in zip(first, second)
        ]

    # Summary metadata: summaries and their source passages.
    path = project / "summary_meta.pkl"
    if path.exists():
        metadata = load_pickle(path)

        if not isinstance(metadata, (tuple, list)) or len(metadata) < 2:
            raise ValueError("Unexpected Summary metadata format.")

        summaries, sources = metadata[0], metadata[1]

        if len(summaries) != len(sources):
            raise ValueError("Summary metadata lengths differ.")

        mappings["Summary"] = [
            map_value(source, text_lookup, corpus_size)
            for source in sources
        ]

    # RAPTOR schema version 2.
    path = project / "raptor_meta.pkl"
    if path.exists():
        metadata = load_pickle(path)

        if not isinstance(metadata, dict):
            raise ValueError("RAPTOR metadata must be a dictionary.")

        if metadata.get("schema_version") != 2:
            raise ValueError("RAPTOR metadata must use schema version 2.")

        source_map = metadata.get("index_source_document_ids")
        if source_map is None:
            raise ValueError("RAPTOR source mappings are missing.")

        mappings["RAPTOR"] = [
            map_value(value, {}, corpus_size)
            for value in source_map
        ]

    return mappings


# ============================================================
# METRICS
# ============================================================

def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return None

    return (
        len(set(ranked_ids[:k]) & relevant_ids)
        / len(relevant_ids)
    )


def mrr_at_k(ranked_ids, judgments, k):
    for rank, docid in enumerate(ranked_ids[:k], start=1):
        if judgments.get(docid, 0) > 0:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(ranked_ids, judgments, k):
    dcg = 0.0

    for rank, docid in enumerate(ranked_ids[:k], start=1):
        relevance = judgments.get(docid, 0)
        dcg += (2 ** relevance - 1) / np.log2(rank + 1)

    ideal_relevance = sorted(
        judgments.values(), reverse=True
    )[:k]

    idcg = sum(
        (2 ** relevance - 1) / np.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevance, start=1)
    )

    return dcg / idcg if idcg else 0.0


def evaluate_relevance(query_ids, results, qrels, corpus_ids):
    """
    Score only queries with at least one judged relevant passage
    present in this corpus.

    Returns blank metrics if no queries are eligible.
    """
    corpus_set = set(corpus_ids)

    eligible_qids = {
        qid
        for qid, judgments in qrels.items()
        if any(
            score > 0 and docid in corpus_set
            for docid, score in judgments.items()
        )
    }

    recalls = []
    mrrs = []
    ndcgs = []

    for qid, ranked in zip(query_ids, results):
        if qid not in eligible_qids:
            continue

        judgments = qrels[qid]

        relevant_in_corpus = {
            docid
            for docid, score in judgments.items()
            if score > 0 and docid in corpus_set
        }

        recalls.append(
            recall_at_k(ranked, relevant_in_corpus, TOP_K)
        )
        mrrs.append(mrr_at_k(ranked, judgments, RANK_K))
        ndcgs.append(ndcg_at_k(ranked, judgments, RANK_K))

    if not recalls:
        return {
            "Eligible queries": 0,
            "Recall@5": "",
            "MRR@10": "",
            "nDCG@10": "",
        }

    return {
        "Eligible queries": len(recalls),
        "Recall@5": round(float(np.mean(recalls)), 4),
        "MRR@10": round(float(np.mean(mrrs)), 4),
        "nDCG@10": round(float(np.mean(ndcgs)), 4),
    }


def search_index(index, query_vectors, corpus_doc_ids, mapping=None):
    """
    Returns latency and ranked original passage IDs.

    For Flat and IVF-PQ, FAISS IDs correspond to local corpus rows.
    For other strategies, metadata maps vectors to local rows.
    """
    if index.ntotal == 0:
        raise ValueError("Index is empty.")

    if mapping is not None and len(mapping) != index.ntotal:
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but metadata contains {len(mapping)} entries."
        )

    for query in query_vectors[:min(WARMUP, len(query_vectors))]:
        index.search(query.reshape(1, -1), min(SEARCH_K, index.ntotal))

    latencies = []
    all_results = []

    for query in query_vectors:
        start = time.perf_counter()
        _, hits = index.search(
            query.reshape(1, -1),
            min(SEARCH_K, index.ntotal)
        )
        latencies.append((time.perf_counter() - start) * 1000)

        ranked = []
        seen = set()

        for vector_id in hits[0]:
            vector_id = int(vector_id)
            if vector_id < 0:
                continue

            if mapping is None:
                local_ids = (
                    [vector_id]
                    if vector_id < len(corpus_doc_ids)
                    else []
                )
            else:
                local_ids = (
                    mapping[vector_id]
                    if vector_id < len(mapping)
                    else []
                )

            for local_id in local_ids:
                if 0 <= local_id < len(corpus_doc_ids):
                    docid = corpus_doc_ids[local_id]

                    if docid not in seen:
                        ranked.append(docid)
                        seen.add(docid)

        all_results.append(ranked)

    if mapping is None:
        coverage = 100.0
    else:
        coverage = (
            100.0 * sum(bool(x) for x in mapping) / len(mapping)
            if mapping else 0.0
        )

    return float(np.mean(latencies)), all_results, coverage


def proxy_agreement(results, flat_results):
    scores = []

    for retrieved, exact in zip(results, flat_results):
        exact_top = exact[:TOP_K]
        if exact_top:
            scores.append(
                len(set(retrieved[:TOP_K]) & set(exact_top))
                / len(exact_top)
            )

    return round(float(np.mean(scores)), 4) if scores else ""


# ============================================================
# PROJECT EVALUATION
# ============================================================

def evaluate_project(folder_name, expected_size):
    project = ROOT / folder_name

    print("\n" + "=" * 65)
    print(f"Evaluating {folder_name}")
    print("=" * 65)

    passages = read_lines(project / "msmarco_passages.txt")
    corpus_vectors = np.load(
        project / "msmarco_corpus_embeddings.npy"
    ).astype(np.float32)
    query_vectors = np.load(
        project / "msmarco_query_embeddings.npy"
    ).astype(np.float32)

    queries = load_queries(project / "msmarco_real_queries.tsv")
    qrels = load_qrels(project / "qrels.dev.tsv")
    corpus_doc_ids = read_lines(
        project / "msmarco_passage_ids.txt"
    )

    if len(passages) != expected_size:
        raise ValueError(
            f"{folder_name}: expected {expected_size} passages, "
            f"found {len(passages)}."
        )

    if len(corpus_doc_ids) != len(passages):
        raise ValueError("Passage ID count does not match corpus size.")

    if corpus_vectors.shape[0] != len(passages):
        raise ValueError("Corpus embedding count mismatch.")

    if query_vectors.shape[0] != len(queries):
        raise ValueError(
            f"Query count mismatch: {len(queries)} query texts vs "
            f"{query_vectors.shape[0]} embeddings."
        )

    if corpus_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Corpus/query embedding dimensions differ.")

    query_ids = [qid for qid, _ in queries]
    corpus_set = set(corpus_doc_ids)
    query_set = set(query_ids)
    qrels_query_set = set(qrels)

    relevant_ids = {
        docid
        for judgments in qrels.values()
        for docid, score in judgments.items()
        if score > 0
    }

    relevant_in_corpus = relevant_ids & corpus_set

    eligible_qids = {
        qid
        for qid, judgments in qrels.items()
        if any(
            score > 0 and docid in corpus_set
            for docid, score in judgments.items()
        )
    }

    print(f"Corpus passages: {len(passages)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Qrels query IDs: {len(qrels_query_set)}")
    print(f"Query IDs overlapping qrels: {len(query_set & qrels_query_set)}")
    print(f"Unique judged relevant passage IDs: {len(relevant_ids)}")
    print(f"Relevant IDs found in corpus: {len(relevant_in_corpus)}")
    print(f"Queries with relevant passages in corpus: {len(eligible_qids)}")
    print("Sample corpus IDs:", corpus_doc_ids[:5])
    print("Sample qrels IDs:", list(relevant_ids)[:5])

    # Normalize vectors for cosine similarity.
    faiss.normalize_L2(query_vectors)

    text_lookup = build_text_lookup(passages)
    mappings = load_strategy_mappings(
        project, text_lookup, len(passages)
    )

    strategies = [
        ("Flat", "flat_full_index.faiss", None),
        ("IVF-PQ", "ivfpq_full_index.faiss", None),
        ("Parent-Child", "parent_child_index.faiss",
         mappings.get("Parent-Child")),
        ("Summary", "summary_index.faiss", mappings.get("Summary")),
        ("RAPTOR", "raptor_index.faiss", mappings.get("RAPTOR")),
    ]

    rows = []
    flat_results = None

    for strategy, filename, mapping in strategies:
        index_path = project / filename

        if not index_path.exists():
            print(f"Skipping {strategy}: missing {filename}")
            continue

        index = faiss.read_index(str(index_path))

        latency, results, coverage = search_index(
            index, query_vectors, corpus_doc_ids, mapping
        )

        if strategy == "Flat":
            flat_results = results

        relevance = evaluate_relevance(
            query_ids, results, qrels, corpus_doc_ids
        )

        if strategy == "Flat":
            agreement = 1.0
        elif flat_results is not None:
            agreement = proxy_agreement(results, flat_results)
        else:
            agreement = ""

        row = {
            "Project": folder_name,
            "Corpus passages": len(passages),
            "Strategy": strategy,
            "Indexed vectors": index.ntotal,
            "Index size (MB)": round(index_size_mb(index_path), 4),
            "Avg latency (ms/query)": round(latency, 4),
            "Source mapping coverage (%)": round(coverage, 2),
            **relevance,
            "Flat agreement@5": agreement,
        }
        rows.append(row)

        print(
            f"{strategy:14} "
            f"latency={latency:.4f} ms, "
            f"eligible={relevance['Eligible queries']}, "
            f"Recall@5={relevance['Recall@5']}, "
            f"MRR@10={relevance['MRR@10']}, "
            f"nDCG@10={relevance['nDCG@10']}"
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    all_rows = []

    for folder_name, corpus_size in PROJECTS:
        all_rows.extend(
            evaluate_project(folder_name, corpus_size)
        )

    if not all_rows:
        raise RuntimeError("No evaluation results were generated.")

    fields = [
        "Project",
        "Corpus passages",
        "Strategy",
        "Indexed vectors",
        "Index size (MB)",
        "Avg latency (ms/query)",
        "Source mapping coverage (%)",
        "Eligible queries",
        "Recall@5",
        "MRR@10",
        "nDCG@10",
        "Flat agreement@5",
    ]

    with open(
        OUTPUT_FILE, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Combined results saved to:\n{OUTPUT_FILE}")
    print(
        "\nIf Eligible queries is zero, the query IDs, qrels, and "
        "corpus passage IDs need further alignment checks."
    )
    print(
        "Flat agreement measures agreement with Flat results; "
        "it is not ground-truth relevance."
    )


if __name__ == "__main__":
    main()

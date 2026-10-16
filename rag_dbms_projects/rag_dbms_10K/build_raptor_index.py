
import pickle
import time
import shutil
from pathlib import Path
from datetime import datetime

import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PASSAGES_FILE = PROJECT_DIR / "msmarco_passages.txt"
INDEX_FILE = PROJECT_DIR / "raptor_index.faiss"
META_FILE = PROJECT_DIR / "raptor_meta.pkl"

# Use the entire corpus in this project folder.
EXPECTED_DOCUMENTS = 10000
N_CLUSTERS = 10

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUMMARY_MODEL = "t5-small"
RANDOM_STATE = 42

DEVICE = torch.device("cpu")


# ============================================================
# BACKUP EXISTING ARTIFACTS
# ============================================================

def backup_existing_files():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path in (INDEX_FILE, META_FILE):
        if path.exists():
            backup_path = path.with_name(
                f"{path.stem}_backup_{timestamp}{path.suffix}"
            )

            counter = 1
            while backup_path.exists():
                backup_path = path.with_name(
                    f"{path.stem}_backup_{timestamp}_{counter}"
                    f"{path.suffix}"
                )
                counter += 1

            shutil.copy2(path, backup_path)
            print(f"Backup created: {backup_path.name}")


# ============================================================
# LOAD ALL PASSAGES
# ============================================================

def load_passages():
    if not PASSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Passages file not found: {PASSAGES_FILE}"
        )

    with PASSAGES_FILE.open("r", encoding="utf-8") as file:
        passages = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(passages) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Expected exactly {EXPECTED_DOCUMENTS:,} passages, "
            f"but found {len(passages):,}. "
            "Check the dataset before rebuilding."
        )

    print(f"Loaded all {len(passages):,} passages.")
    return passages


# ============================================================
# EMBEDDING HELPER
# ============================================================

def embed_documents(model, texts, batch_size=64):
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Embedding output must be a 2D array.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    return vectors


# ============================================================
# GENERATE CLUSTER SUMMARIES
# ============================================================

def generate_cluster_summaries(passages, labels, n_clusters):
    print(f"\nLoading summarization model: {SUMMARY_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL)
    model.to(DEVICE)
    model.eval()

    summaries = []
    cluster_members = []
    summary_source_document_ids = []

    for cluster_id in range(n_clusters):
        member_ids = np.where(labels == cluster_id)[0].tolist()
        cluster_members.append(member_ids)

        if not member_ids:
            summaries.append("")
            summary_source_document_ids.append([])
            print(f"Cluster {cluster_id + 1}: empty")
            continue

        # Preserve the existing approach: concatenate cluster text
        # and let the tokenizer truncate it to the model's input limit.
        combined_text = " ".join(
            passages[doc_id] for doc_id in member_ids
        ).strip()

        input_text = "summarize: " + combined_text

        print(
            f"Summarizing cluster {cluster_id + 1}/{n_clusters} "
            f"({len(member_ids)} documents)"
        )

        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=2,
                    do_sample=False,
                )

            summary = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()

        except Exception as error:
            print(
                f"Warning: cluster {cluster_id + 1} summary failed: "
                f"{error}"
            )
            summary = combined_text[:500]

        if not summary:
            summary = combined_text[:500]

        summaries.append(summary)
        summary_source_document_ids.append(member_ids)

    del model
    del tokenizer

    return summaries, cluster_members, summary_source_document_ids


# ============================================================
# MAIN RAPTOR BUILD
# ============================================================

def main():
    start_time = time.perf_counter()

    print("=" * 65)
    print("RAPTOR INDEX BUILD — FULL 10K CORPUS")
    print("=" * 65)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Device: {DEVICE}")

    # 1. Load all 10,000 source passages.
    passages = load_passages()
    document_ids = list(range(len(passages)))

    # 2. Embed base passages.
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nEmbedding all base passages...")
    base_vectors = embed_documents(
        embedding_model,
        passages,
        batch_size=64,
    )

    print(f"Base embedding shape: {base_vectors.shape}")

    # 3. Cluster the embeddings.
    actual_clusters = min(N_CLUSTERS, len(passages))

    print(
        f"\nClustering {len(passages):,} passages "
        f"into {actual_clusters} groups..."
    )

    # Diagonal covariance is more practical for 10K documents
    # than the default full-covariance Gaussian mixture.
    gmm = GaussianMixture(
        n_components=actual_clusters,
        covariance_type="diag",
        random_state=RANDOM_STATE,
        n_init=1,
        max_iter=100,
        reg_covar=1e-5,
    )

    labels = gmm.fit_predict(base_vectors)

    print("Clustering complete.")

    # 4. Generate cluster summaries.
    (
        summaries,
        cluster_members,
        summary_source_document_ids,
    ) = generate_cluster_summaries(
        passages,
        labels,
        actual_clusters,
    )

    # 5. Embed non-empty summaries.
    nonempty_summary_ids = [
        cluster_id
        for cluster_id, summary in enumerate(summaries)
        if summary.strip()
    ]

    nonempty_summaries = [
        summaries[cluster_id]
        for cluster_id in nonempty_summary_ids
    ]

    print("\nEmbedding cluster summaries...")

    if nonempty_summaries:
        summary_vectors = embed_documents(
            embedding_model,
            nonempty_summaries,
            batch_size=32,
        )
    else:
        summary_vectors = np.empty(
            (0, base_vectors.shape[1]),
            dtype=np.float32,
        )

    # 6. Combine base and hierarchy vectors.
    all_vectors = np.vstack(
        [base_vectors, summary_vectors]
    ).astype(np.float32)

    all_texts = list(passages) + list(nonempty_summaries)

    text_types = (
        ["Base Document"] * len(passages)
        + ["Cluster Summary"] * len(nonempty_summaries)
    )

    # Base vector -> its own source passage ID.
    # Cluster summary -> all passage IDs in its cluster.
    index_source_document_ids = (
        [[doc_id] for doc_id in document_ids]
        + [
            summary_source_document_ids[cluster_id]
            for cluster_id in nonempty_summary_ids
        ]
    )

    cluster_summary_ids = {
        cluster_id: len(passages) + position
        for position, cluster_id in enumerate(nonempty_summary_ids)
    }

    # 7. Validate alignment before saving.
    assert len(all_texts) == len(all_vectors)
    assert len(text_types) == len(all_vectors)
    assert len(index_source_document_ids) == len(all_vectors)

    for doc_ids in index_source_document_ids:
        if not doc_ids:
            raise ValueError("An index vector has no source document IDs.")

        if any(
            doc_id < 0 or doc_id >= len(passages)
            for doc_id in doc_ids
        ):
            raise ValueError("Invalid source document ID detected.")

    # 8. Build FAISS index.
    print("\nBuilding FAISS IndexFlatIP...")

    index = faiss.IndexFlatIP(all_vectors.shape[1])
    index.add(all_vectors)

    # 9. Create Step 6-compatible schema-v2 metadata.
    metadata = {
        "schema_version": 2,
        "n_base_documents": len(passages),
        "n_clusters": actual_clusters,
        "document_ids": document_ids,
        "cluster_labels": labels.tolist(),
        "cluster_members": cluster_members,
        "cluster_summary_ids": cluster_summary_ids,
        "summary_source_document_ids": summary_source_document_ids,
        "all_texts": all_texts,
        "text_types": text_types,
        "index_source_document_ids": index_source_document_ids,
    }

    # 10. Back up old artifacts, then write temporary files.
    backup_existing_files()

    temp_index = PROJECT_DIR / "raptor_index.tmp.faiss"
    temp_meta = PROJECT_DIR / "raptor_meta.tmp.pkl"

    try:
        print("\nSaving temporary RAPTOR artifacts...")
        faiss.write_index(index, str(temp_index))

        with temp_meta.open("wb") as file:
            pickle.dump(
                metadata,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Verify temporary artifacts before replacing final files.
        saved_index = faiss.read_index(str(temp_index))

        with temp_meta.open("rb") as file:
            saved_metadata = pickle.load(file)

        assert saved_index.ntotal == len(passages) + len(nonempty_summaries)
        assert saved_index.ntotal == len(saved_metadata["all_texts"])
        assert saved_index.ntotal == len(saved_metadata["text_types"])
        assert saved_index.ntotal == len(
            saved_metadata["index_source_document_ids"]
        )
        assert saved_metadata["schema_version"] == 2
        assert saved_metadata["n_base_documents"] == 10000

        # Replace the previous artifacts only after validation.
        temp_index.replace(INDEX_FILE)
        temp_meta.replace(META_FILE)

    finally:
        if temp_index.exists():
            temp_index.unlink()
        if temp_meta.exists():
            temp_meta.unlink()

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 65)
    print("RAPTOR INDEX BUILD COMPLETE")
    print("=" * 65)
    print(f"Base documents: {len(passages):,}")
    print(f"Clusters: {actual_clusters}")
    print(f"Non-empty summaries: {len(nonempty_summaries)}")
    print(f"Total FAISS vectors: {saved_index.ntotal:,}")
    print(f"Embedding dimension: {saved_index.d}")
    print(
        f"Index size: "
        f"{INDEX_FILE.stat().st_size / (1024 ** 2):.4f} MB"
    )
    print(f"Metadata schema: {saved_metadata['schema_version']}")
    print(f"Total build time: {elapsed / 60:.2f} minutes")
    print(f"Index saved: {INDEX_FILE}")
    print(f"Metadata saved: {META_FILE}")


if __name__ == "__main__":
    main()

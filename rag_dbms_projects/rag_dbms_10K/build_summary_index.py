
import pickle
import shutil
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PASSAGES_FILE = PROJECT_DIR / "msmarco_passages.txt"
INDEX_FILE = PROJECT_DIR / "summary_index.faiss"
META_FILE = PROJECT_DIR / "summary_meta.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARY_MODEL_NAME = "t5-small"

SUMMARY_BATCH_SIZE = 8
EMBEDDING_BATCH_SIZE = 32

# Safety: preserve existing index and metadata before replacement.
BACKUP_SUFFIX = ".pre_full_corpus_backup"


def backup_existing_file(path):
    if path.exists():
        backup_path = path.with_name(path.name + BACKUP_SUFFIX)

        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"Backup created: {backup_path.name}")
        else:
            print(
                f"Existing backup retained: {backup_path.name}"
            )


def load_passages():
    if not PASSAGES_FILE.exists():
        raise FileNotFoundError(
            f"Passage file not found: {PASSAGES_FILE}"
        )

    with PASSAGES_FILE.open("r", encoding="utf-8") as file:
        passages = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not passages:
        raise ValueError("No passages found.")

    print(f"Loaded {len(passages):,} passages.")

    if len(passages) not in (5000, 10000):
        print(
            "WARNING: The passage count is not exactly 5,000 "
            "or 10,000. Verify the dataset before continuing."
        )

    return passages


def generate_summaries(passages, device):
    print(f"\nLoading summarization model: {SUMMARY_MODEL_NAME}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARY_MODEL_NAME
    ).to(device)

    model.eval()

    summaries = []
    start_time = time.perf_counter()

    print("\nGenerating summaries for the full corpus...")

    for start in range(0, len(passages), SUMMARY_BATCH_SIZE):
        batch = passages[start:start + SUMMARY_BATCH_SIZE]

        inputs = tokenizer(
            ["summarize: " + passage for passage in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=2,
                do_sample=False,
            )

        batch_summaries = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        for original, summary in zip(batch, batch_summaries):
            summary = summary.strip()

            # Avoid empty summaries.
            summaries.append(summary if summary else original)

        completed = min(start + len(batch), len(passages))

        if completed % 100 == 0 or completed == len(passages):
            elapsed = time.perf_counter() - start_time
            print(
                f"Summarized {completed:,}/{len(passages):,} "
                f"passages | elapsed: {elapsed / 60:.1f} min"
            )

    del model
    del tokenizer

    if len(summaries) != len(passages):
        raise RuntimeError(
            f"Summary count mismatch: {len(summaries)} summaries "
            f"for {len(passages)} passages."
        )

    print(
        f"Summary generation finished in "
        f"{(time.perf_counter() - start_time) / 60:.2f} minutes."
    )

    return summaries


def build_embeddings(summaries):
    print(
        f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding all summaries...")

    start_time = time.perf_counter()

    embeddings = model.encode(
        summaries,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if embeddings.shape[0] != len(summaries):
        raise RuntimeError("Embedding count does not match summaries.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    print(f"Embedding shape: {embeddings.shape}")
    print(
        f"Embedding generation time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    return embeddings, model


def save_index_and_metadata(embeddings, summaries, passages):
    print("\nBuilding the Summary FAISS index...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != len(passages):
        raise RuntimeError(
            f"Expected {len(passages)} vectors, got {index.ntotal}."
        )

    # Back up the old files before replacing them.
    backup_existing_file(INDEX_FILE)
    backup_existing_file(META_FILE)

    # Write temporary files first, then replace final files.
    temp_index = PROJECT_DIR / "summary_index.tmp.faiss"
    temp_meta = PROJECT_DIR / "summary_meta.tmp.pkl"

    try:
        faiss.write_index(index, str(temp_index))

        with temp_meta.open("wb") as file:
            pickle.dump(
                (summaries, passages),
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Verify the temporary artifacts before replacement.
        check_index = faiss.read_index(str(temp_index))

        with temp_meta.open("rb") as file:
            check_summaries, check_passages = pickle.load(file)

        if check_index.ntotal != len(passages):
            raise RuntimeError("Temporary FAISS index verification failed.")

        if (
            len(check_summaries) != len(passages)
            or len(check_passages) != len(passages)
        ):
            raise RuntimeError("Temporary metadata verification failed.")

        temp_index.replace(INDEX_FILE)
        temp_meta.replace(META_FILE)

    finally:
        if temp_index.exists():
            temp_index.unlink()

        if temp_meta.exists():
            temp_meta.unlink()

    return index


def test_retrieval(index, summaries, passages, embedding_model):
    print("\nTesting Summary retrieval...")

    query = "What is cholestasis?"

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    k = min(2, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    print(f"Query: {query}")

    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0:
            continue

        print(f"\nResult {rank} | Score: {scores[0][rank - 1]:.4f}")
        print(f"Summary: {summaries[idx]}")
        print(f"Source passage: {passages[idx][:500]}")


def main():
    total_start = time.perf_counter()

    print("=" * 65)
    print("FULL-CORPUS DOCUMENT SUMMARY INDEX BUILDER")
    print("=" * 65)
    print(f"Project: {PROJECT_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    passages = load_passages()

    summaries = generate_summaries(passages, device)

    embeddings, embedding_model = build_embeddings(summaries)

    index = save_index_and_metadata(
        embeddings,
        summaries,
        passages,
    )

    test_retrieval(
        index,
        summaries,
        passages,
        embedding_model,
    )

    print("\n" + "=" * 65)
    print("SUMMARY INDEX REBUILD COMPLETE")
    print("=" * 65)
    print(f"Source passages: {len(passages):,}")
    print(f"Summaries:       {len(summaries):,}")
    print(f"FAISS vectors:   {index.ntotal:,}")
    print(f"Dimensions:      {index.d}")
    print(
        f"Index size:      "
        f"{INDEX_FILE.stat().st_size / (1024 * 1024):.3f} MB"
    )
    print(f"Index file:      {INDEX_FILE}")
    print(f"Metadata file:   {META_FILE}")
    print(
        f"Total time:      "
        f"{(time.perf_counter() - total_start) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()

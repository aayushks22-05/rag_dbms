
import csv
import gzip
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
PROJECTS = [ROOT / "rag_dbms", ROOT / "rag_dbms_10K"]
N_QUERIES = 1000

BASE_URL = (
    "https://msmarco.z22.web.core.windows.net/"
    "msmarcoranking/"
)
QUERIES_URL = BASE_URL + "queries.tar.gz"
QRELS_URL = BASE_URL + "qrels.dev.tsv"

DATA_DIR = ROOT / "msmarco_query_data"
DATA_DIR.mkdir(exist_ok=True)

QUERIES_ARCHIVE = DATA_DIR / "queries.tar.gz"
QRELS_FILE = DATA_DIR / "qrels.dev.tsv"


def download(url, destination):
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination.name}")
        return

    print(f"Downloading {destination.name}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved: {destination}")


def read_queries_from_archive(path):
    # The official archive contains query IDs and query text.
    with tarfile.open(path, "r:gz") as archive:
        members = [
            m for m in archive.getmembers()
            if m.isfile() and m.name.endswith(".tsv")
        ]

        if not members:
            raise RuntimeError(
                "No TSV found in queries.tar.gz. "
                "Inspect the archive contents before continuing."
            )

        member = members[0]
        print("Reading query file:", member.name)

        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError("Could not read query TSV.")

        rows = []
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                qid, query = parts
                if query.strip():
                    rows.append((qid, query.strip()))

    return rows


def read_relevant_query_ids(path):
    # qrels.dev.tsv is in TREC-style format:
    # query_id, 0, passage_id, relevance
    query_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qid = parts[0]
                try:
                    relevance = int(parts[3])
                except ValueError:
                    continue

                if relevance > 0:
                    query_ids.add(qid)

    return query_ids


def main():
    download(QUERIES_URL, QUERIES_ARCHIVE)
    download(QRELS_URL, QRELS_FILE)

    queries = read_queries_from_archive(QUERIES_ARCHIVE)
    relevant_ids = read_relevant_query_ids(QRELS_FILE)

    # Only use queries that have at least one positive judgment.
    selected = []
    seen = set()

    for qid, query in queries:
        if qid in relevant_ids and qid not in seen:
            selected.append((qid, query))
            seen.add(qid)

        if len(selected) == N_QUERIES:
            break

    if len(selected) < N_QUERIES:
        raise RuntimeError(
            f"Only found {len(selected)} judged queries; "
            f"{N_QUERIES} were requested. Nothing was updated."
        )

    print(f"\nSelected {len(selected)} real, judged queries.")

    # Preserve IDs, query text, and the official qrels for later evaluation.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_texts = [query for _, query in selected]

    print("Generating embeddings for 1,000 queries...")
    embeddings = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # Keep one shared, reproducible query set for both projects.
    for project in PROJECTS:
        if not project.exists():
            raise FileNotFoundError(
                f"Project folder not found: {project}"
            )

        with open(
            project / "msmarco_real_queries.tsv",
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["query_id", "query"])
            writer.writerows(selected)

        np.save(
            project / "msmarco_query_embeddings.npy",
            embeddings
        )

        # Keep a local copy of the official judgments.
        (project / "qrels.dev.tsv").write_bytes(
            QRELS_FILE.read_bytes()
        )

        print(
            f"Updated {project.name}: "
            f"{len(selected)} queries, "
            f"embeddings shape {embeddings.shape}"
        )

    print("\nSUCCESS: Both projects now use the same real queries.")
    print("Existing corpus embeddings and indexes were not rebuilt.")


if __name__ == "__main__":
    main()

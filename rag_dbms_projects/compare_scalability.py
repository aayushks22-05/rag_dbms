
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

FILE_5K = ROOT / "rag_dbms" / "corrected_benchmark_results.csv"
FILE_10K = ROOT / "rag_dbms_10K" / "corrected_benchmark_results.csv"

OUTPUT_CSV = ROOT / "scalability_comparison.csv"
LATENCY_PLOT = ROOT / "scalability_latency.png"
SIZE_PLOT = ROOT / "scalability_index_size.png"
AGREEMENT_PLOT = ROOT / "scalability_agreement.png"

STRATEGIES = [
    "Flat",
    "IVF-PQ",
    "Parent-Child",
    "Summary",
    "RAPTOR",
]


# ============================================================
# READ AND VALIDATE CSV FILES
# ============================================================

def read_results(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{path}\n"
            "Run Step 6 for this dataset first."
        )

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        # Handle BOMs and accidental whitespace in column names.
        reader.fieldnames = [
            name.strip() if name else name
            for name in reader.fieldnames
        ]

        rows = list(reader)

    if not rows:
        raise ValueError(f"No benchmark results found in {path}")

    results = {}

    for row in rows:
        # Normalize keys and values defensively.
        cleaned = {
            (key.strip() if key else key):
            (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }

        strategy = cleaned.get("Strategy")

        if strategy:
            results[strategy.strip()] = cleaned

    if not results:
        raise ValueError(
            f"No Strategy values found in {path}.\n"
            f"CSV headers: {reader.fieldnames}"
        )

    print(f"\nLoaded {len(results)} strategies from: {path.name}")
    print("Strategies:", ", ".join(results.keys()))

    return results


def number(row, field):
    value = row.get(field)

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Missing or invalid field '{field}'.\n"
            f"Row data: {row}"
        )


def percentage_change(old, new):
    if old == 0:
        return None

    return (new - old) / old * 100


def get_change(old, new):
    change = percentage_change(old, new)
    return round(change, 2) if change is not None else ""


# ============================================================
# PLOTTING HELPER
# ============================================================

def save_comparison_plot(
    labels,
    values_5k,
    values_10k,
    ylabel,
    title,
    output_path,
):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        values_5k,
        width,
        label="5K corpus",
    )

    ax.bar(
        x + width / 2,
        values_10k,
        width,
        label="10K corpus",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():
    print("Loading benchmark results...")

    results_5k = read_results(FILE_5K)
    results_10k = read_results(FILE_10K)

    output_rows = []

    for strategy in STRATEGIES:
        if strategy not in results_5k:
            print(f"Skipping {strategy}: missing from 5K results.")
            continue

        if strategy not in results_10k:
            print(f"Skipping {strategy}: missing from 10K results.")
            continue

        a = results_5k[strategy]
        b = results_10k[strategy]

        # Latency and index storage.
        latency_5k = number(a, "Avg query latency (ms)")
        latency_10k = number(b, "Avg query latency (ms)")

        size_5k = number(a, "Index size (MB)")
        size_10k = number(b, "Index size (MB)")

        # Agreement with Flat's top-5 results.
        # This is NOT ground-truth relevance recall.
        agreement_5k = number(a, "Flat agreement@5")
        agreement_10k = number(b, "Flat agreement@5")

        mapping_5k = number(a, "Source mapping coverage (%)")
        mapping_10k = number(b, "Source mapping coverage (%)")

        vectors_5k = int(number(a, "Indexed vectors"))
        vectors_10k = int(number(b, "Indexed vectors"))

        eligible_5k = int(number(a, "Eligible queries"))
        eligible_10k = int(number(b, "Eligible queries"))

        # Additional proxy metrics from Step 6.
        mrr_5k = number(a, "MRR@5 proxy")
        mrr_10k = number(b, "MRR@5 proxy")

        ndcg_5k = number(a, "nDCG@5 proxy")
        ndcg_10k = number(b, "nDCG@5 proxy")

        output_rows.append({
            "Strategy": strategy,

            "Latency 5K (ms)": round(latency_5k, 4),
            "Latency 10K (ms)": round(latency_10k, 4),
            "Latency change (%)": get_change(
                latency_5k, latency_10k
            ),

            "Index size 5K (MB)": round(size_5k, 4),
            "Index size 10K (MB)": round(size_10k, 4),
            "Index size change (%)": get_change(
                size_5k, size_10k
            ),

            "Flat agreement@5 5K": round(agreement_5k, 4),
            "Flat agreement@5 10K": round(agreement_10k, 4),

            "MRR@5 proxy 5K": round(mrr_5k, 4),
            "MRR@5 proxy 10K": round(mrr_10k, 4),

            "nDCG@5 proxy 5K": round(ndcg_5k, 4),
            "nDCG@5 proxy 10K": round(ndcg_10k, 4),

            "Mapping coverage 5K (%)": round(mapping_5k, 2),
            "Mapping coverage 10K (%)": round(mapping_10k, 2),

            "Indexed vectors 5K": vectors_5k,
            "Indexed vectors 10K": vectors_10k,

            "Eligible queries 5K": eligible_5k,
            "Eligible queries 10K": eligible_10k,

            "Notes 5K": a.get("Notes", ""),
            "Notes 10K": b.get("Notes", ""),
        })

    if not output_rows:
        raise ValueError(
            "No strategies have results for both corpus sizes."
        )

    # ========================================================
    # SAVE COMPARISON CSV
    # ========================================================

    with open(
        OUTPUT_CSV, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    labels = [row["Strategy"] for row in output_rows]

    # ========================================================
    # 1. LATENCY COMPARISON
    # ========================================================

    save_comparison_plot(
        labels,
        [row["Latency 5K (ms)"] for row in output_rows],
        [row["Latency 10K (ms)"] for row in output_rows],
        "Average latency (ms/query)",
        "Retrieval Latency: 5K vs 10K",
        LATENCY_PLOT,
    )

    # ========================================================
    # 2. INDEX SIZE COMPARISON
    # ========================================================

    save_comparison_plot(
        labels,
        [row["Index size 5K (MB)"] for row in output_rows],
        [row["Index size 10K (MB)"] for row in output_rows],
        "Index size (MB)",
        "Index Storage: 5K vs 10K",
        SIZE_PLOT,
    )

    # ========================================================
    # 3. FLAT AGREEMENT COMPARISON
    # ========================================================

    save_comparison_plot(
        labels,
        [row["Flat agreement@5 5K"] for row in output_rows],
        [row["Flat agreement@5 10K"] for row in output_rows],
        "Agreement with Flat top-5 (0-1)",
        "Retrieval Agreement: 5K vs 10K",
        AGREEMENT_PLOT,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("SCALABILITY COMPARISON COMPLETE")
    print("=" * 60)

    print(f"\nComparison CSV: {OUTPUT_CSV}")
    print(f"Latency graph:  {LATENCY_PLOT}")
    print(f"Index-size graph: {SIZE_PLOT}")
    print(f"Agreement graph:  {AGREEMENT_PLOT}")

    print("\nLATENCY CHANGE: 5K -> 10K")

    for row in output_rows:
        change = row["Latency change (%)"]
        value = f"{change:+.2f}%" if change != "" else "N/A"
        print(f"{row['Strategy']:15} {value}")

    print("\nINDEX-SIZE CHANGE: 5K -> 10K")

    for row in output_rows:
        change = row["Index size change (%)"]
        value = f"{change:+.2f}%" if change != "" else "N/A"
        print(f"{row['Strategy']:15} {value}")

    print("\nFLAT AGREEMENT@5 (NOT RELEVANCE RECALL)")

    for row in output_rows:
        print(
            f"{row['Strategy']:15} "
            f"5K={row['Flat agreement@5 5K']:.4f}, "
            f"10K={row['Flat agreement@5 10K']:.4f}"
        )

    print("\nIMPORTANT INTERPRETATION NOTES")
    print("- Flat is the exact-search baseline over the full corpus.")
    print("- IVF-PQ is approximate search over the full corpus.")
    print("- Parent-Child retrieves sentence vectors and maps them to parents.")
    print("- Summary uses one summary vector per source passage.")
    print("- RAPTOR uses base passage vectors plus cluster-summary vectors.")
    print("- Summary and RAPTOR now cover the full corpus at both sizes.")
    print("- Flat agreement@5 measures agreement with Flat's top-5 results.")
    print("- Flat agreement@5 is not ground-truth semantic relevance.")
    print("- Source mapping coverage measures successfully mapped index vectors.")
    print("- Latency varies with hardware, workload, and index configuration.")
if __name__ == "__main__":
    main()

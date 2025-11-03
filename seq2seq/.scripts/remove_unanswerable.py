import argparse
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import csv

UNANSWERABLE = {"unanswerable_medical", "unanswerable_non_medical"}

def get_class_histogram(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    db_id_counts = Counter(item.get("db_id") for item in data)
    return db_id_counts

def filter_and_count(json_path, save_filtered=True):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    db_id_counts = Counter()
    filtered = []
    removed = 0
    for item in data:
        db_id = item.get("db_id")
        if db_id in UNANSWERABLE:
            removed += 1
            continue
        filtered.append(item)
        db_id_counts[db_id] += 1
    if save_filtered:
        out_path = json_path.with_name(json_path.stem.replace("_filtered", "") + "_answerable.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
    return db_id_counts, len(data), removed, len(filtered)

def save_histogram(db_id_counts, output_path, title="Class Histogram"):
    labels, values = zip(*sorted(db_id_counts.items(), key=lambda x: x[0]))
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Save histogram and filter stats for train.json/dev.json.")
    parser.add_argument("data_dir", type=Path, help="Folder containing train.json and dev.json")
    args = parser.parse_args()

    files = ["train.json", "dev.json"]
    combined_hist = Counter()
    stats = []

    for fname in files:
        path = args.data_dir / fname
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue

        # Full histogram including unanswerable
        full_counts = get_class_histogram(path)
        save_histogram(
            full_counts, 
            args.data_dir / f"{path.stem}_class_hist.png",
            title=f"{fname} Class Histogram"
        )
        combined_hist += full_counts

        # Filter and collect statistics
        counts, n_total, n_removed, n_kept = filter_and_count(path)
        percent_removed = (n_removed / n_total) * 100 if n_total else 0
        stats.append({
            "file": fname,
            "total": n_total,
            "filtered_out": n_removed,
            "kept": n_kept,
            "filtered_percent": f"{percent_removed:.2f}"
        })

    # Save combined histogram
    save_histogram(
        combined_hist, 
        args.data_dir / "combined_class_hist.png",
        title="Combined Class Histogram"
    )

    # Add combined stats
    total = sum(int(s["total"]) for s in stats)
    total_removed = sum(int(s["filtered_out"]) for s in stats)
    total_kept = sum(int(s["kept"]) for s in stats)
    percent_removed = (total_removed / total) * 100 if total else 0
    stats.append({
        "file": "combined",
        "total": total,
        "filtered_out": total_removed,
        "kept": total_kept,
        "filtered_percent": f"{percent_removed:.2f}"
    })

    # Save CSV
    csv_path = args.data_dir / "filter_stats.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file", "total", "filtered_out", "kept", "filtered_percent"])
        writer.writeheader()
        for s in stats:
            writer.writerow(s)

if __name__ == "__main__":
    main()

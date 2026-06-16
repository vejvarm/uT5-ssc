import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def merge_chunk_files(chunk_files: List[Path]) -> List[Dict[str, Any]]:
    records_by_index: Dict[int, Dict[str, Any]] = {}
    for chunk_file in chunk_files:
        payload = json.loads(chunk_file.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {chunk_file}")
        for record in payload:
            if "_source_record_index" not in record:
                raise ValueError(f"Missing _source_record_index in {chunk_file}")
            index = int(record["_source_record_index"])
            if index in records_by_index:
                raise ValueError(f"Duplicate _source_record_index {index}")
            records_by_index[index] = record
    return [records_by_index[index] for index in sorted(records_by_index)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge chunked execution-analysis JSON files into one ordered file."
    )
    parser.add_argument("chunk_files", nargs="+", type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    merged = merge_chunk_files(args.chunk_files)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()

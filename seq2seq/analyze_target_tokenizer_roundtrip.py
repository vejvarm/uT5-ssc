from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import AddedToken
from transformers.models.auto import AutoTokenizer
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from third_party.test_suite import exec_eval

LANGUAGES = ("sql", "sparql", "cypher")
T5_EXTRA_TOKENS = [" {", " }", " <=", " <", "^^"]
LEXICAL_ITEM_RE = re.compile(
    r"\?[A-Za-z_][A-Za-z0-9_]*|:[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*|\S"
)
WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> dict[str, float]:
    if total <= 0:
        return {"lower": 0.0, "upper": 0.0}
    p_hat = successes / total
    denominator = 1 + (z * z / total)
    centre = p_hat + (z * z / (2 * total))
    margin = z * math.sqrt((p_hat * (1 - p_hat) + (z * z / (4 * total))) / total)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    if math.isclose(upper, 1.0):
        upper = 1.0
    return {
        "lower": max(0.0, lower),
        "upper": min(1.0, upper),
    }


def normalize_for_match(query: str) -> str:
    return " ".join(exec_eval.postprocess(query).split())


def count_lexical_items(query: str) -> int:
    return len(LEXICAL_ITEM_RE.findall(query))


def p90(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = max(0, math.ceil(0.9 * len(sorted_values)) - 1)
    return sorted_values[index]


def summarize_token_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for language, language_rows in grouped.items():
        token_counts = [int(row["token_count"]) for row in language_rows]
        lexical_counts = [int(row["lexical_item_count"]) for row in language_rows]
        token_ratios = [
            token_count / max(lexical_count, 1)
            for token_count, lexical_count in zip(token_counts, lexical_counts)
        ]
        summary[language] = {
            "language": language,
            "n": len(language_rows),
            "mean_tokens": mean(token_counts),
            "median_tokens": median(token_counts),
            "p90_tokens": p90(token_counts),
            "max_tokens": max(token_counts),
            "mean_lexical_items": mean(lexical_counts),
            "median_lexical_items": median(lexical_counts),
            "mean_tokens_per_lexical_item": mean(token_ratios),
        }
    return summary


def summarize_roundtrip(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for language, language_rows in grouped.items():
        n = len(language_rows)
        raw_matches = sum(1 for row in language_rows if row["raw_exact_match"])
        normalized_matches = sum(1 for row in language_rows if row["normalized_match"])
        execution_equivalent = sum(
            1 for row in language_rows if row["execution_equivalent"]
        )
        execution_errors = sum(1 for row in language_rows if row.get("execution_error"))
        summary[language] = {
            "language": language,
            "n": n,
            "raw_exact_matches": raw_matches,
            "raw_exact_match_rate": raw_matches / n if n else 0.0,
            "normalized_matches": normalized_matches,
            "normalized_match_rate": normalized_matches / n if n else 0.0,
            "execution_equivalent": execution_equivalent,
            "execution_equivalent_rate": execution_equivalent / n if n else 0.0,
            "execution_failures": n - execution_equivalent,
            "execution_errors": execution_errors,
            "execution_equivalent_ci95": wilson_interval(execution_equivalent, n),
        }
    return summary


def score_execution_equivalence(
    db_root: Path,
    example: dict[str, Any],
    language: str,
    decoded_query: str,
) -> int:
    db_id = example["db_id"]
    extension = ".sqlite" if language == "sql" else ".ttl"
    db_path = db_root / db_id / f"{db_id}{extension}"
    return exec_eval.eval_exec_match(
        db=str(db_path),
        p_str=decoded_query,
        g_str=example[language],
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
        lang=language,
    )


def load_tokenizer(tokenizer_name_or_path: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError("Only fast tokenizers are supported")
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(tok, normalized=True) for tok in T5_EXTRA_TOKENS])
    return tokenizer


def load_examples(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    split_file = dataset_root / f"{split}.json"
    with split_file.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list of examples in {split_file}")
    return rows


def audit_examples(
    examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    db_root: Path,
    languages: tuple[str, ...] = LANGUAGES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example_id, example in enumerate(examples):
        for language in languages:
            gold_query = example.get(language) or ""
            if not gold_query:
                continue

            encoded = tokenizer.encode(gold_query, add_special_tokens=True)
            decoded_query = tokenizer.decode(encoded, skip_special_tokens=True)
            lexical_count = count_lexical_items(gold_query)
            normalized_original = normalize_for_match(gold_query)
            normalized_decoded = normalize_for_match(decoded_query)

            execution_error = ""
            try:
                execution_score = score_execution_equivalence(
                    db_root=db_root,
                    example=example,
                    language=language,
                    decoded_query=decoded_query,
                )
            except Exception as error:
                execution_score = 0
                execution_error = f"{type(error).__name__}: {error}"

            rows.append(
                {
                    "example_id": example_id,
                    "db_id": example["db_id"],
                    "language": language,
                    "original_query": gold_query,
                    "decoded_query": decoded_query,
                    "token_count": len(encoded),
                    "lexical_item_count": lexical_count,
                    "tokens_per_lexical_item": len(encoded) / max(lexical_count, 1),
                    "raw_exact_match": gold_query == decoded_query,
                    "normalized_match": normalized_original == normalized_decoded,
                    "execution_equivalent": bool(execution_score),
                    "execution_error": execution_error,
                }
            )
    return rows


def write_token_stats_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "language",
        "n",
        "mean_tokens",
        "median_tokens",
        "p90_tokens",
        "max_tokens",
        "mean_lexical_items",
        "median_lexical_items",
        "mean_tokens_per_lexical_item",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for language in LANGUAGES:
            if language in summary:
                writer.writerow(summary[language])


def write_examples_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _shorten(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_examples_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    token_summary: dict[str, dict[str, Any]],
    roundtrip_summary: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Target-Query Tokenizer Round-Trip Audit",
        "",
        "## Token Statistics",
        "",
        "| language | n | mean tokens | median | p90 | max | mean tokens / lexical item |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language in LANGUAGES:
        if language not in token_summary:
            continue
        stats = token_summary[language]
        lines.append(
            f"| {language} | {stats['n']} | {stats['mean_tokens']:.2f} | "
            f"{stats['median_tokens']:.2f} | {stats['p90_tokens']} | "
            f"{stats['max_tokens']} | {stats['mean_tokens_per_lexical_item']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Round-Trip Execution Equivalence",
            "",
            "| language | n | exact match | normalized match | execution equivalent | 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for language in LANGUAGES:
        if language not in roundtrip_summary:
            continue
        stats = roundtrip_summary[language]
        ci = stats["execution_equivalent_ci95"]
        lines.append(
            f"| {language} | {stats['n']} | {stats['raw_exact_match_rate']:.4f} | "
            f"{stats['normalized_match_rate']:.4f} | "
            f"{stats['execution_equivalent_rate']:.4f} | "
            f"[{ci['lower']:.4f}, {ci['upper']:.4f}] |"
        )

    lines.extend(["", "## Representative Successes", ""])
    for language in LANGUAGES:
        success = next(
            (
                row
                for row in rows
                if row["language"] == language and row["execution_equivalent"]
            ),
            None,
        )
        if success is None:
            continue
        lines.extend(
            [
                f"### {language}",
                "",
                f"- db: `{success['db_id']}`, example: `{success['example_id']}`",
                f"- original: `{_shorten(success['original_query'])}`",
                f"- decoded: `{_shorten(success['decoded_query'])}`",
                "",
            ]
        )

    failures = [row for row in rows if not row["execution_equivalent"]]
    if failures:
        lines.extend(["## Failures", ""])
        for language in LANGUAGES:
            language_failures = [row for row in failures if row["language"] == language][:10]
            if not language_failures:
                continue
            lines.extend([f"### {language}", ""])
            for row in language_failures:
                error = f", error: `{row['execution_error']}`" if row["execution_error"] else ""
                lines.extend(
                    [
                        f"- db: `{row['db_id']}`, example: `{row['example_id']}`{error}",
                        f"  - original: `{_shorten(row['original_query'])}`",
                        f"  - decoded: `{_shorten(row['decoded_query'])}`",
                    ]
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    token_summary = summarize_token_stats(rows)
    roundtrip_summary = summarize_roundtrip(rows)

    write_token_stats_csv(output_dir / "target_token_stats.csv", token_summary)
    (output_dir / "roundtrip_summary.json").write_text(
        json.dumps(roundtrip_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_examples_jsonl(output_dir / "roundtrip_examples.jsonl", rows)
    write_examples_markdown(
        output_dir / "roundtrip_examples.md",
        rows,
        token_summary,
        roundtrip_summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit T5 tokenizer round-trip effects on Spider4SSC target queries.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Spider4SSC"))
    parser.add_argument("--split", choices=["dev"], default="dev")
    parser.add_argument(
        "--tokenizer-name-or-path",
        default=os.getenv("TOKENIZER_NAME_OR_PATH"),
        help="Tokenizer path/name. Defaults to TOKENIZER_NAME_OR_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to results/tokenizer_roundtrip/<split>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tokenizer_name_or_path:
        raise SystemExit(
            "--tokenizer-name-or-path or TOKENIZER_NAME_OR_PATH is required"
        )

    output_dir = args.output_dir or Path("results/tokenizer_roundtrip") / args.split
    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    examples = load_examples(args.dataset_root, args.split)
    rows = audit_examples(
        examples=examples,
        tokenizer=tokenizer,
        db_root=args.dataset_root / "database",
    )
    write_outputs(output_dir, rows)
    print(f"Wrote tokenizer round-trip audit to {output_dir}")


if __name__ == "__main__":
    main()

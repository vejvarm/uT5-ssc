#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")


def find_trainer_state_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name == "trainer_state.json" else []
    if root.is_dir():
        return sorted(root.rglob("trainer_state.json"))
    return []


def discover_run_dirs(trainer_state_files: list[Path]) -> set[Path]:
    run_dirs: set[Path] = set()
    for file_path in trainer_state_files:
        parent = file_path.parent
        match = CHECKPOINT_DIR_RE.fullmatch(parent.name)
        if match:
            run_dirs.add(parent.parent)
            continue

        # Main trainer_state.json for a run should live in a folder
        # that has checkpoint-* subdirectories.
        has_checkpoint_children = any(
            child.is_dir() and CHECKPOINT_DIR_RE.fullmatch(child.name)
            for child in parent.iterdir()
        )
        if has_checkpoint_children:
            run_dirs.add(parent)
    return run_dirs


def get_highest_checkpoint(run_dir: Path) -> tuple[int | None, Path | None]:
    highest_num: int | None = None
    highest_path: Path | None = None
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_DIR_RE.fullmatch(child.name)
        if not match:
            continue
        num = int(match.group(1))
        if highest_num is None or num > highest_num:
            highest_num = num
            highest_path = child
    return highest_num, highest_path


def collect_run_trainer_state_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    main_file = run_dir / "trainer_state.json"
    if main_file.exists():
        files.append(main_file)

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if not CHECKPOINT_DIR_RE.fullmatch(child.name):
            continue
        checkpoint_file = child / "trainer_state.json"
        if checkpoint_file.exists():
            files.append(checkpoint_file)

    return files


def process_trainer_state(file_path: Path, target_checkpoint_path: str, dry_run: bool) -> str:
    """
    Returns status:
    - updated
    - already_target
    - error: <message>
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return f"error: {exc}"

    old_value = data.get("best_model_checkpoint")
    if old_value == target_checkpoint_path:
        return "already_target"

    if not dry_run:
        data["best_model_checkpoint"] = target_checkpoint_path
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    return "updated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find trainer_state.json files and synchronize each run's "
            "best_model_checkpoint to the highest available checkpoint in that run folder."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Folder (or trainer_state.json file) to process recursively.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help=(
            "Write best_model_checkpoint as a path relative to current working directory. "
            "By default, absolute paths are written."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    discovered_files = find_trainer_state_files(args.root)
    if not discovered_files:
        print(f"No trainer_state.json files found under: {args.root}")
        return 1

    run_dirs = sorted(discover_run_dirs(discovered_files))
    if not run_dirs:
        print("No run directories with checkpoint-* subfolders found.")
        return 1

    counts = Counter()
    action = "WOULD_UPDATE" if args.dry_run else "UPDATED"

    for run_dir in run_dirs:
        max_num, max_path = get_highest_checkpoint(run_dir)
        if max_num is None or max_path is None:
            counts["runs_without_checkpoints"] += 1
            print(f"SKIP RUN {run_dir}: no checkpoint-* subfolders found")
            continue

        target = max_path.resolve()
        if args.relative:
            target_path_str = os.path.relpath(target, start=Path.cwd().resolve())
        else:
            target_path_str = str(target)

        run_files = collect_run_trainer_state_files(run_dir)
        if not run_files:
            counts["runs_without_trainer_state"] += 1
            print(f"SKIP RUN {run_dir}: no trainer_state.json files found")
            continue

        print(f"RUN {run_dir}")
        print(f"  highest_checkpoint: checkpoint-{max_num}")
        print(f"  target_path: {target_path_str}")

        counts["runs_processed"] += 1
        for file_path in run_files:
            status = process_trainer_state(
                file_path=file_path,
                target_checkpoint_path=target_path_str,
                dry_run=args.dry_run,
            )
            counts[status] += 1

            if status == "updated":
                print(f"  {action} {file_path}")
            elif status == "already_target":
                print(f"  OK      {file_path}")
            else:
                print(f"  ERROR   {file_path}: {status}")

    print("\nSummary")
    print(f"  discovered_trainer_state_files: {len(discovered_files)}")
    print(f"  runs_processed: {counts['runs_processed']}")
    print(f"  runs_without_checkpoints: {counts['runs_without_checkpoints']}")
    print(f"  runs_without_trainer_state: {counts['runs_without_trainer_state']}")
    print(f"  updated: {counts['updated']}")
    print(f"  already_target: {counts['already_target']}")
    error_count = sum(v for k, v in counts.items() if k.startswith("error:"))
    print(f"  errors: {error_count}")

    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse
import itertools

sns.set_theme(style="whitegrid")


def smooth_series(series, window: int = 1):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def get_lang_and_schema_from_checkpoint_path(checkpoint_path_str):
    m = re.findall(
        r"(sparql|sql|cypher)[\\/].*schema[\\/](compact|norange|no_schema)",
        checkpoint_path_str,
        re.IGNORECASE,
    )
    if m:
        lang, schema = m[0]
        return lang.lower(), schema.lower()
    parts = Path(checkpoint_path_str).parts
    lang = None
    schema = None
    for p in parts:
        if p.lower() in ("sparql", "sql", "cypher"):
            lang = p.lower()
        if p.lower() in ("compact", "norange", "no_schema"):
            schema = p.lower()
    return lang, schema


def find_latest_trainer_state_files(root):
    trainer_state_paths = list(root.glob("**/checkpoint-*/trainer_state.json"))
    subfolder_to_best = {}
    for p in trainer_state_paths:
        run_subfolder = p.parent.parent
        try:
            checkpoint_num = int(p.parent.name.split("-")[-1])
        except Exception:
            continue
        key = str(run_subfolder)
        if key not in subfolder_to_best or checkpoint_num > subfolder_to_best[key][1]:
            subfolder_to_best[key] = (p, checkpoint_num)
    return {k: v[0] for k, v in subfolder_to_best.items()}


def parse_log_history(trainer_state_path, exp_name):
    with open(trainer_state_path, "r") as f:
        d = json.load(f)
    log = d["log_history"]
    best_ckpt_path = d.get("best_model_checkpoint", "")
    lang, schema = get_lang_and_schema_from_checkpoint_path(best_ckpt_path)
    return {
        "lang": lang,
        "schema": schema,
        "log_history": log,
        "best_model_checkpoint": best_ckpt_path,
        "trainer_state_path": trainer_state_path,
        "exp_name": exp_name,  # label for plots/tables
        "run_id": f"{exp_name}_{lang}_{schema}" if lang else str(trainer_state_path.parent),
    }


def build_metric_dfs(parsed_runs):
    dfs = {}
    for parsed in parsed_runs:
        log = parsed["log_history"]
        train_rows = [e for e in log if "loss" in e and "step" in e and "epoch" in e]
        eval_rows = []
        for e in log:
            if "eval_loss" in e and "eval_exec" in e:
                if "epoch" not in e and "step" in e:
                    e["epoch"] = e["step"]
                eval_rows.append(e)

        df_train = pd.DataFrame(train_rows)
        df_eval = pd.DataFrame(eval_rows)

        if not df_train.empty:
            df_train["epoch"] = pd.to_numeric(df_train["epoch"], errors="coerce")
            df_train = df_train.sort_values("epoch")
            df_train["perplexity"] = np.exp(df_train["loss"])
        if not df_eval.empty:
            df_eval["epoch"] = pd.to_numeric(df_eval["epoch"], errors="coerce")
            df_eval = df_eval.sort_values("epoch")
            df_eval["perplexity"] = np.exp(df_eval["eval_loss"])

        merged = None
        if not df_train.empty and not df_eval.empty:
            merged = pd.merge_asof(
                df_train.sort_values("epoch"),
                df_eval.sort_values("epoch"),
                on="epoch",
                direction="nearest",
                suffixes=("_train", "_eval"),
            )

        dfs[parsed["run_id"]] = {
            "train": df_train.reset_index(drop=True),
            "eval": df_eval.reset_index(drop=True),
            "all": merged,
            "lang": parsed["lang"],
            "schema": parsed["schema"],
            "exp_name": parsed["exp_name"],
            "run_id": parsed["run_id"],
        }
    return dfs


def _get_colors_bright():
    return {
        "cypher": "#4477AA",  # blue
        "sparql": "#228833",  # green
        "sql": "#CCBB44",     # yellow
    }


def plot_all_runs(dfs_by_id, res_root: Path, schema_filter: str, languages_to_plot=("sparql", "sql", "cypher"), smooth_window=1):
    lang_palette = _get_colors_bright()
    line_styles = ["-", "--", "-.", ":"]
    style_cycle = itertools.cycle(line_styles)

    # Assign line styles globally across experiments so they match for all languages
    exp_names = sorted(set(dfs["exp_name"] for dfs in dfs_by_id.values()))
    exp_to_style = {exp: style for exp, style in zip(exp_names, style_cycle)}

    for lang in languages_to_plot:
        plt.figure(figsize=(8, 6))
        for run_id, dfs in dfs_by_id.items():
            if dfs["lang"] != lang or dfs["schema"] != schema_filter:
                continue
            df = dfs["eval"]
            if not df.empty:
                y = smooth_series(df["eval_exec"], smooth_window)
                sns.lineplot(
                    x=df["epoch"],
                    y=y,
                    label=dfs["exp_name"],
                    color=lang_palette.get(lang, "black"),
                    linestyle=exp_to_style[dfs["exp_name"]],
                )
        plt.title(f"Eval Execution Accuracy vs. Epoch ({lang}, {schema_filter})")
        plt.xlabel("Epoch")
        plt.ylabel("Execution Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_root.joinpath(f"eval_exec_vs_epoch_{lang}_{schema_filter}.png"), dpi=200)
        plt.close()


def print_max_exec_table(dfs_by_id, res_root: Path, schema_filter: str):
    import io
    lang_order = ["sparql", "sql", "cypher"]

    records = []
    for run_id, dfs in dfs_by_id.items():
        lang, exp, schema = dfs["lang"], dfs["exp_name"], dfs["schema"]
        df = dfs["eval"]
        if not df.empty and lang in lang_order and schema == schema_filter:
            max_exec = float(df["eval_exec"].max()) * 100.0
            records.append({"lang": lang, "exp": exp, "max_eval_exec": max_exec})

    df = pd.DataFrame(records)
    pivot = df.pivot(index="lang", columns="exp", values="max_eval_exec")
    pivot = pivot.reindex(index=lang_order)

    pivot["avg"] = pivot.mean(axis=1)
    avg_row = pivot.mean(axis=0)
    avg_row.name = "avg"
    tab = pd.concat([pivot, pd.DataFrame([avg_row])])

    out = io.StringIO()
    out.write("\\begin{table}[htb]\n\\centering\n")
    out.write(f"\\caption{{Execution accuracy (\\%) for schema={schema_filter}.}}\n")
    out.write("\\label{tab:exec-acc-multirun}\n")
    out.write("\\begin{tabular}{|r|" + "c" * len(tab.columns) + "|}\n\\hline\n")
    header = " & ".join([str(c) for c in tab.columns])
    out.write(f" & {header} \\\\ \\hline\n")
    for lang in tab.index:
        vals = [f"{tab.loc[lang,c]:.1f}" if not pd.isna(tab.loc[lang,c]) else "" for c in tab.columns]
        out.write(f"{lang:<10} & " + " & ".join(vals) + " \\\\\n")
    out.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    latex = out.getvalue()
    with open(res_root.joinpath(f"max_eval_exec_summary_{schema_filter}.tex"), "w") as f:
        f.write(latex)
    tab.to_csv(res_root.joinpath(f"max_eval_exec_summary_{schema_filter}.csv"))


def main(args):
    dfs_by_id = {}

    root = Path(args.results_root)
    for exp_name in args.subfolders:
        exp_path = root.joinpath(exp_name)
        trainer_state_files = find_latest_trainer_state_files(exp_path)
        for ts_path in trainer_state_files.values():
            parsed = parse_log_history(ts_path, exp_name)
            if parsed["lang"] is not None:
                dfs_by_id[parsed["run_id"]] = build_metric_dfs([parsed])[parsed["run_id"]]

    plot_root = Path("plots_multi")
    plot_root.mkdir(exist_ok=True)

    plot_all_runs(dfs_by_id, plot_root, args.schema, args.languages_to_plot, args.smooth_window)
    print_max_exec_table(dfs_by_id, plot_root, args.schema)

    print(f"\nPlots saved under {plot_root}, schema={args.schema}. Table summaries written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fine-tuning runs across subfolders under one root for one schema.")
    parser.add_argument("--results-root", type=str, required=True,
                        help="Root folder containing all experiment subfolders.")
    parser.add_argument("--subfolders", nargs="+", required=True,
                        help="Names of subfolders under results-root (these become columns/legends).")
    parser.add_argument("--schema", type=str, choices=["compact", "norange", "no_schema"], default="compact",
                        help="Which schema type to include (compact, norange, no_schema).")
    parser.add_argument("--languages-to-plot", nargs="+", default=["sparql", "sql", "cypher"])
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()
    main(args)

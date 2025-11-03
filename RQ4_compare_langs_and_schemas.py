import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse

sns.set_theme(style="whitegrid")


def smooth_series(series, window: int = 1):
    """
    Apply a simple moving average with the given window to a Pandas Series.
    window=1 returns the original.
    """
    if window <= 1:
        return series
    # Centered; drop NaNs introduced at beginning/end
    return series.rolling(window=window, min_periods=1, center=True).mean()


# Helper: extract language and schema from a best_model_checkpoint path
def get_lang_and_schema_from_checkpoint_path(checkpoint_path_str):
    m = re.findall(r"(sparql|sql|cypher)[\\/].*schema[\\/](compact|norange|no_schema)", checkpoint_path_str, re.IGNORECASE)
    if m:
        lang, schema = m[0]
        return lang.lower(), schema.lower()
    # fallback: try to extract directly (may need to adjust pattern!)
    parts = Path(checkpoint_path_str).parts
    lang = None
    schema = None
    for i, p in enumerate(parts):
        if p.lower() in ('sparql', 'sql', 'cypher'):
            lang = p.lower()
        if p.lower() in ('compact', 'norange', 'no_schema'):
            schema = p.lower()
    return lang, schema

def find_latest_trainer_state_files(root):
    trainer_state_paths = list(root.glob("**/checkpoint-*/trainer_state.json"))
    subfolder_to_best = dict()
    for p in trainer_state_paths:
        run_subfolder = p.parent.parent
        # Protect for old checkpoints accidentally left behind without number
        try:
            checkpoint_num = int(p.parent.name.split('-')[-1])
        except Exception:
            continue
        key = str(run_subfolder)
        # For each training run, keep the checkpoint with highest number
        if key not in subfolder_to_best or checkpoint_num > subfolder_to_best[key][1]:
            subfolder_to_best[key] = (p, checkpoint_num)
    return {k: v[0] for k, v in subfolder_to_best.items()}

def parse_log_history(trainer_state_path):
    with open(trainer_state_path, 'r') as f:
        d = json.load(f)
    log = d['log_history']
    best_ckpt_path = d.get('best_model_checkpoint', '')
    lang, schema = get_lang_and_schema_from_checkpoint_path(best_ckpt_path)
    return {
        "lang": lang,
        "schema": schema,
        "log_history": log,
        "best_model_checkpoint": best_ckpt_path,
        "trainer_state_path": trainer_state_path,
        "run_id": f"{lang}_{schema}" if lang and schema else str(trainer_state_path.parent),
    }

def build_metric_dfs(parsed_runs):
    dfs = {}
    for parsed in parsed_runs:
        log = parsed['log_history']
        # Train: 'loss', 'step', 'epoch' must exist
        train_rows = [e for e in log if 'loss' in e and 'step' in e and 'epoch' in e]
        # Eval: 'eval_loss', 'eval_exec', and 'epoch' (guaranteed in good logs, but let's fallback from 'step' if missing)
        eval_rows  = []
        for e in log:
            if 'eval_loss' in e and 'eval_exec' in e:
                if 'epoch' not in e and 'step' in e:
                    # You can try to infer epoch if your steps/epochs mapping is regular, else set None
                    e['epoch'] = e['step']
                eval_rows.append(e)
        df_train = pd.DataFrame(train_rows)
        df_eval = pd.DataFrame(eval_rows)

        # Make sure 'epoch' is present and numeric
        if not df_train.empty:
            df_train['epoch'] = pd.to_numeric(df_train['epoch'], errors='coerce')
            df_train = df_train.sort_values('epoch', ascending=True)
            df_train['perplexity'] = np.exp(df_train['loss'])
        if not df_eval.empty:
            df_eval['epoch'] = pd.to_numeric(df_eval['epoch'], errors='coerce')
            df_eval = df_eval.sort_values('epoch', ascending=True)
            df_eval['perplexity'] = np.exp(df_eval['eval_loss'])
        # Join for convenience, but keep 'epoch' column
        merged = None
        if not df_train.empty and not df_eval.empty:
            merged = pd.merge_asof(
                df_train.sort_values('epoch'), 
                df_eval.sort_values('epoch'),
                on='epoch', 
                direction='nearest', 
                suffixes=("_train", "_eval")
            )
        dfs[parsed['run_id']] = {
            "train": df_train.reset_index(drop=True),
            "eval":  df_eval.reset_index(drop=True),
            "all":   merged,
            "lang":  parsed['lang'],
            "schema":parsed['schema'],
            "run_id":parsed['run_id'],
        }
    return dfs

def _get_colors_bright():
    # Language → RGB color
    return {
        "cypher":  [0x44/255, 0x77/255, 0xAA/255],  # blue
        "sparql":  [0x22/255, 0x88/255, 0x33/255],  # green
        "sql":     [0xCC/255, 0xBB/255, 0x44/255],  # yellow
    }

def plot_all_runs(
    dfs_by_id, 
    res_root: Path, 
    schemas_to_plot = ("compact", "norange", "no_schema"),
    languages_to_plot = ("sparql", "sql", "cypher"),
    smooth_window: int = 1,
    xmax: int = None,
    ymax: int = None
):
    lang_palette = _get_colors_bright()

    def _should_plot(dfs):
        return (dfs['lang'] in languages_to_plot) and (dfs['schema'] in schemas_to_plot)

    # ---- TRAINING LOSS ----
    plt.figure(figsize=(8,6))
    for run_id, dfs in dfs_by_id.items():
        if not _should_plot(dfs):
            continue
        df = dfs['train']
        label = f"{dfs['lang']} - {dfs['schema']}"
        color = lang_palette.get(dfs['lang'], None)
        if not df.empty:
            y = smooth_series(df["loss"], smooth_window)
            sns.lineplot(x=df["epoch"], y=y, label=label, color=color)
    plt.title("Training Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_root.joinpath("training_loss_vs_epoch.png"), dpi=200)
    plt.close()

    # ---- TRAINING PERPLEXITY ----
    plt.figure(figsize=(8,6))
    mean_perplexity = 0.
    for run_id, dfs in dfs_by_id.items():
        if not _should_plot(dfs):
            continue
        df = dfs['train']
        if not df.empty:
            mean_perplexity = max(df["perplexity"].mean(), mean_perplexity)
            label = f"{dfs['lang']} - {dfs['schema']}"
            color = lang_palette.get(dfs['lang'], None)
            y = smooth_series(df["perplexity"], smooth_window)
            sns.lineplot(x=df["epoch"], y=y, label=label, color=color)
    plt.axhline(1, color="red", linestyle=(0, (5, 5)), label="perplexity=1")
    plt.title("Training Perplexity vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.xlim(0, xmax)
    ymax_perplex = ymax
    if ymax_perplex is None:
        ymax_perplex = mean_perplexity+1
    plt.ylim(0, ymax_perplex)
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_root.joinpath("train_perplexity_vs_epoch.png"), dpi=200)
    plt.close()

    # ---- EVAL LOSS ----
    plt.figure(figsize=(8,6))
    for run_id, dfs in dfs_by_id.items():
        if not _should_plot(dfs):
            continue
        df = dfs['eval']
        label = f"{dfs['lang']} - {dfs['schema']}"
        color = lang_palette.get(dfs['lang'], None)
        if not df.empty:
            y = smooth_series(df["eval_loss"], smooth_window)
            sns.lineplot(x=df["epoch"], y=y, label=label, color=color)
    plt.title("Eval Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_root.joinpath("eval_loss_vs_epoch.png"), dpi=200)
    plt.close()

    # ---- EVAL PERPLEXITY ----
    plt.figure(figsize=(8,6))
    mean_perplexity = 0.
    for run_id, dfs in dfs_by_id.items():
        if not _should_plot(dfs):
            continue
        df = dfs['eval']
        if not df.empty:
            mean_perplexity = max(df["perplexity"].mean(), mean_perplexity)
            label = f"{dfs['lang']} - {dfs['schema']}"
            color = lang_palette.get(dfs['lang'], None)
            y = smooth_series(df["perplexity"], smooth_window)
            sns.lineplot(x=df["epoch"], y=y, label=label, color=color)
    plt.axhline(1, color="red", linestyle=(0, (5, 5)), label="perplexity=1")
    plt.title("Eval Perplexity vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.xlim(0, xmax)
    ymax_perplex = ymax
    if ymax_perplex is None:
        ymax_perplex = mean_perplexity + 1
    plt.ylim(0, ymax_perplex)
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_root.joinpath("eval_perplexity_vs_epoch.png"), dpi=200)
    plt.close()

    # ---- EVAL EXEC ----
    plt.figure(figsize=(8,6))
    for run_id, dfs in dfs_by_id.items():
        if not _should_plot(dfs):
            continue
        df = dfs['eval']
        label = f"{dfs['lang']} - {dfs['schema']}"
        color = lang_palette.get(dfs['lang'], None)
        if not df.empty:
            y = smooth_series(df["eval_exec"], smooth_window)
            sns.lineplot(x=df["epoch"], y=y, label=label, color=color)
    plt.title("Eval Execution Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Execution Accuracy")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(res_root.joinpath("eval_exec_vs_epoch.png"), dpi=200)
    plt.close()

    
def print_max_exec_table(dfs_by_id, res_root: Path):
    import io

    # Collect as flat list
    records = []
    schema_order = ['no_schema', 'compact', 'norange']
    lang_order = ['sparql', 'sql', 'cypher']

    for run_id, dfs in dfs_by_id.items():
        lang, schema = dfs['lang'], dfs['schema']
        df = dfs['eval']
        if not df.empty and lang in lang_order and schema in schema_order:
            max_exec = float(df['eval_exec'].max()) * 100.0
            records.append({'lang': lang, 'schema': schema, 'max_eval_exec': max_exec})

    df = pd.DataFrame(records)
    pivot = df.pivot(index='lang', columns='schema', values='max_eval_exec')
    # Fill missing with NaN or 0 if preferred
    pivot = pivot.reindex(index=lang_order, columns=schema_order)
    # Add averages
    pivot['avg'] = pivot.mean(axis=1)
    avg_row = pivot.mean(axis=0)
    avg_row.name = 'avg'
    tab = pd.concat([pivot, pd.DataFrame([avg_row])])

    # LaTeX formatting
    out = io.StringIO()
    out.write("\\begin{table}[htb]\n")
    out.write("\\centering\n")
    out.write("\\caption{Execution accuracy (\\%) of pre-trained and fine-tuned T5-base model.}\n")
    out.write("\\label{tab:exec-acc-t5-base}\n")
    out.write("\\resizebox{\\columnwidth}{!}{%\n")
    out.write("\\begin{tabular}{|r|ccc|c|}\n")
    out.write("\\hline\n")
    out.write("\\multicolumn{1}{|c|}{\\textbf{t5-base (fine-tuned)}} & no\\_schema & schema/compact & schema/norange & \\textbf{avg}  \\\\ \\hline\n")
    for lang in lang_order:
        vals = [tab.loc[lang, sch] for sch in schema_order]
        # Format: fill with .1f or empty string for NaN
        vals_f = [f"{v:.1f}" if not pd.isna(v) else "" for v in vals]
        avg = f"{tab.loc[lang, 'avg']:.1f}" if not pd.isna(tab.loc[lang,'avg']) else ""
        out.write(f"{lang:<51} & " + " & ".join(vals_f) + f" & {avg}  \\\\\n")
    out.write("\\hline\n")
    # final avg row
    rowname = "\\textbf{avg}"
    vals = [f"{tab.loc['avg', sch]:.1f}" if not pd.isna(tab.loc['avg', sch]) else "" for sch in schema_order]
    avgval = f"{tab.loc['avg', 'avg']:.1f}" if not pd.isna(tab.loc['avg', 'avg']) else ""
    out.write(f"{rowname:<51} & " + " & ".join(vals) + f" & \\textbf{{{avgval}}} \\\\ \\hline\n")
    out.write("\\end{tabular}%\n}\n\\end{table}\n")

    latex = out.getvalue()
    print(latex)
    with open(res_root.joinpath("max_eval_exec_summary_latex.txt"), "w") as f:
        f.write(latex)

    # Also output a plain CSV for backup as before
    tab.round(2).to_csv(res_root.joinpath("max_eval_exec_summary_pivoted.csv"))
    print("\nLaTeX table written to max_eval_exec_summary_latex.txt")

def _val_to_hex_color(val, vmin=0, vmax=100, rgb_hi=(0xFF,0x00,0x00), rgb_lo=(150,100,50)):
    '''
    val: value between vmin and vmax, returns hex as xcolor wants (e.g. F9EA79)
    Higher val → closer to rgb_hi, lower → rgb_lo
    '''
    norm = 0 if val is None or np.isnan(val) else min(max((val-vmin)/(vmax-vmin), 0.0), 1.0)
    rgb = [int(lo + (hi-lo)*norm) for lo,hi in zip(rgb_lo, rgb_hi)]
    return '{:02X}{:02X}{:02X}'.format(*rgb)

def print_max_exec_table_colored(dfs_by_id, res_root: Path):
    import io

    records = []
    schema_order = ['no_schema', 'compact', 'norange']
    lang_order = ['sparql', 'sql', 'cypher']

    for run_id, dfs in dfs_by_id.items():
        lang, schema = dfs['lang'], dfs['schema']
        df = dfs['eval']
        if not df.empty and lang in lang_order and schema in schema_order:
            max_exec = float(df['eval_exec'].max()) * 100.0
            records.append({'lang': lang, 'schema': schema, 'max_eval_exec': max_exec})

    df = pd.DataFrame(records)
    pivot = df.pivot(index='lang', columns='schema', values='max_eval_exec')
    pivot = pivot.reindex(index=lang_order, columns=schema_order)
    pivot['avg'] = pivot.mean(axis=1)
    avg_row = pivot.mean(axis=0)
    avg_row.name = 'avg'
    tab = pd.concat([pivot, pd.DataFrame([avg_row])])

    # vmin = 0
    # vmax = 100
    vmin = max(0, df["max_eval_exec"].min(skipna=True))
    vmax = min(100, df["max_eval_exec"].max(skipna=True)*1.5)

    out = io.StringIO()
    out.write("\\begin{table}[htb]\n")
    out.write("\\centering\n")
    out.write("\\caption{Execution accuracy (\\%) of pre-trained and fine-tuned T5-base model.}\n")
    out.write("\\label{tab:exec-acc-t5-base}\n")
    out.write("\\rowcolors{2}{gray!10}{white}\n")
    out.write("\\setlength{\\arrayrulewidth}{0.4mm}\n")
    out.write("\\setlength{\\tabcolsep}{10pt}\n")
    out.write("\\renewcommand{\\arraystretch}{1.4}\n")
    out.write("\\begin{tabular}{|r|ccc|c|}\n")
    out.write("\\hline\n")
    out.write("\\multicolumn{1}{|c|}{\\textbf{t5-base (fine-tuned)}} & no\\_schema & schema/compact & schema/norange & \\textbf{avg}  \\\\ \\hline\n")
    for lang in lang_order:
        vals = []
        for sch in schema_order + ['avg']:
            v = tab.loc[lang, sch]
            if pd.isna(v):
                vals.append("")
            else:
                color = _val_to_hex_color(v, vmin, vmax)
                vals.append(f"\\cellcolor[HTML]{{{color}}}{v:.1f}")
        out.write(f"{lang:<51} & " + " & ".join(vals) + "  \\\\\n")
    out.write("\\hline\n")
    # final avg row
    vals = []
    for sch in schema_order + ['avg']:
        v = tab.loc['avg', sch]
        if pd.isna(v):
            vals.append("")
        else:
            color = _val_to_hex_color(v, vmin, vmax)
            if sch == 'avg':
                vals.append(f"\\cellcolor[HTML]{{{color}}}\\textbf{{{v:.1f}}}")
            else:
                vals.append(f"\\cellcolor[HTML]{{{color}}}{v:.1f}")
    out.write(f"\\textbf{{avg}}{' '*44} & " + " & ".join(vals) + " \\\\ \\hline\n")
    out.write("\\end{tabular}\n\\end{table}\n")

    latex = out.getvalue()
    print(latex)
    with open(res_root.joinpath("max_eval_exec_summary_latex_colored.txt"), "w") as f:
        f.write(latex)
    # Also save CSV
    tab.round(2).to_csv(res_root.joinpath("max_eval_exec_summary_pivoted.csv"))
    print("\nLaTeX table with color written to max_eval_exec_summary_latex_colored.txt")

def main(args):
    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Specified results_root does not exist: {results_root}")
        return

    # 1. Find all latest trainer_state.json
    trainer_state_files = find_latest_trainer_state_files(results_root)
    if not trainer_state_files:
        print(f"No trainer_state.json files found under {results_root.resolve()}!")
        return
    print(f"Found {len(trainer_state_files)} distinct training runs with checkpoints.")

    # 2. Parse log history and identify language/schema for each run
    parsed_runs = []
    for ts_path in trainer_state_files.values():
        parsed = parse_log_history(ts_path)
        if parsed["lang"] is not None and parsed["schema"] is not None:
            parsed_runs.append(parsed)
        else:
            print(f"WARNING: Could not infer lang/schema for {ts_path}, skipping!")

    print(f"Parsed {len(parsed_runs)} runs with extracted lang+schema.")

    # 3. Build DataFrames
    dfs_by_id = build_metric_dfs(parsed_runs)

    # 4. Plot everything
    plot_root = results_root.joinpath(".plots")
    plot_root.mkdir(exist_ok=True)
    plot_all_runs(
        dfs_by_id, 
        plot_root,
        schemas_to_plot = args.schemas_to_plot,
        languages_to_plot = args.languages_to_plot,
        smooth_window = args.smooth_window,
        xmax = args.xmax,
        ymax = args.ymax
    )

    # 5. Tabulate max eval_exec as %
    print_max_exec_table(dfs_by_id, plot_root)
    print_max_exec_table_colored(dfs_by_id, plot_root)

    print("\nAll plots saved as PNG. Tabular summary as max_eval_exec_summary.csv.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fine-tuning progress for uT5-base models.")
    parser.add_argument("results_root", type=str, help="Root folder containing all experiment subfolders (with checkpoint directories).")
    parser.add_argument("--languages-to-plot", nargs='+', default=["sparql", "sql", "cypher"],
                        help="Which languages to include in plots. E.g.: --languages-to-plot sparql cypher")
    parser.add_argument("--schemas-to-plot", nargs='+', default=["compact", "norange", "no_schema"],
                        help="Which schema types to include in plots. E.g.: --schemas-to-plot compact norange")
    parser.add_argument("--smooth-window", type=int, default=1,
                        help="Window size for smoothing plotted curves (moving average). Default: 1 (no smoothing).")
    parser.add_argument("--xmax", type=int, default=None,
                    help="Limit on the maximum X-axis value to plot. Default: None (no limit).")
    parser.add_argument("--ymax", type=float, default=None,
                    help="Limit on the maximum Y-axis value to plot. Default: None (no limit).")
    args = parser.parse_args()
    main(args)
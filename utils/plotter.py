import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib
from .config import PLOT_OUTPUT_DIR, DPI, SUFFIXES, GROUPED_YMAX, PRIORITY_MODEL, MODEL_COLORS

def get_ordered_models(df):
    models = sorted(df["Model"].unique().tolist())
    if PRIORITY_MODEL in models:
        models.remove(PRIORITY_MODEL)
        return [PRIORITY_MODEL] + models
    return models

def get_model_colors(ordered_models):
    palette = {}
    for i, model in enumerate(ordered_models):
        if model in MODEL_COLORS:
            palette[model] = MODEL_COLORS[model]
        else:
            # fallback: use seaborn color cycle
            palette[model] = sns.color_palette()[i % len(sns.color_palette())]
    return palette

def save_plot(fig, filename, suffixes=SUFFIXES):
    for suffix in suffixes:
        fig.savefig(pathlib.Path(PLOT_OUTPUT_DIR).joinpath(filename).with_suffix(suffix), bbox_inches='tight', dpi=DPI)
    plt.close(fig)

def plot_grouped_bar(df, ymax=GROUPED_YMAX):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="QueryLanguage", y="Mean", hue="SchemaType", ax=ax)
    ax.set_ylim(0., ymax)
    ax.set_title("Schema type effect. Average from all models.")
    save_plot(fig, "grouped_bar_by_schema_type.png")

def plot_faceted(df, ymax=GROUPED_YMAX):
    ordered_models = get_ordered_models(df)
    palette = get_model_colors(ordered_models)

    g = sns.catplot(
        data=df,
        x="QueryLanguage",
        y="Mean",
        hue="SchemaType",
        col="Model",
        col_order=ordered_models,
        col_wrap=3,
        kind="bar",
        height=4,
        aspect=1.2,
        sharey=False
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Query Language", "Mean Score")
    g.figure.suptitle("Faceted Bar Plots of Model Performance", y=1.05)
    
    for ax in g.axes.flat:
        ax.set_ylim(0., ymax)
    for suffix in SUFFIXES:
        g.savefig(pathlib.Path(PLOT_OUTPUT_DIR).joinpath("faceted_bar_plots.png").with_suffix(suffix), dpi=DPI)
    plt.close(g.figure)


def plot_heatmaps(df):
    for model in df["Model"].unique():
        pivot = df[df["Model"] == model].pivot(index="QueryLanguage", columns="SchemaType", values="Mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", ax=ax)
        ax.set_title(f"Heatmap for {model}")
        save_plot(fig, f"heatmap_{model.replace('/', '_')}.png")

def plot_avg_by_schema(df):
    avg_df = df.groupby(["Model", "SchemaType"])["Mean"].mean().reset_index()
    ordered_models = get_ordered_models(df)
    palette = get_model_colors(ordered_models)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=avg_df, x="SchemaType", y="Mean", hue="Model", hue_order=ordered_models, palette=palette, ax=ax)
    ax.set_title("Average Model Performance per Schema Type (Sorted by Overall Avg)")
    save_plot(fig, "avg_by_schema_type_sorted.png")

def plot_group_by_model(df, ymax=GROUPED_YMAX):
    schema_types = df["SchemaType"].unique()
    num_schemas = len(schema_types)
    ordered_models = get_ordered_models(df)
    palette = get_model_colors(ordered_models)

    fig, axes = plt.subplots(1, num_schemas, figsize=(5 * num_schemas, 4.5), sharey=True)

    for i, schema in enumerate(schema_types):
        ax = axes[i]
        sub = df[df["SchemaType"] == schema]
        sns.barplot(data=sub, x="QueryLanguage", y="Mean", hue="Model", hue_order=ordered_models, palette=palette, ax=ax)
        ax.set_title(f"Schema: {schema}")
        ax.set_ylim(0., ymax)

        if i == 0:
            # Place legend inside first plot (upper right corner)
            ax.legend(title="Model", loc="upper right", frameon=True)
        else:
            ax.get_legend().remove()

    fig.suptitle("Model-wise Comparison per Schema Type", y=1.02)
    fig.tight_layout()
    save_plot(fig, "group_by_model_combined.png")



def plot_group_by_model_individual(df, ymax=GROUPED_YMAX):
    for schema in df["SchemaType"].unique():
        sub = df[df["SchemaType"] == schema]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=sub, x="QueryLanguage", y="Mean", hue="Model", ax=ax)
        ax.set_title(f"Model-wise Comparison for Schema Type: {schema}")
        ax.set_ylim(0., ymax)
        filename = f"group_by_model_{schema.replace('/', '_')}.png"
        save_plot(fig, filename)
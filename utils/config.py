import pathlib

# Root folder where results are stored
BASE_DIR = pathlib.Path("~/git/uT5-ssc")
RESULTS_DIR = BASE_DIR.joinpath("results")
PLOT_OUTPUT_DIR = BASE_DIR.joinpath("plot_outputs")
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Supported schema types
SCHEMA_TYPES = ["no_schema", "schema/norange", "schema/hybrid", "schema/compact"]

# Supported languages
LANGUAGES = ["sql", "sparql", "cypher"]

# PLOT settings
DPI = 200
GROUPED_YMAX = .8
SUFFIXES = ['.png', '.pdf', '.svg']
MODEL_COLORS = {
    "t5-base (fine-tuned)": "tab:blue",
    # Others will be added dynamically if needed
}
PRIORITY_MODEL = "t5-base (fine-tuned)" # Always put t5-base first

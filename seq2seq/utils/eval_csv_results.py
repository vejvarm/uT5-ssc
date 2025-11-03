import pathlib
import pandas as pd
import re

RESULT_FOLDER = pathlib.Path("./results/") 
input_path = RESULT_FOLDER.joinpath("ssc-results-2025-01-27.csv")
output_path = RESULT_FOLDER.joinpath("ssc-results.xlsx")

# --- 1) Load the data, ignoring blank lines at the end ---
df = pd.read_csv(
    input_path,
    index_col=0,
    skip_blank_lines=True
)

# Make sure the index is named properly (if you like):
df.index.name = "Step"

# --- 2) Parse column names to extract (language, method) ---
# We expect column names like: 't5-spider_SQL-compact - eval/exec'
# We'll remove the prefix "t5-spider_" and the suffix " - eval/exec"
# Then what's left is something like "SQL-compact" or "SPARQL-no-schema", etc.
# We'll use a regex to extract the language and method.

def extract_lang_method(col_name):
    """
    col_name examples:
      "t5-spider_SQL-compact - eval/exec"
      "t5-spider_SPARQL-no-schema - eval/exec"
    We'll capture something like: "SQL|SPARQL|Cypher" and "compact|norange|no-schema"
    """
    pattern = r"t5-spider_(SQL|SPARQL|Cypher)-(compact|norange|no-schema) - eval/exec"
    match = re.match(pattern, col_name.strip())
    if match:
        return match.group(1), match.group(2)  # (language, method)
    else:
        return col_name, col_name  # fallback if it doesn't match

# Create a dictionary for renaming:
new_columns = {}
for c in df.columns:
    lang, method = extract_lang_method(c)
    new_columns[c] = (lang, method)

df.rename(columns=new_columns, inplace=True)

# --- 3) Now df has multi-level columns of the form (language, method).
#         Or if you prefer them as normal columns with tuple names, that works too.
#         Example columns: df.columns = [(SQL, compact), (Cypher, compact), (Cypher, norange), ...]

# --- 4) Create pivot tables for max, min, and mean values across Steps ---
# We want a DataFrame with rows = methods, columns = languages, and values = aggregator.

# First, convert the wide data into a long form so we can group by (language, method).
long_df = df.stack()  # This will move the (language, method) to a new index level
long_df.name = "value"  # name the column of values

# long_df will have a MultiIndex: (Step, (language, method)) -> value
# Let's reset_index so we can group easily:
long_df = long_df.reset_index()
long_df.columns = ["Step", "LanguageMethod", "value"]

# 'LanguageMethod' is a tuple (language, method). We can split it into two separate columns:
long_df["Language"] = long_df["LanguageMethod"].apply(lambda x: x[0])
long_df["Method"]   = long_df["LanguageMethod"].apply(lambda x: x[1])

# Now group by Language and Method for each aggregator:
grouped = long_df.groupby(["Method", "Language"])["value"]

max_series = grouped.max()
min_series = grouped.min()
mean_series = grouped.mean()

# Convert these Series to pivoted DataFrames:
df_max = max_series.unstack(level="Language")
df_min = min_series.unstack(level="Language")
df_mean = mean_series.unstack(level="Language")

# Desired orders
desired_language_order = ["SQL", "SPARQL", "Cypher"]
desired_method_order = ["no-schema", "norange", "compact"]

# Reindex each DataFrame
df_max = df_max.reindex(index=desired_method_order, columns=desired_language_order)
df_min = df_min.reindex(index=desired_method_order, columns=desired_language_order)
df_mean = df_mean.reindex(index=desired_method_order, columns=desired_language_order)

# Now df_max, df_min, and df_mean each have:
# rows -> Method
# columns -> Language
# values -> aggregator (max, min, mean)

# --- 5) Print or otherwise use the resulting DataFrames ---
print("=== Original DataFrame (first few rows) ===")
print(df.head(), "\n")

print("=== Max values pivot (rows=methods, cols=languages) ===")
print(df_max, "\n")

print("=== Min values pivot (rows=methods, cols=languages) ===")
print(df_min, "\n")

print("=== Mean values pivot (rows=methods, cols=languages) ===")
print(df_mean, "\n")

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
    # 1) Raw data (the wide DataFrame)
    df.to_excel(writer, sheet_name="raw")
    
    # 2) Maximum values pivot
    df_max.to_excel(writer, sheet_name="ssc - max")
    
    # 3) Minimum values pivot
    df_min.to_excel(writer, sheet_name="ssc - min")
    
    # 4) Average values pivot
    df_mean.to_excel(writer, sheet_name="ssc - avg")

print(f"DataFrames have been written to {output_path}")

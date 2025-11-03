import os
import pandas as pd
from utils.data_loader import load_vllm_results, load_t5_result
from .config import LANGUAGES, SCHEMA_TYPES

def aggregate_results(results_root):
    rows = []

    for model_dir in os.listdir(results_root):
        model_path = os.path.join(results_root, model_dir)
        if not os.path.isdir(model_path):
            continue

        if model_dir == "t5-base":
            # Handle t5-base directly
            labeled_model = "t5-base (fine-tuned)"
            for lang in LANGUAGES:
                for schema in SCHEMA_TYPES:
                    json_path = os.path.join(model_path, "ssc", lang, schema, "all_results.json")
                    scores = load_t5_result(json_path)
                    if scores:
                        rows.append({
                            "Model": labeled_model,
                            "QueryLanguage": lang,
                            "SchemaType": schema,
                            "Mean": sum(scores) / len(scores),
                            "Variance": pd.Series(scores).var(ddof=0),
                            "StdDev": pd.Series(scores).std(ddof=0)
                        })
        else:
            # Handle vllm-style models (meta-llama, deepseek-ai, etc.)
            for sub_model_name in os.listdir(model_path):
                full_model_path = os.path.join(model_path, sub_model_name)
                if not os.path.isdir(full_model_path):
                    continue
                model_full_id = f"{model_dir}/{sub_model_name}"

                for lang in LANGUAGES:
                    for schema in SCHEMA_TYPES:
                        eval_path = os.path.join(full_model_path, "ssc", lang, schema)
                        scores = load_vllm_results(eval_path)
                        if scores:
                            rows.append({
                                "Model": model_full_id,
                                "QueryLanguage": lang,
                                "SchemaType": schema,
                                "Mean": sum(scores) / len(scores),
                                "Variance": pd.Series(scores).var(ddof=0),
                                "StdDev": pd.Series(scores).std(ddof=0)
                            })

    df = pd.DataFrame(rows)

    # Normalize model names
    df["Model"] = df["Model"].apply(lambda x: x.split("/")[-1])

    return df

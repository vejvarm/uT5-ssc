import os
import json
from glob import glob

def load_vllm_results(base_path):
    """Aggregate scores across runs for vllm models"""
    run_paths = glob(os.path.join(base_path, "evaluation_results_run*", "evaluation_results.json"))
    scores = []
    for run_path in run_paths:
        try:
            with open(run_path) as f:
                data = json.load(f)
                score = data["scores"]["turn 1"]["exec"]
                scores.append(score)
        except Exception as e:
            print(f"Failed to read {run_path}: {e}")
    return scores

def load_t5_result(json_path):
    """Extract the score from a t5 all_results.json"""
    try:
        with open(json_path) as f:
            return [json.load(f)["eval_exec"]]
    except Exception as e:
        print(f"Failed to read {json_path}: {e}")
        return []

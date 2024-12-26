import os
import json
import numpy as np
from opro_optimizer.settings import settings

def generate_metrics():
    metrics = {}
    if "outputs" in os.listdir(settings.ROOT_PATH):
        model_path = os.path.join(settings.ROOT_PATH, "outputs", "optimization-results")
        inferences = [f for f in os.listdir(model_path) if f not in ['__pycache__', '.DS_Store']]
        for model_dir in inferences:
            inference_model = os.path.join(model_dir, "results.json")
            with open(os.path.join(model_path, inference_model), "r") as f:
                results = json.load(f)
                
                num_steps_per_rep = [len(results[rep]["meta_prompts"]) for rep in results.keys()]
                mean_steps = np.round(np.mean(num_steps_per_rep),2)
                std_steps = np.round(np.std(num_steps_per_rep),2)

                unique_pairs = [len(results[rep]["old_value_pairs_with_i_step"]) for rep in results.keys()]
                mean_unique_pairs = np.round(np.mean(unique_pairs),2)
                std_unique_pairs = np.round(np.std(unique_pairs),2)

                metrics[model_dir] = {
                    "mean_steps": mean_steps,
                    "std_steps": std_steps,
                    "mean_unique_pairs": mean_unique_pairs,
                    "std_unique_pairs": std_unique_pairs
                }

    return metrics

def main():
    metrics = generate_metrics()

    with open(os.path.join(settings.ROOT_PATH, "outputs", "metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()

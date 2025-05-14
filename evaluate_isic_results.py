"""
Evaluates results from ISIC 2019 fine-tuning and linear probing experiments.
Conducts paired t-tests to assess statistical significance across JPEG quality levels
and model variations, requiring multiple runs. For single runs, skips t-tests and
summarizes performance. Evaluates performance under degradation settings (JPEG
qualities 90, 50, 20). Generates plots and saves results to JSON.
"""

import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel


def load_results(
    finetune_file="results_metrics_finetune.json",
    linear_probe_file="results_metrics_linear_probe.json",
):
    """
    Loads fine-tuning and linear probing results from JSON files, supporting
    multiple runs or single run per condition.
    """
    results = {}
    for file, mode in [(finetune_file, "finetune"), (linear_probe_file, "linear_probe")]:
        if not Path(file).exists():
            print(f"Warning: {file} not found, skipping {mode} results.")
            continue
        with open(file, "r") as f:
            data = json.load(f)
        for model, qualities in data.items():
            for quality, metrics in qualities.items():
                key = (model, int(quality), mode)
                eval_metrics = metrics["eval_metrics"]
                # Handle single run (dict) or multiple runs (list of dicts)
                if isinstance(eval_metrics, dict):
                    results[key] = [eval_metrics]  # Wrap single run in list
                elif isinstance(eval_metrics, list):
                    results[key] = eval_metrics
                else:
                    print(f"Warning: Invalid eval_metrics format for {key}, skipping.")
                    continue
    return results


def paired_t_tests(metrics, models, qualities, modes):
    """Conducts paired t-tests across JPEG qualities, models, and modes if multiple
    runs are available.
    """
    t_test_results = []
    has_multiple_runs = any(len(runs) > 1 for runs in metrics.values())

    if not has_multiple_runs:
        print(
            "Warning: Only single-run metrics available. Skipping t-tests. "
            "Multiple runs required for statistical significance testing."
        )
        return t_test_results

    # Within-model, across JPEG quality
    for model, mode in itertools.product(models, modes):
        for q1, q2 in itertools.combinations(qualities, 2):
            key1 = (model, q1, mode)
            key2 = (model, q2, mode)
            if key1 not in metrics or key2 not in metrics:
                continue
            if len(metrics[key1]) != len(metrics[key2]):
                print(f"Warning: Mismatched run counts for {key1} vs {key2}, skipping.")
                continue
            for metric in ["accuracy", "f1", "auc"]:
                data1 = [run[metric] for run in metrics[key1]]
                data2 = [run[metric] for run in metrics[key2]]
                try:
                    stat, p = ttest_rel(data1, data2)
                    t_test_results.append({
                        "comparison": f"{model}_{mode}_{metric}_jpeg{q1}_vs_jpeg{q2}",
                        "statistic": stat,
                        "p_value": p,
                    })
                except ValueError as e:
                    print(f"Error in t-test for {model}_{mode}_{metric}_jpeg{q1}_vs_jpeg{q2}: {e}")

    # Across models, same JPEG quality and mode
    for m1, m2 in itertools.combinations(models, 2):
        for quality, mode in itertools.product(qualities, modes):
            key1 = (m1, quality, mode)
            key2 = (m2, quality, mode)
            if key1 not in metrics or key2 not in metrics:
                continue
            if len(metrics[key1]) != len(metrics[key2]):
                print(f"Warning: Mismatched run counts for {key1} vs {key2}, skipping.")
                continue
            for metric in ["accuracy", "f1", "auc"]:
                data1 = [run[metric] for run in metrics[key1]]
                data2 = [run[metric] for run in metrics[key2]]
                try:
                    stat, p = ttest_rel(data1, data2)
                    t_test_results.append({
                        "comparison": f"{m1}_vs_{m2}_{mode}_{metric}_jpeg{quality}",
                        "statistic": stat,
                        "p_value": p,
                    })
                except ValueError as e:
                    print(f"Error in t-test for {m1}_vs_{m2}_{mode}_{metric}_jpeg{quality}: {e}")

    # Fine-tune vs. linear probe, same model and JPEG quality
    for model, quality in itertools.product(models, qualities):
        key1 = (model, quality, "finetune")
        key2 = (model, quality, "linear_probe")
        if key1 not in metrics or key2 not in metrics:
            continue
        if len(metrics[key1]) != len(metrics[key2]):
            print(f"Warning: Mismatched run counts for {key1} vs {key2}, skipping.")
            continue
        for metric in ["accuracy", "f1", "auc"]:
            data1 = [run[metric] for run in metrics[key1]]
            data2 = [run[metric] for run in metrics[key2]]
            try:
                stat, p = ttest_rel(data1, data2)
                t_test_results.append({
                    "comparison": f"{model}_finetune_vs_linear_probe_{metric}_jpeg{quality}",
                    "statistic": stat,
                    "p_value": p,
                })
            except ValueError as e:
                print(
                    f"Error in t-test for {model}_finetune_vs_linear_probe_{metric}_jpeg{quality}: {e}"
                )

    return t_test_results


def summarize_performance(metrics):
    """Summarizes performance metrics (mean, std) across models, qualities, and modes."""
    summary = []
    for (model, quality, mode), runs in metrics.items():
        if not runs:
            continue
        # Compute mean and std for each metric
        accuracies = [run["accuracy"] for run in runs]
        f1s = [run["f1"] for run in runs]
        aucs = [run["auc"] for run in runs]
        summary.append({
            "model": model,
            "jpeg_quality": quality,
            "mode": mode,
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies, ddof=1) if len(runs) > 1 else 0,
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s, ddof=1) if len(runs) > 1 else 0,
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs, ddof=1) if len(runs) > 1 else 0,
        })
    return pd.DataFrame(summary)


def plot_performance(df):
    """Generates plots for performance metrics across JPEG qualities and models."""
    metrics = ["accuracy", "f1", "auc"]
    modes = ["finetune", "linear_probe"]

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for mode in modes:
            subset = df[df["mode"] == mode]
            sns.lineplot(
                x="jpeg_quality",
                y=f"{metric}_mean",
                hue="model",
                style="model",
                markers=True,
                dashes=False,
                data=subset,
                label=f"{mode}",
            )
        plt.title(f"{metric.capitalize()} vs. JPEG Quality")
        plt.xlabel("JPEG Quality")
        plt.ylabel(f"{metric.capitalize()} Mean")
        plt.legend()
        plt.savefig(f"{metric}_vs_jpeg_quality.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Bar plot for model comparison
    for mode in modes:
        plt.figure(figsize=(12, 6))
        subset = df[df["mode"] == mode]
        subset_melted = subset.melt(
            id_vars=["model", "jpeg_quality", "mode"],
            value_vars=["accuracy_mean", "f1_mean", "auc_mean"],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(
            x="jpeg_quality",
            y="value",
            hue="model",
            style="metric",
            data=subset_melted,
        )
        plt.title(f"Performance Metrics ({mode.capitalize()})")
        plt.xlabel("JPEG Quality")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.savefig(f"metrics_{mode}_bar.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main function to evaluate results and conduct statistical tests."""
    # Load results
    metrics = load_results()

    if not metrics:
        print("No results found. Please ensure JSON files exist.")
        return

    # Conduct paired t-tests (if multiple runs)
    models = ["vit", "dinov2", "simclr"]
    qualities = [90, 50, 20]
    modes = ["finetune", "linear_probe"]
    t_test_results = paired_t_tests(metrics, models, qualities, modes)

    # Summarize performance
    performance_df = summarize_performance(metrics)

    # Generate plots
    plot_performance(performance_df)

    # Save results
    results = {
        "t_test_results": t_test_results,
        "performance_summary": performance_df.to_dict(orient="records"),
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation complete. Results saved to 'evaluation_results.json'.")
    print(
        "Plots saved: accuracy_vs_jpeg_quality.png, f1_vs_jpeg_quality.png, "
        "auc_vs_jpeg_quality.png, metrics_finetune_bar.png, metrics_linear_probe_bar.png"
    )


if __name__ == "__main__":
    main()
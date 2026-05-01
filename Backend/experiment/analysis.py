import json
import math
from pathlib import Path
from collections import defaultdict
from .metrics import compute_all_metrics


def analyze_experiment(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["experiment_type"] == "diversity":
        return _analyze_diversity(data)
    elif data["experiment_type"] == "group_size":
        return _analyze_group_size(data)
    else:
        raise ValueError(f"Unknown experiment type: {data['experiment_type']}")


def _analyze_diversity(data):
    results_by_level = defaultdict(list)
    for session in data["results"]:
        results_by_level[session["diversity_level"]].append(compute_all_metrics(session))

    comparisons = _build_comparisons(results_by_level)
    analysis = {
        "experiment_type": "diversity",
        "experiment_id": data["experiment_id"],
        "comparisons": comparisons,
    }

    out_path = Path("results") / f"analysis_diversity_{data['experiment_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    _print_report("DIVERSITY EXPERIMENT", comparisons, list(results_by_level.keys()))
    return analysis


def _analyze_group_size(data):
    results_by_size = defaultdict(list)
    for session in data["results"]:
        results_by_size[str(session["group_size"])].append(compute_all_metrics(session))

    comparisons = _build_comparisons(results_by_size)
    analysis = {
        "experiment_type": "group_size",
        "experiment_id": data["experiment_id"],
        "comparisons": comparisons,
    }

    out_path = Path("results") / f"analysis_group_size_{data['experiment_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    _print_report("GROUP SIZE EXPERIMENT", comparisons,
                  sorted(results_by_size.keys(), key=lambda x: int(x) if x.isdigit() else x))
    return analysis


def _build_comparisons(results_by_condition):
    metric_keys = [
        "unique_claims", "type_token_ratio", "repetition_rate",
        "cross_reference_rate", "question_rate", "disagreement_rate",
        "speaker_balance", "claim_density", "self_repetition_rate",
    ]
    comparisons = {}
    for key in metric_keys:
        comparisons[key] = {}
        for cond, metrics_list in results_by_condition.items():
            values = [m[key] for m in metrics_list if key in m]
            comparisons[key][cond] = {
                "mean": _mean(values), "std": _std(values), "n": len(values),
            }
    return comparisons


def _mean(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return round(math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1)), 4)


def _print_report(title, comparisons, conditions):
    print(f"\n{'='*70}")
    print(f"  {title} ANALYSIS")
    print(f"{'='*70}")

    header = f"{'Metric':<25}"
    for c in conditions:
        header += f" {c:>14}"
    print(f"\n{header}")
    print("-" * (25 + 15 * len(conditions)))

    for metric, data in comparisons.items():
        row = f"{metric:<25}"
        for c in conditions:
            if c in data:
                row += f" {data[c]['mean']:>7.3f}+{data[c]['std']:<5.3f}"
            else:
                row += f" {'N/A':>14}"
        print(row)
    print(f"\n{'='*70}\n")

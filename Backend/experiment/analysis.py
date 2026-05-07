import json
import math
from pathlib import Path
from collections import defaultdict
from .metrics import compute_all_metrics


def analyze_experiment(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp_type = data["experiment_type"]
    if exp_type == "conviviality":
        return _analyze_by_key(data, "conviviality_level", "CONVIVIALITY EXPERIMENT")
    elif exp_type == "group_size":
        return _analyze_by_key(data, "group_size", "GROUP SIZE EXPERIMENT", sort_as_int=True)
    elif exp_type == "combined":
        return _analyze_combined(data)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


def _analyze_by_key(data, key, title, sort_as_int=False):
    results_by_cond = defaultdict(list)
    for session in data["results"]:
        metrics = compute_all_metrics(session)
        if "llm_judge" in session:
            metrics.update(session["llm_judge"])
        results_by_cond[str(session[key])].append(metrics)

    comparisons = _build_comparisons(results_by_cond)

    if sort_as_int:
        conditions = sorted(results_by_cond.keys(), key=lambda x: int(x) if x.isdigit() else x)
    else:
        conditions = list(results_by_cond.keys())

    analysis = {
        "experiment_type": data["experiment_type"],
        "experiment_id": data["experiment_id"],
        "comparisons": comparisons,
    }

    out_path = Path("results") / f"analysis_{data['experiment_type']}_{data['experiment_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    _print_report(title, comparisons, conditions)
    return analysis


def _analyze_combined(data):
    results_by_cell = defaultdict(list)
    for session in data["results"]:
        metrics = compute_all_metrics(session)
        if "llm_judge" in session:
            metrics.update(session["llm_judge"])
        cell_key = f"{session['conviviality_level']}_n{session['group_size']}"
        results_by_cell[cell_key].append(metrics)

    comparisons = _build_comparisons(results_by_cell)

    analysis = {
        "experiment_type": "combined",
        "experiment_id": data["experiment_id"],
        "comparisons": comparisons,
    }

    out_path = Path("results") / f"analysis_combined_{data['experiment_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    conditions = sorted(results_by_cell.keys())
    _print_report("COMBINED EXPERIMENT (Conviviality x Group Size)", comparisons, conditions)
    return analysis


def _build_comparisons(results_by_condition):
    metric_keys = [
        "unique_claims", "type_token_ratio", "repetition_rate",
        "cross_reference_rate", "question_rate", "disagreement_rate",
        "speaker_balance", "claim_density", "self_repetition_rate",
        "argument_depth", "perspective_diversity", "logical_coherence", "engagement_quality",
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

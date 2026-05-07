import json
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations
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

    sig_tests = _run_pairwise_tests(results_by_cond)

    analysis = {
        "experiment_type": data["experiment_type"],
        "experiment_id": data["experiment_id"],
        "comparisons": comparisons,
        "significance_tests": sig_tests,
    }

    out_path = Path("results") / f"analysis_{data['experiment_type']}_{data['experiment_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    _print_report(title, comparisons, conditions)
    _print_significance(sig_tests)
    return analysis


def _analyze_combined(data):
    results_by_cell = defaultdict(list)
    results_by_conv = defaultdict(list)
    results_by_size = defaultdict(list)

    for session in data["results"]:
        metrics = compute_all_metrics(session)
        if "llm_judge" in session:
            metrics.update(session["llm_judge"])
        cell_key = f"{session['conviviality_level']}_n{session['group_size']}"
        results_by_cell[cell_key].append(metrics)
        results_by_conv[session["conviviality_level"]].append(metrics)
        results_by_size[str(session["group_size"])].append(metrics)

    comparisons = _build_comparisons(results_by_cell)
    conv_sig = _run_pairwise_tests(results_by_conv)
    size_sig = _run_pairwise_tests(results_by_size)

    analysis = {
        "experiment_type": "combined",
        "experiment_id": data["experiment_id"],
        "comparisons": comparisons,
        "conviviality_significance": conv_sig,
        "group_size_significance": size_sig,
    }

    out_path = Path("results") / f"analysis_combined_{data['experiment_id']}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    conditions = sorted(results_by_cell.keys())
    _print_report("COMBINED EXPERIMENT (Conviviality x Group Size)", comparisons, conditions)
    print("\n  --- Conviviality Main Effect ---")
    _print_significance(conv_sig)
    print("  --- Group Size Main Effect ---")
    _print_significance(size_sig)
    return analysis


def _run_pairwise_tests(results_by_condition):
    metric_keys = [
        "unique_claims", "type_token_ratio", "repetition_rate",
        "cross_reference_rate", "disagreement_rate",
        "speaker_balance", "claim_density", "self_repetition_rate",
        "argument_depth", "perspective_diversity", "logical_coherence", "engagement_quality",
    ]

    cond_names = list(results_by_condition.keys())
    sig_results = {}

    for key in metric_keys:
        sig_results[key] = {}
        for c1, c2 in combinations(cond_names, 2):
            v1 = [m[key] for m in results_by_condition[c1] if key in m]
            v2 = [m[key] for m in results_by_condition[c2] if key in m]
            if len(v1) < 2 or len(v2) < 2:
                continue
            t_stat, p_value = _welch_t_test(v1, v2)
            sig_results[key][f"{c1}_vs_{c2}"] = {
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
                "mean_diff": round(_mean(v1) - _mean(v2), 4),
            }

    return sig_results


def _welch_t_test(x, y):
    n1, n2 = len(x), len(y)
    m1, m2 = sum(x) / n1, sum(y) / n2
    v1 = sum((xi - m1) ** 2 for xi in x) / (n1 - 1) if n1 > 1 else 0
    v2 = sum((yi - m2) ** 2 for yi in y) / (n2 - 1) if n2 > 1 else 0

    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 / n1 + v2 / n2) > 0 else 1e-10
    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1 / n1 + v2 / n2) ** 2
    d1 = (v1 / n1) ** 2 / (n1 - 1) if n1 > 1 and v1 > 0 else 1e-10
    d2 = (v2 / n2) ** 2 / (n2 - 1) if n2 > 1 and v2 > 0 else 1e-10
    df = num / (d1 + d2) if (d1 + d2) > 0 else 1

    p_value = _t_to_p(abs(t_stat), df)
    return t_stat, p_value


def _t_to_p(t, df):
    # Approximation of two-tailed p-value from t-distribution
    # using regularized incomplete beta function approximation
    x = df / (df + t * t)
    if df <= 0:
        return 1.0

    # Simple approximation for large df
    if df > 100:
        import math
        z = t
        p = math.erfc(abs(z) / math.sqrt(2))
        return min(1.0, p)

    # Beta function approximation for smaller df
    a = df / 2.0
    b = 0.5
    p = _regularized_beta(x, a, b)
    return min(1.0, p)


def _regularized_beta(x, a, b, max_iter=200):
    # Continued fraction approximation of regularized incomplete beta
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use log-beta for numerical stability
    lbeta = _log_beta(a, b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a

    # Lentz's algorithm for continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for i in range(1, max_iter):
        m = i
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return front * f


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


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
    print()


def _print_significance(sig_tests):
    print(f"\n  {'Metric':<25} {'Comparison':<35} {'t':>8} {'p':>8} {'Sig?':>6}")
    print("  " + "-" * 85)
    for metric, pairs in sig_tests.items():
        for pair_name, result in pairs.items():
            sig_mark = " ***" if result["p_value"] < 0.001 else " **" if result["p_value"] < 0.01 else " *" if result["p_value"] < 0.05 else ""
            print(f"  {metric:<25} {pair_name:<35} {result['t_statistic']:>8.3f} {result['p_value']:>8.4f} {sig_mark:>6}")
    print()

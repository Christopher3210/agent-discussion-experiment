import json
from pathlib import Path
from collections import defaultdict
from .metrics import compute_all_metrics

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_diversity_comparison(result_path, output_dir="figures"):
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualisation")
        return

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    results_by_level = defaultdict(list)
    for session in data["results"]:
        results_by_level[session["diversity_level"]].append(compute_all_metrics(session))

    levels = ["low", "medium", "high"]
    colors = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}

    metric_configs = [
        ("unique_claims", "Unique Claims", "Count"),
        ("type_token_ratio", "Lexical Diversity (TTR)", "Ratio"),
        ("repetition_rate", "Repetition Rate", "Rate"),
        ("cross_reference_rate", "Cross-Reference Rate", "Rate"),
        ("claim_density", "Claim Density", "Claims per Turn"),
        ("disagreement_rate", "Disagreement Rate", "Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (key, title, ylabel) in enumerate(metric_configs):
        ax = axes[idx]
        means, stds = [], []
        for level in levels:
            values = [m[key] for m in results_by_level.get(level, []) if key in m]
            means.append(sum(values) / len(values) if values else 0)
            stds.append(_std(values))

        bars = ax.bar(levels, means, yerr=stds, capsize=5,
                      color=[colors[l] for l in levels], alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle("Effect of Agent Perspective Diversity on Discussion Quality",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / "diversity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'diversity_comparison.png'}")


def plot_group_size_comparison(result_path, output_dir="figures"):
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualisation")
        return

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    results_by_size = defaultdict(list)
    for session in data["results"]:
        results_by_size[session["group_size"]].append(compute_all_metrics(session))

    sizes = sorted(results_by_size.keys())

    metric_configs = [
        ("unique_claims", "Unique Claims", "Count"),
        ("type_token_ratio", "Lexical Diversity (TTR)", "Ratio"),
        ("repetition_rate", "Repetition Rate", "Rate"),
        ("self_repetition_rate", "Self-Repetition Rate", "Rate"),
        ("speaker_balance", "Speaker Balance", "Balance"),
        ("claim_density", "Claim Density", "Claims per Turn"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (key, title, ylabel) in enumerate(metric_configs):
        ax = axes[idx]
        means, stds = [], []
        for size in sizes:
            values = [m[key] for m in results_by_size[size] if key in m]
            means.append(sum(values) / len(values) if values else 0)
            stds.append(_std(values))

        ax.errorbar(sizes, means, yerr=stds, marker='o', capsize=5,
                    linewidth=2, markersize=8, color='#2196F3')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Number of Agents", fontsize=9)
        ax.set_xticks(sizes)

    plt.suptitle("Effect of Group Size on Multi-Agent Discussion Quality",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / "group_size_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'group_size_comparison.png'}")


def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5

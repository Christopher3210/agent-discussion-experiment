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


def plot_conviviality_comparison(result_path, output_dir="figures"):
    if not HAS_MATPLOTLIB:
        return

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    results_by_level = defaultdict(list)
    for session in data["results"]:
        metrics = compute_all_metrics(session)
        if "llm_judge" in session:
            metrics.update(session["llm_judge"])
        results_by_level[session["conviviality_level"]].append(metrics)

    levels = ["confrontational", "balanced", "cooperative"]
    colors = {"confrontational": "#F44336", "balanced": "#FF9800", "cooperative": "#4CAF50"}

    metric_configs = [
        ("unique_claims", "Unique Claims", "Count"),
        ("type_token_ratio", "Lexical Diversity (TTR)", "Ratio"),
        ("repetition_rate", "Repetition Rate", "Rate"),
        ("disagreement_rate", "Disagreement Rate", "Rate"),
        ("argument_depth", "Argument Depth (LLM)", "Score (1-10)"),
        ("engagement_quality", "Engagement Quality (LLM)", "Score (1-10)"),
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

    plt.suptitle("Effect of Debate Intensity on Discussion Quality",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / "conviviality_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'conviviality_comparison.png'}")


def plot_group_size_comparison(result_path, output_dir="figures"):
    if not HAS_MATPLOTLIB:
        return

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    results_by_size = defaultdict(list)
    for session in data["results"]:
        metrics = compute_all_metrics(session)
        if "llm_judge" in session:
            metrics.update(session["llm_judge"])
        results_by_size[session["group_size"]].append(metrics)

    sizes = sorted(results_by_size.keys())

    metric_configs = [
        ("unique_claims", "Unique Claims", "Count"),
        ("type_token_ratio", "Lexical Diversity (TTR)", "Ratio"),
        ("repetition_rate", "Repetition Rate", "Rate"),
        ("self_repetition_rate", "Self-Repetition Rate", "Rate"),
        ("argument_depth", "Argument Depth (LLM)", "Score (1-10)"),
        ("perspective_diversity", "Perspective Diversity (LLM)", "Score (1-10)"),
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

    plt.suptitle("Effect of Group Size on Discussion Quality",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / "group_size_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'group_size_comparison.png'}")


def plot_combined_heatmap(result_path, output_dir="figures"):
    if not HAS_MATPLOTLIB:
        return

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    results = defaultdict(lambda: defaultdict(list))
    for session in data["results"]:
        metrics = compute_all_metrics(session)
        if "llm_judge" in session:
            metrics.update(session["llm_judge"])
        conv = session["conviviality_level"]
        size = session["group_size"]
        results[conv][size].append(metrics)

    conv_order = ["confrontational", "balanced", "cooperative"]
    sizes = sorted(set(s["group_size"] for s in data["results"]))

    metric_configs = [
        ("unique_claims", "Unique Claims"),
        ("repetition_rate", "Repetition Rate"),
        ("disagreement_rate", "Disagreement Rate"),
        ("argument_depth", "Argument Depth (LLM)"),
        ("perspective_diversity", "Perspective Diversity (LLM)"),
        ("engagement_quality", "Engagement Quality (LLM)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(metric_configs):
        ax = axes[idx]
        matrix = []
        for conv in conv_order:
            row = []
            for size in sizes:
                values = [m[key] for m in results[conv][size] if key in m]
                row.append(sum(values) / len(values) if values else 0)
            matrix.append(row)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_yticks(range(len(conv_order)))
        ax.set_yticklabels(conv_order)
        ax.set_xlabel("Group Size")
        ax.set_ylabel("Conviviality")
        ax.set_title(title, fontsize=11, fontweight='bold')

        for i in range(len(conv_order)):
            for j in range(len(sizes)):
                ax.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center', fontsize=9)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Conviviality x Group Size Interaction Effects",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / "combined_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out / 'combined_heatmap.png'}")


def _std(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return (sum((x - m) ** 2 for x in values) / (len(values) - 1)) ** 0.5

import asyncio
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Discussion Experiment")
    parser.add_argument("experiment", choices=["conviviality", "group_size", "combined", "single", "analyze"])
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--conviviality", type=float, default=0.5)
    parser.add_argument("--topic", type=str, default="Should artificial intelligence be regulated by governments?")
    parser.add_argument("--agents", type=str, nargs="+")
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--api-key", type=str)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.experiment == "analyze":
        if not args.result_file:
            print("Error: --result-file required for analysis")
            sys.exit(1)
        from experiment.analysis import analyze_experiment
        from experiment.visualize import plot_conviviality_comparison, plot_group_size_comparison, plot_combined_heatmap
        import json

        analyze_experiment(args.result_file)
        with open(args.result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["experiment_type"] == "conviviality":
            plot_conviviality_comparison(args.result_file)
        elif data["experiment_type"] == "group_size":
            plot_group_size_comparison(args.result_file)
        elif data["experiment_type"] == "combined":
            plot_combined_heatmap(args.result_file)
        print("\nAnalysis complete.")
        return

    from experiment.experiment_runner import ExperimentRunner

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY env var or pass --api-key")
        sys.exit(1)

    runner = ExperimentRunner(api_key=api_key)

    if args.experiment == "single":
        agent_names = args.agents or ["Physicist", "Economist", "Ethicist"]
        result = asyncio.run(runner.run_single(
            agent_names=agent_names, topic=args.topic,
            num_turns=args.turns, conviviality=args.conviviality,
        ))
        print(f"\nSession {result['session_id']} complete. {result['num_turns']} turns.")

    elif args.experiment == "conviviality":
        asyncio.run(runner.run_conviviality_experiment(
            num_turns=args.turns, repetitions=args.reps,
        ))

    elif args.experiment == "group_size":
        asyncio.run(runner.run_group_size_experiment(
            num_turns=args.turns, repetitions=args.reps,
        ))

    elif args.experiment == "combined":
        asyncio.run(runner.run_combined_experiment(
            num_turns=args.turns, repetitions=args.reps,
        ))


if __name__ == "__main__":
    main()

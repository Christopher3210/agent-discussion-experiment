import asyncio
import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from agents.agent_base import Agent
from agents.agents_manager import AgentsManager
from llm.cloud_model_manager import CloudModelManager
from controller.dialogue_controller import DialogueController
from experiment.llm_judge import judge_session

# Disciplinary classification based on Biglan (1973) taxonomy:
#   Hard/Soft axis: empirical-quantitative vs interpretive-qualitative
#   Pure/Applied axis: theory-driven vs practice-driven
#
# Hard-Pure:    Physicist, Biologist (natural sciences, empirical laws)
# Hard-Applied: Engineer (technical problem-solving, design constraints)
# Soft-Pure:    Ethicist, Sociologist, Psychologist (interpretive, theory-oriented)
# Soft-Applied: Economist, Political Scientist (policy, institutional design)
#
# Low diversity  = agents from the SAME quadrant
# Medium diversity = agents from ADJACENT quadrants (one axis differs)
# High diversity = agents from OPPOSITE quadrants (both axes differ)

DISCIPLINE_CATEGORY = {
    "Physicist":           "hard-pure",
    "Biologist":           "hard-pure",
    "Engineer":            "hard-applied",
    "Ethicist":            "soft-pure",
    "Sociologist":         "soft-pure",
    "Psychologist":        "soft-pure",
    "Economist":           "soft-applied",
    "Political Scientist": "soft-applied",
}

DIVERSITY_CONFIGS = {
    "low": {
        "description": "Same quadrant (Biglan taxonomy)",
        "groups": [
            ["Physicist", "Biologist", "Engineer"],           # hard cluster
            ["Ethicist", "Sociologist", "Psychologist"],      # soft-pure cluster
            ["Economist", "Political Scientist", "Engineer"],  # applied cluster
        ],
    },
    "medium": {
        "description": "Adjacent quadrants (one axis differs)",
        "groups": [
            ["Physicist", "Economist", "Biologist"],           # hard-pure + soft-applied
            ["Engineer", "Psychologist", "Ethicist"],          # hard-applied + soft-pure
            ["Sociologist", "Economist", "Biologist"],         # soft + hard mix
        ],
    },
    "high": {
        "description": "Opposite quadrants (both axes differ)",
        "groups": [
            ["Physicist", "Ethicist", "Political Scientist"],  # hard-pure + soft-pure + soft-applied
            ["Engineer", "Sociologist", "Economist"],          # hard-applied + soft-pure + soft-applied
            ["Biologist", "Psychologist", "Political Scientist"],  # hard-pure + soft-pure + soft-applied
        ],
    },
}

TOPICS = [
    "Should artificial intelligence be regulated by governments?",
    "Is economic growth compatible with environmental sustainability?",
    "Should genetic editing of human embryos be permitted?",
    "How should society allocate limited healthcare resources?",
    "Is universal basic income a viable policy?",
]


class ExperimentRunner:

    def __init__(self, api_key=None, output_dir="results"):
        self.model_manager = CloudModelManager(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._load_all_agents()

    def _load_all_agents(self):
        self.agent_pool: Dict[str, Agent] = {}
        manager = AgentsManager(cfg_dir="agents/configs")
        for agent in manager.get_all_agents():
            self.agent_pool[agent.name] = agent
        print(f"[ExperimentRunner] Loaded {len(self.agent_pool)} agents: "
              f"{', '.join(sorted(self.agent_pool.keys()))}")

    def _get_agents(self, names):
        agents = []
        for name in names:
            if name not in self.agent_pool:
                raise ValueError(f"Agent '{name}' not found in pool")
            original = self.agent_pool[name]
            agent = Agent(
                name=original.name,
                system_prompt=original.system_prompt,
                model_key=original.model_key,
            )
            agents.append(agent)
        return agents

    async def run_single(self, agent_names, topic, num_turns=20, conviviality=0.5):
        agents = self._get_agents(agent_names)
        controller = DialogueController(
            model_manager=self.model_manager, agents=agents,
            history_window=8, conviviality=conviviality,
        )
        result = await controller.run_dialogue(topic=topic, num_turns=num_turns)
        result["llm_judge"] = judge_session(result, self.model_manager)
        print(f"    LLM Judge: {result['llm_judge']}")
        return result

    async def run_diversity_experiment(self, num_turns=20, repetitions=3, topics=None, conviviality=0.5):
        topics = topics or TOPICS
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []

        print(f"\n{'='*60}")
        print(f"  DIVERSITY EXPERIMENT - {experiment_id}")
        print(f"  Turns: {num_turns} | Repetitions: {repetitions}")
        print(f"{'='*60}\n")

        for diversity_level, config in DIVERSITY_CONFIGS.items():
            print(f"\n--- Diversity: {diversity_level.upper()} ({config['description']}) ---")
            for group_idx, agent_names in enumerate(config["groups"]):
                for rep in range(repetitions):
                    topic = topics[(group_idx * repetitions + rep) % len(topics)]
                    print(f"\n[{diversity_level}] Group {group_idx+1}, Rep {rep+1}/{repetitions}")
                    result = await self.run_single(
                        agent_names=agent_names, topic=topic,
                        num_turns=num_turns, conviviality=conviviality,
                    )
                    result["diversity_level"] = diversity_level
                    result["group_index"] = group_idx
                    result["repetition"] = rep
                    all_results.append(result)

        experiment_data = {
            "experiment_id": experiment_id,
            "experiment_type": "diversity",
            "config": {
                "num_turns": num_turns, "repetitions": repetitions,
                "conviviality": conviviality, "topics": topics,
            },
            "results": all_results,
        }

        output_path = self.output_dir / f"diversity_{experiment_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Experiment] Results saved to {output_path}")
        return experiment_data

    async def run_group_size_experiment(self, sizes=None, num_turns=20, repetitions=3, topics=None, conviviality=0.5):
        sizes = sizes or [2, 3, 4, 6]
        topics = topics or TOPICS
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_results = []
        all_agent_names = sorted(self.agent_pool.keys())

        print(f"\n{'='*60}")
        print(f"  GROUP SIZE EXPERIMENT - {experiment_id}")
        print(f"  Sizes: {sizes} | Turns: {num_turns} | Repetitions: {repetitions}")
        print(f"{'='*60}\n")

        for size in sizes:
            if size > len(all_agent_names):
                print(f"[Warning] Size {size} exceeds available agents ({len(all_agent_names)}), skipping")
                continue
            print(f"\n--- Group Size: {size} ---")
            for rep in range(repetitions):
                selected = random.sample(all_agent_names, size)
                topic = topics[rep % len(topics)]
                print(f"\n[size={size}] Rep {rep+1}/{repetitions}: {', '.join(selected)}")
                result = await self.run_single(
                    agent_names=selected, topic=topic,
                    num_turns=num_turns, conviviality=conviviality,
                )
                result["group_size"] = size
                result["repetition"] = rep
                all_results.append(result)

        experiment_data = {
            "experiment_id": experiment_id,
            "experiment_type": "group_size",
            "config": {
                "sizes": sizes, "num_turns": num_turns,
                "repetitions": repetitions, "conviviality": conviviality,
                "topics": topics,
            },
            "results": all_results,
        }

        output_path = self.output_dir / f"group_size_{experiment_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Experiment] Results saved to {output_path}")
        return experiment_data

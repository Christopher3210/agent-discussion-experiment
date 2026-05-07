import json


JUDGE_PROMPT = """You are a strict evaluator of multi-agent discussions. You must critically assess the quality and assign scores using the FULL 1-5 scale.

Score definitions:
  1 = Very poor: Agents repeat themselves, ignore each other, or make no substantive points
  2 = Poor: Mostly surface-level, limited interaction, few distinct arguments
  3 = Average: Some valid points made, moderate engagement, but nothing stands out
  4 = Good: Clear distinct arguments, agents respond to each other meaningfully
  5 = Excellent: Deep reasoning, genuine back-and-forth, multiple unique perspectives explored

Rate this discussion on 4 dimensions:

1. **Argument Depth**: Do agents provide specific evidence, examples, or reasoning? Or just vague claims?
2. **Perspective Diversity**: Do agents bring genuinely DIFFERENT viewpoints? Count how many distinct angles appear.
3. **Logical Coherence**: Do responses logically follow from what was said before? Or do agents ignore each other?
4. **Engagement Quality**: Do agents directly reference and respond to specific points others made? Or talk past each other?

IMPORTANT: Most discussions should score 2-4. Reserve 5 for truly exceptional quality. Use 1 for clearly bad discussions. Do NOT default to high scores.

Discussion topic: {topic}
Participants: {participants}

Transcript:
{transcript}

First, briefly note one strength and one weakness of this discussion (1 sentence each).
Then output your scores as JSON:
{{"argument_depth": <int 1-5>, "perspective_diversity": <int 1-5>, "logical_coherence": <int 1-5>, "engagement_quality": <int 1-5>}}"""


def judge_session(session_data, model_manager):
    topic = session_data["topic"]
    participants = ", ".join(session_data["participants"])
    history = session_data["history"]

    transcript = "\n".join(f"{h['agent']}: {h['response']}" for h in history)

    prompt = JUDGE_PROMPT.format(
        topic=topic, participants=participants, transcript=transcript
    )

    response = model_manager.chat_once(
        model_key="gpt35",
        system_prompt="You are a strict discussion quality evaluator. Use the full 1-5 scale. Most discussions are average (3).",
        user_prompt=prompt,
        max_new_tokens=200,
        temperature=0.2,
    )

    try:
        response = response.strip()
        json_start = response.rfind("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            scores = json.loads(json_str)
            required = ["argument_depth", "perspective_diversity", "logical_coherence", "engagement_quality"]
            for key in required:
                if key not in scores or not isinstance(scores[key], (int, float)):
                    return _default_scores()
                scores[key] = max(1, min(5, int(scores[key])))
            return scores
        else:
            print(f"[LLM Judge] No JSON found in response: {response[:100]}")
            return _default_scores()
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"[LLM Judge] Failed to parse response: {response[:100]}")
        return _default_scores()


def _default_scores():
    return {
        "argument_depth": 3,
        "perspective_diversity": 3,
        "logical_coherence": 3,
        "engagement_quality": 3,
    }

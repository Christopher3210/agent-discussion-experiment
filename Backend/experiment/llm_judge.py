import json
from typing import Dict, Any


JUDGE_PROMPT = """You are an expert evaluator of multi-agent discussions. Rate the following discussion on these 4 dimensions, each from 1 (very poor) to 10 (excellent):

1. **Argument Depth**: How substantive and well-reasoned are the arguments? Do agents go beyond surface-level claims?
2. **Perspective Diversity**: How varied are the viewpoints? Do agents bring genuinely different angles or just repeat similar ideas?
3. **Logical Coherence**: Do arguments follow logically? Do agents build on or properly counter previous points?
4. **Engagement Quality**: How well do agents interact with each other? Do they reference, challenge, and respond to specific points made by others?

Discussion topic: {topic}
Participants: {participants}

Transcript:
{transcript}

Respond with ONLY a JSON object in this exact format, no other text:
{{"argument_depth": <int>, "perspective_diversity": <int>, "logical_coherence": <int>, "engagement_quality": <int>}}"""


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
        system_prompt="You are a discussion quality evaluator. Respond only with valid JSON.",
        user_prompt=prompt,
        max_new_tokens=100,
        temperature=0.1,
    )

    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        scores = json.loads(response)
        required = ["argument_depth", "perspective_diversity", "logical_coherence", "engagement_quality"]
        for key in required:
            if key not in scores or not isinstance(scores[key], (int, float)):
                return _default_scores()
            scores[key] = max(1, min(10, int(scores[key])))
        return scores
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"[LLM Judge] Failed to parse response: {response[:100]}")
        return _default_scores()


def _default_scores():
    return {
        "argument_depth": 5,
        "perspective_diversity": 5,
        "logical_coherence": 5,
        "engagement_quality": 5,
    }

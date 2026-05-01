import re
import math
from collections import Counter
from typing import List, Dict, Any


def compute_all_metrics(session_data):
    utterances = session_data["utterances"]
    participants = session_data["participants"]
    texts = [u["content"] for u in utterances]
    speakers = [u["speaker"] for u in utterances]

    metrics = {}
    metrics["num_turns"] = len(utterances)
    metrics["num_agents"] = len(participants)
    metrics["avg_word_count"] = _avg_word_count(texts)
    metrics["total_words"] = sum(len(t.split()) for t in texts)
    metrics["speaker_entropy"] = _speaker_entropy(speakers, participants)
    metrics["speaker_balance"] = _speaker_balance(speakers, participants)
    metrics["type_token_ratio"] = _type_token_ratio(texts)
    metrics["unique_claims"] = _count_unique_claims(texts)
    metrics["claim_density"] = metrics["unique_claims"] / max(1, len(texts))
    metrics["repetition_rate"] = _repetition_rate(texts)
    metrics["self_repetition_rate"] = _self_repetition_rate(utterances)
    metrics["cross_reference_rate"] = _cross_reference_rate(utterances, participants)
    metrics["question_rate"] = _question_rate(texts)
    metrics["disagreement_rate"] = _disagreement_rate(utterances)
    return metrics


def _avg_word_count(texts):
    if not texts:
        return 0.0
    return sum(len(t.split()) for t in texts) / len(texts)


def _speaker_entropy(speakers, participants):
    if not speakers:
        return 0.0
    counts = Counter(speakers)
    total = len(speakers)
    entropy = 0.0
    for name in participants:
        p = counts.get(name, 0) / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _speaker_balance(speakers, participants):
    if not speakers or len(participants) <= 1:
        return 1.0
    max_entropy = math.log2(len(participants))
    if max_entropy == 0:
        return 1.0
    return round(_speaker_entropy(speakers, participants) / max_entropy, 4)


def _type_token_ratio(texts):
    all_words = []
    for text in texts:
        all_words.extend(re.findall(r'\b\w+\b', text.lower()))
    if not all_words:
        return 0.0
    return round(len(set(all_words)) / len(all_words), 4)


def _count_unique_claims(texts):
    all_sentences = []
    for text in texts:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        all_sentences.extend(s.strip() for s in sents if len(s.strip()) > 10)
    if not all_sentences:
        return 0

    unique = []
    for sent in all_sentences:
        words_a = set(re.findall(r'\b\w+\b', sent.lower()))
        is_duplicate = False
        for existing in unique:
            words_b = set(re.findall(r'\b\w+\b', existing.lower()))
            if not words_a or not words_b:
                continue
            if len(words_a & words_b) / len(words_a | words_b) > 0.6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(sent)
    return len(unique)


def _repetition_rate(texts):
    all_sentences = []
    for text in texts:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        all_sentences.extend(s.strip() for s in sents if len(s.strip()) > 10)
    if len(all_sentences) <= 1:
        return 0.0

    repeated = 0
    seen = []
    for sent in all_sentences:
        words_a = set(re.findall(r'\b\w+\b', sent.lower()))
        for prev in seen:
            words_b = set(re.findall(r'\b\w+\b', prev.lower()))
            if words_a and words_b and len(words_a & words_b) / len(words_a | words_b) > 0.6:
                repeated += 1
                break
        seen.append(sent)
    return round(repeated / len(all_sentences), 4)


def _self_repetition_rate(utterances):
    if len(utterances) <= 1:
        return 0.0
    agent_history = {}
    self_repeats = 0
    for utt in utterances:
        speaker = utt["speaker"]
        words_a = set(re.findall(r'\b\w+\b', utt["content"].lower()))
        if speaker in agent_history:
            for prev in agent_history[speaker]:
                words_b = set(re.findall(r'\b\w+\b', prev.lower()))
                if words_a and words_b and len(words_a & words_b) / len(words_a | words_b) > 0.5:
                    self_repeats += 1
                    break
        agent_history.setdefault(speaker, []).append(utt["content"])
    return round(self_repeats / max(1, len(utterances)), 4)


def _cross_reference_rate(utterances, participants):
    if not utterances:
        return 0.0
    references = 0
    for utt in utterances:
        text_lower = utt["content"].lower()
        for name in participants:
            if name != utt["speaker"] and name.lower() in text_lower:
                references += 1
                break
    return round(references / len(utterances), 4)


def _question_rate(texts):
    if not texts:
        return 0.0
    return round(sum(1 for t in texts if '?' in t) / len(texts), 4)


def _disagreement_rate(utterances):
    if not utterances:
        return 0.0
    disagree = sum(1 for u in utterances if u.get("stance", "") in ("DISAGREE", "STRONGLY_DISAGREE"))
    return round(disagree / len(utterances), 4)

import asyncio
import re
import random
from typing import List, Dict, Any

from llm.cloud_model_manager import CloudModelManager
from .speaker_selector import SpeakerSelector
from .motivation_scorer import MotivationScorer
from .stance_analyzer import StanceAnalyzer
from .debate_logger import DebateLogger


class DialogueController:

    def __init__(self, model_manager, agents, history_window=8, conviviality=0.5):
        self.model_manager = model_manager
        self.agents = agents
        self.history_window = history_window
        self.conviviality = conviviality
        self.history: List[Dict[str, Any]] = []
        self.speech_count = 0
        self.last_speaker = None
        self.speaker_selector = SpeakerSelector(self.agents, self.history)
        self.motivation_scorer = MotivationScorer(self.model_manager)
        self.stance_analyzer = StanceAnalyzer(self.model_manager)
        self.logger = None

    def _build_context(self):
        recent = self.history[-self.history_window:]
        if not recent:
            return ""
        return "\n".join(f"{item['agent']}: {item['response']}" for item in recent)

    async def run_dialogue(self, topic, num_turns=20):
        participant_names = [a.name for a in self.agents]
        self.logger = DebateLogger(topic, participant_names, conviviality=self.conviviality)

        print(f"  Topic: {topic}")
        print(f"  Agents: {', '.join(participant_names)}")
        print(f"  Turns: {num_turns} | Conviviality: {self.conviviality}")

        for turn in range(num_turns):
            speaker = self.speaker_selector.select_next_speaker()

            stance = self.stance_analyzer.analyze_stance(
                agent=speaker,
                recent_history=self.history[-3:],
                conviviality=self.conviviality,
            )
            tone_instruction = self.stance_analyzer.get_tone_instruction(
                stance=stance, conviviality=self.conviviality
            )

            context = self._build_context()
            context_block = f"Recent discussion:\n{context}\n\n" if context else ""

            engagement_instruction = ""
            if self.last_speaker:
                engagement_instruction = (
                    f"- Directly engage with a specific point {self.last_speaker} just made — "
                    f"challenge it, qualify it, or build on it with your own expertise.\n"
                )

            question_instruction = ""
            if random.random() < 0.35:
                other_names = [a.name for a in self.agents if a.name != speaker.name]
                question_instruction = (
                    f"- End your response with a direct question to another participant "
                    f"({', '.join(other_names)}) to push the discussion deeper.\n"
                )

            length_roll = random.random()
            if length_roll < 0.35:
                length_instruction = "Reply with a single short reaction — under 20 words."
                target_sentences = 1
            elif length_roll < 0.75:
                length_instruction = "Reply in two sentences."
                target_sentences = 2
            else:
                length_instruction = "Reply in three sentences."
                target_sentences = 3

            max_tokens = 40 if target_sentences == 1 else 150

            user_prompt = (
                f"Discussion topic: {topic}\n\n"
                f"{context_block}"
                f"Respond directly in first person from your professional perspective.\n"
                f"- {length_instruction}\n"
                f"- {tone_instruction}\n"
                f"{engagement_instruction}"
                f"{question_instruction}"
                f"- Your response must be shaped by what was just said.\n"
                f"- Do NOT repeat points already made. Introduce a NEW angle, concrete example, or specific counterargument that has not appeared yet.\n"
                f"- Avoid generic phrases like 'regulation is crucial/essential/important'. Be specific about WHAT should be regulated and HOW.\n"
                f"- Do NOT refer to yourself in third person.\n"
                f"- Do NOT start with 'As a [role]' or 'As an [role]' — jump straight into your argument.\n"
                f"- Do NOT start with labels like 'Response:' or any prefix.\n"
            )

            loop = asyncio.get_event_loop()
            reply = await loop.run_in_executor(
                None,
                self.model_manager.chat_once,
                speaker.model_key,
                speaker.system_prompt,
                user_prompt,
                max_tokens,
                0.7,
            )

            reply = reply.replace("\n", " ").strip()
            reply = self._clean_reply(reply, speaker.name)
            reply = re.sub(
                r'^(Reaction|Rejection|Response|Reply|Answer|Observation|Rebuttal)\s*:\s*',
                '', reply, flags=re.IGNORECASE
            )

            sentences = re.split(r'(?<=[.!?])\s+', reply.strip())
            sentences = [s for s in sentences if s.strip()]
            if len(sentences) > target_sentences:
                reply = " ".join(sentences[:target_sentences])
                if reply[-1] not in '.!?':
                    reply += '.'

            max_chars = 120 if target_sentences == 1 else 300
            if len(reply) > max_chars:
                truncated = reply[:max_chars]
                last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                if last_period > 30:
                    reply = truncated[:last_period + 1]

            self.last_speaker = speaker.name
            speaker.add_memory(user_prompt, reply)
            self.history.append({"agent": speaker.name, "response": reply})
            self.speech_count += 1

            motivation_snapshot = {a.name: a.motivation_score for a in self.agents}
            self.logger.log_utterance(
                speaker=speaker.name, content=reply, turn=self.speech_count,
                is_qa=False, stance=stance, motivation_scores=motivation_snapshot,
            )

            self.motivation_scorer.analyze_utterance(
                speaker_name=speaker.name, text=reply,
                all_agents=self.agents, recent_history=self.history[-5:],
                conviviality=self.conviviality,
            )

            print(f"    [{turn+1}/{num_turns}] {speaker.name}: {reply[:80]}...")

        self.logger.finalize()

        return {
            "session_id": self.logger.session_id,
            "topic": topic,
            "participants": [a.name for a in self.agents],
            "num_agents": len(self.agents),
            "conviviality": self.conviviality,
            "num_turns": num_turns,
            "utterances": self.logger.utterances,
            "statistics": self.logger.stats,
            "history": self.history,
        }

    @staticmethod
    def _clean_reply(reply, speaker_name):
        if reply.startswith(speaker_name + ":"):
            reply = reply[len(speaker_name) + 1:].strip()
        elif reply.startswith(speaker_name):
            reply = reply[len(speaker_name):].strip()
            if reply.startswith(":"):
                reply = reply[1:].strip()
        reply = re.sub(r'^As an? \w+,?\s*', '', reply)
        return reply

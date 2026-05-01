# controller/stance_analyzer.py

from typing import List, Dict, Literal
from agents.agent_base import Agent


StanceType = Literal["STRONGLY_AGREE", "AGREE", "NEUTRAL", "DISAGREE", "STRONGLY_DISAGREE"]


class StanceAnalyzer:
    """
    Analyzes whether a philosopher agrees or disagrees with recent statements.

    Uses conviviality parameter to control debate intensity:
    - High conviviality (0.7-1.0): Friendly, agreement-focused
    - Medium conviviality (0.3-0.7): Balanced debate
    - Low conviviality (0.0-0.3): Heated, confrontational
    """

    def __init__(self, model_manager):
        """
        Parameters
        ----------
        model_manager : ModelManager
            Used for LLM-based stance detection
        """
        self.model_manager = model_manager

    def analyze_stance(
        self,
        agent: Agent,
        recent_history: List[Dict],
        conviviality: float
    ) -> StanceType:
        """
        Determine agent's stance toward the most recent statement.

        Parameters
        ----------
        agent : Agent
            The agent whose stance we're analyzing
        recent_history : list of dict
            Recent dialogue history
        conviviality : float
            Friendliness level (0.0 = confrontational, 1.0 = friendly)

        Returns
        -------
        StanceType
            One of: STRONGLY_AGREE, AGREE, NEUTRAL, DISAGREE, STRONGLY_DISAGREE
        """
        if not recent_history:
            return "NEUTRAL"

        # Get the last statement
        last_statement = recent_history[-1]['response']
        last_speaker = recent_history[-1]['agent']

        # Don't analyze stance toward own previous statement
        if last_speaker == agent.name:
            if len(recent_history) < 2:
                return "NEUTRAL"
            last_statement = recent_history[-2]['response']
            last_speaker = recent_history[-2]['agent']

        # Use simple heuristic-based detection for now
        # TODO: Add LLM-based stance analysis for more accuracy
        base_stance = self._heuristic_stance_detection(
            agent, last_statement, last_speaker
        )

        # Adjust based on conviviality
        adjusted_stance = self._adjust_for_conviviality(base_stance, conviviality)

        return adjusted_stance

    def _heuristic_stance_detection(
        self,
        agent: Agent,
        statement: str,
        speaker: str
    ) -> StanceType:
        """
        Simple keyword-based stance detection.

        Returns base stance before conviviality adjustment.
        """
        statement_lower = statement.lower()

        # Check for agreement keywords in the statement itself
        # (This means others are already agreeing, so this agent might disagree)
        agreement_keywords = ["agree", "exactly", "precisely", "indeed", "correct"]
        has_agreement = any(kw in statement_lower for kw in agreement_keywords)

        # Check for disagreement keywords
        disagreement_keywords = [
            "disagree", "wrong", "incorrect", "however", "but",
            "on the contrary", "actually", "not quite"
        ]
        has_disagreement = any(kw in statement_lower for kw in disagreement_keywords)

        # Simple heuristic: alternate between agreement and disagreement
        if has_disagreement:
            return "AGREE"  # If last person disagreed, this person might agree
        elif has_agreement:
            return "DISAGREE"  # If last person agreed, create some debate
        else:
            return "NEUTRAL"  # Default to neutral for natural flow

    def _adjust_for_conviviality(
        self,
        base_stance: StanceType,
        conviviality: float
    ) -> StanceType:
        """
        Adjust stance based on conviviality parameter.

        High conviviality → soften disagreements, promote agreement
        Low conviviality → amplify disagreements, reduce agreement
        """
        if conviviality >= 0.7:
            # High conviviality: friendly discussion
            if base_stance == "STRONGLY_DISAGREE":
                return "DISAGREE"
            elif base_stance == "DISAGREE":
                return "NEUTRAL"
            elif base_stance == "NEUTRAL":
                return "AGREE"
            # AGREE and STRONGLY_AGREE remain unchanged

        elif conviviality <= 0.3:
            # Low conviviality: heated debate
            if base_stance == "STRONGLY_AGREE":
                return "AGREE"
            elif base_stance == "AGREE":
                return "NEUTRAL"
            elif base_stance == "NEUTRAL":
                return "DISAGREE"
            # DISAGREE and STRONGLY_DISAGREE remain unchanged

        # Medium conviviality: return base stance unchanged
        return base_stance

    def get_tone_instruction(
        self,
        stance: StanceType,
        conviviality: float
    ) -> str:
        """
        Generate tone/style instruction based on stance and conviviality.

        Returns
        -------
        str
            Instruction to add to the generation prompt
        """
        if stance == "STRONGLY_DISAGREE":
            if conviviality >= 0.7:
                return "Respectfully present your different view while noting areas of agreement."
            elif conviviality <= 0.3:
                return "Forcefully reject this argument. Attack its core assumptions and show why it's fundamentally flawed."
            else:
                return "Strongly disagree with this view and explain the critical flaws in this reasoning."

        elif stance == "DISAGREE":
            if conviviality >= 0.7:
                return "Politely express your different perspective while acknowledging valid points."
            elif conviviality <= 0.3:
                return "Challenge this view aggressively. Point out its errors and defend your opposing position."
            else:
                return "Critically engage with this view and explain your disagreement."

        elif stance in ["STRONGLY_AGREE", "AGREE"]:
            if conviviality >= 0.7:
                return "Build upon this idea enthusiastically and add your insights."
            elif conviviality <= 0.3:
                return "Agree with the core point, but identify weaknesses and push back on specifics."
            else:
                return "Support this view while adding your unique perspective."

        else:  # NEUTRAL
            if conviviality >= 0.7:
                return "Offer a complementary perspective that enriches the discussion."
            elif conviviality <= 0.3:
                return "Take a contrarian stance. Challenge the underlying assumptions and provoke debate."
            else:
                return "Provide your distinct professional perspective on this topic."

# controller/debate_logger.py

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import uuid


class DebateLogger:
    """
    Comprehensive logging system for philosophical debates.

    Records:
    - Metadata (session ID, topic, participants, timestamps)
    - Every utterance (speaker, content, timing, type)
    - Statistics (speech counts, lengths, duration)
    - Future extensions (conflict scores, sentiment, etc.)

    Exports to:
    - JSON (structured data for analysis)
    - CSV (tabular data for Excel/pandas)
    - Markdown (human-readable format)
    """

    def __init__(self, topic: str, participants: List[str], output_dir: str = "logs", conviviality: float = 0.5):
        """
        Parameters
        ----------
        topic : str
            The debate topic
        participants : list of str
            Names of participating philosophers
        output_dir : str
            Directory to save log files
        conviviality : float
            Debate intensity setting (0.0 = confrontational, 1.0 = friendly)
        """
        self.session_id = str(uuid.uuid4())[:8]
        self.topic = topic
        self.participants = participants
        self.conviviality = conviviality
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Session metadata
        self.start_time = datetime.now()
        self.end_time = None

        # Utterance log - main data structure
        self.utterances: List[Dict[str, Any]] = []

        # Interrupt log - detailed interrupt events
        self.interrupts: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total_utterances": 0,
            "total_qa_utterances": 0,
            "speech_counts": {name: 0 for name in participants},
            "qa_counts": {name: 0 for name in participants},
            "total_words": 0,
            "interrupts": 0,
        }

        print(f"[Logger] Session {self.session_id} started - Topic: {topic} (conviviality: {conviviality})")

    def log_utterance(
        self,
        speaker: str,
        content: str,
        turn: int,
        is_qa: bool = False,
        stance: str = None,
        motivation_scores: Dict[str, float] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Log a single utterance.

        Parameters
        ----------
        speaker : str
            Name of the speaker
        content : str
            The utterance content
        turn : int
            Turn/round number
        is_qa : bool
            Whether this is a Q&A response
        stance : str, optional
            Speaker's stance (STRONGLY_AGREE, AGREE, NEUTRAL, DISAGREE, STRONGLY_DISAGREE)
        motivation_scores : dict, optional
            Motivation scores for all philosophers at this moment
        metadata : dict, optional
            Additional metadata (e.g., question for Q&A)
        """
        timestamp = datetime.now()

        utterance = {
            "id": len(self.utterances) + 1,
            "timestamp": timestamp.isoformat(),
            "speaker": speaker,
            "content": content,
            "turn": turn,
            "is_qa": is_qa,
            "word_count": len(content.split()),
            "char_count": len(content),
            "elapsed_seconds": (timestamp - self.start_time).total_seconds(),
        }

        # Add stance information
        if stance:
            utterance["stance"] = stance

        # Add motivation scores snapshot
        if motivation_scores:
            utterance["motivation_scores"] = motivation_scores.copy()

        # Add optional metadata
        if metadata:
            utterance.update(metadata)

        self.utterances.append(utterance)

        # Update statistics
        self.stats["total_utterances"] += 1
        if is_qa:
            self.stats["total_qa_utterances"] += 1
            self.stats["qa_counts"][speaker] += 1
        else:
            self.stats["speech_counts"][speaker] += 1

        self.stats["total_words"] += utterance["word_count"]

    def log_interrupt(self, turn: int = None, during_speaker: str = None):
        """
        Log an interrupt event with context.

        Parameters
        ----------
        turn : int, optional
            Turn number when interrupt occurred
        during_speaker : str, optional
            Name of speaker who was interrupted (if mid-speech)
        """
        timestamp = datetime.now()

        interrupt_event = {
            "id": len(self.interrupts) + 1,
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": (timestamp - self.start_time).total_seconds(),
            "turn": turn,
            "during_speaker": during_speaker,
        }

        self.interrupts.append(interrupt_event)
        self.stats["interrupts"] += 1

    def finalize(self):
        """Finalize the session and calculate final statistics."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        self.stats["duration_seconds"] = duration
        self.stats["duration_formatted"] = self._format_duration(duration)
        self.stats["avg_words_per_utterance"] = (
            self.stats["total_words"] / self.stats["total_utterances"]
            if self.stats["total_utterances"] > 0 else 0
        )

        print(f"[Logger] Session {self.session_id} finalized - Duration: {self.stats['duration_formatted']}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    def export_json(self, filename: str = None) -> str:
        """
        Export complete session data to JSON.

        Parameters
        ----------
        filename : str, optional
            Custom filename (default: auto-generated)

        Returns
        -------
        str
            Path to the exported file
        """
        if filename is None:
            filename = f"debate_{self.session_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        data = {
            "session_id": self.session_id,
            "topic": self.topic,
            "participants": self.participants,
            "conviviality": self.conviviality,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "statistics": self.stats,
            "utterances": self.utterances,
            "interrupts": self.interrupts,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[Logger] Exported JSON: {filepath}")
        return str(filepath)

    def export_csv(self, filename: str = None) -> str:
        """
        Export utterances to CSV for tabular analysis.

        Parameters
        ----------
        filename : str, optional
            Custom filename (default: auto-generated)

        Returns
        -------
        str
            Path to the exported file
        """
        if filename is None:
            filename = f"debate_{self.session_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = self.output_dir / filename

        if not self.utterances:
            print("[Logger] No utterances to export")
            return str(filepath)

        # Get all keys from utterances
        fieldnames = list(self.utterances[0].keys())

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.utterances)

        print(f"[Logger] Exported CSV: {filepath}")
        return str(filepath)

    def export_markdown(self, filename: str = None) -> str:
        """
        Export session to Markdown for human reading.

        Parameters
        ----------
        filename : str, optional
            Custom filename (default: auto-generated)

        Returns
        -------
        str
            Path to the exported file
        """
        if filename is None:
            filename = f"debate_{self.session_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.md"

        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Multi-Agent Discussion Log\n\n")
            f.write(f"**Session ID**: {self.session_id}\n\n")
            f.write(f"**Topic**: {self.topic}\n\n")
            f.write(f"**Participants**: {', '.join(self.participants)}\n\n")

            # Debate configuration
            conviviality_desc = "Friendly" if self.conviviality >= 0.7 else "Heated" if self.conviviality <= 0.3 else "Balanced"
            f.write(f"**Debate Intensity**: {conviviality_desc} (conviviality: {self.conviviality})\n\n")

            f.write(f"**Start Time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            if self.end_time:
                f.write(f"**End Time**: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Duration**: {self.stats['duration_formatted']}\n\n")

            # Statistics
            f.write(f"## Statistics\n\n")
            f.write(f"- Total Utterances: {self.stats['total_utterances']}\n")
            f.write(f"- Q&A Utterances: {self.stats['total_qa_utterances']}\n")
            f.write(f"- Total Words: {self.stats['total_words']}\n")
            f.write(f"- Avg Words/Utterance: {self.stats.get('avg_words_per_utterance', 0):.1f}\n")
            f.write(f"- User Interrupts: {self.stats['interrupts']}\n\n")

            f.write(f"### Speech Distribution\n\n")
            f.write(f"| Agent | Regular Speeches | Q&A Responses | Total |\n")
            f.write(f"|-------------|------------------|---------------|-------|\n")
            for name in self.participants:
                regular = self.stats['speech_counts'][name]
                qa = self.stats['qa_counts'][name]
                total = regular + qa
                f.write(f"| {name} | {regular} | {qa} | {total} |\n")
            f.write("\n")

            # Dialogue transcript
            f.write(f"## Dialogue Transcript\n\n")
            current_turn = -1
            for utt in self.utterances:
                # New turn header
                if utt['turn'] != current_turn:
                    current_turn = utt['turn']
                    f.write(f"### Turn {current_turn + 1}\n\n")

                # Utterance
                marker = "💬" if not utt['is_qa'] else "❓"
                elapsed = self._format_duration(utt['elapsed_seconds'])

                # Add stance indicator if available
                stance_indicator = ""
                if 'stance' in utt and utt['stance']:
                    stance_map = {
                        "STRONGLY_AGREE": "✅✅",
                        "AGREE": "✅",
                        "NEUTRAL": "〰️",
                        "DISAGREE": "❌",
                        "STRONGLY_DISAGREE": "❌❌"
                    }
                    stance_indicator = f" {stance_map.get(utt['stance'], '')}"

                f.write(f"{marker} **{utt['speaker']}**{stance_indicator} [{elapsed}]: {utt['content']}\n\n")

        print(f"[Logger] Exported Markdown: {filepath}")
        return str(filepath)

    def export_all(self):
        """Export to all formats (JSON, CSV, Markdown)."""
        self.export_json()
        self.export_csv()
        self.export_markdown()
        print(f"[Logger] All exports completed for session {self.session_id}")

    @staticmethod
    def clean_old_logs(output_dir: str = "logs", keep_recent: int = 10):
        """
        Clean old log files, keeping only the most recent sessions.

        Parameters
        ----------
        output_dir : str
            Directory containing log files
        keep_recent : int
            Number of recent sessions to keep (default: 10)

        Note
        ----
        This is optional - logs are research data and should generally be kept.
        Use this only if disk space is limited.
        """
        log_dir = Path(output_dir)
        if not log_dir.exists():
            print(f"[Logger] Log directory does not exist: {log_dir}")
            return

        # Get all log files grouped by session
        json_files = sorted(log_dir.glob("debate_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if len(json_files) <= keep_recent:
            print(f"[Logger] {len(json_files)} sessions found, no cleanup needed (keeping {keep_recent})")
            return

        # Delete old sessions
        sessions_to_delete = json_files[keep_recent:]
        deleted_count = 0

        for json_file in sessions_to_delete:
            # Extract session ID from filename
            session_id = json_file.stem.split('_')[1]

            # Delete all related files
            for ext in ['json', 'csv', 'md']:
                pattern = f"debate_{session_id}_*.{ext}"
                for file in log_dir.glob(pattern):
                    file.unlink()
                    deleted_count += 1

        print(f"[Logger] Cleaned {deleted_count} old log files, kept {keep_recent} most recent sessions")

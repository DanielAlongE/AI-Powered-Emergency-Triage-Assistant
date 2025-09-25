from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import yaml
from logger import get_logger

logger = get_logger(__name__)


def normalize(text: str) -> str:
    return text.lower().strip()


@dataclass(frozen=True)
class RedFlag:
    term: str
    severity: str
    follow_up: str
    escalation: bool


class RedFlagDetector:
    def __init__(self, lexicon_path: Path) -> None:
        self.lexicon_path = lexicon_path
        self._flags = self._load_lexicon(lexicon_path)
        logger.info("red_flag_loaded", count=len(self._flags), path=str(lexicon_path))

    @staticmethod
    def _load_lexicon(path: Path) -> List[RedFlag]:
        if not path.exists():
            logger.warning("red_flag_missing", path=str(path))
            return []
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        flags: List[RedFlag] = []
        for entry in data:
            try:
                flags.append(
                    RedFlag(
                        term=normalize(entry["term"]),
                        severity=entry.get("severity", "unknown"),
                        follow_up=entry.get("follow_up", ""),
                        escalation=bool(entry.get("escalation", False)),
                    )
                )
            except KeyError as exc:
                logger.warning("red_flag_parse_error", error=str(exc), entry=entry)
        return flags

    def scan(self, text: str) -> List[RedFlag]:
        normalized = normalize(text)
        matches = [flag for flag in self._flags if flag.term in normalized]
        if matches:
            logger.info("red_flag_detected", hits=[flag.term for flag in matches])
        return matches

    def serialize_matches(self, matches: Iterable[RedFlag]) -> List[str]:
        return [flag.term for flag in matches]

    def suggested_questions(self, matches: Iterable[RedFlag]) -> List[str]:
        return [flag.follow_up for flag in matches if flag.follow_up]
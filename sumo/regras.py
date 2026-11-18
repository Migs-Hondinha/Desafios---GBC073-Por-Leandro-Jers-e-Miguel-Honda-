"""Game rules: actions, who wins a round, and match state.

Pure Python (no torch, no pygame) so it is easy to test and reuse.
Attack beats grab, grab beats defense, defense beats attack.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

ACTIONS: tuple[str, str, str] = ("ataque", "defesa", "agarrao")
NONE = "nenhum"
MAX_LIVES = 10
HISTORY = 3

# Maps each action to the action it beats.
_BEATS = {"ataque": "agarrao", "agarrao": "defesa", "defesa": "ataque"}
_BEATEN_BY = {loser: winner for winner, loser in _BEATS.items()}


def _check(action: str) -> None:
    if action not in ACTIONS:
        raise ValueError(f"Ação inválida: {action!r}")


def counter(action: str) -> str:
    """Return the action that beats `action`."""
    _check(action)
    return _BEATEN_BY[action]


def round_result(player: str, ai: str) -> str:
    """Return "jogador", "ia" or "empate"."""
    _check(player)
    _check(ai)
    if player == ai:
        return "empate"
    return "jogador" if _BEATS[player] == ai else "ia"


@dataclass(frozen=True)
class RoundContext:
    """Everything known BEFORE a round is played (never the current actions)."""
    player_last: tuple[str, str, str]  # most recent first, padded with NONE
    ai_last: tuple[str, str, str]
    player_lives: int
    ai_lives: int


def _last(history: list[str]) -> tuple[str, str, str]:
    recent = list(reversed(history[-HISTORY:]))
    return tuple(recent + [NONE] * (HISTORY - len(recent)))  # type: ignore[return-value]


class Match:
    def __init__(self, match_id: str | None = None):
        self.match_id = match_id or uuid.uuid4().hex
        self.round_number = 0
        self.player_lives = MAX_LIVES
        self.ai_lives = MAX_LIVES
        self.player_history: list[str] = []
        self.ai_history: list[str] = []

    @property
    def finished(self) -> bool:
        return self.player_lives == 0 or self.ai_lives == 0

    @property
    def winner(self) -> str | None:
        if self.ai_lives == 0:
            return "jogador"
        if self.player_lives == 0:
            return "ia"
        return None

    def context(self) -> RoundContext:
        return RoundContext(_last(self.player_history), _last(self.ai_history),
                            self.player_lives, self.ai_lives)

    def play(self, player: str, ai: str) -> dict:
        """Apply one round and return its CSV row (features = state BEFORE the round)."""
        if self.finished:
            raise ValueError("Partida já terminou.")
        result = round_result(player, ai)
        ctx = self.context()
        self.round_number += 1
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "partida_id": self.match_id,
            "round": self.round_number,
            "jogador_t1": ctx.player_last[0], "jogador_t2": ctx.player_last[1],
            "jogador_t3": ctx.player_last[2],
            "ia_t1": ctx.ai_last[0], "ia_t2": ctx.ai_last[1], "ia_t3": ctx.ai_last[2],
            "vidas_jogador": ctx.player_lives, "vidas_ia": ctx.ai_lives,
            "acao_jogador": player, "acao_ia": ai, "resultado": result,
        }
        if result == "jogador":
            self.ai_lives -= 1
        elif result == "ia":
            self.player_lives -= 1
        self.player_history.append(player)
        self.ai_history.append(ai)
        return row

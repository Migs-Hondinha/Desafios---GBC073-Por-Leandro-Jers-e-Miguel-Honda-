"""Fake dataset to bootstrap the neuron before any real round exists.

Each synthetic player mixes five habits with random weights, so there is a
learnable pattern: preferred opening, repeating, cycling, countering the AI's
last move, and pure noise. The AI in these matches plays uniformly at random.
Run as a script to (re)write dados/falsos.csv.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

from dados_io import FAKE_CSV, read_rounds, write_rounds
from regras import ACTIONS, Match, counter

BEHAVIORS = ("opening", "repeat", "cycle", "counter_ai", "noise")
# Cycle order: ataque -> agarrao -> defesa -> ataque.
_NEXT_IN_CYCLE = {"ataque": "agarrao", "agarrao": "defesa", "defesa": "ataque"}


def _synthetic_player(rng: random.Random) -> tuple[str, list[float]]:
    preferred = rng.choices(ACTIONS, weights=[0.5, 0.25, 0.25])[0]
    weights = [rng.random() for _ in BEHAVIORS]
    weights[BEHAVIORS.index("repeat")] += 0.5  # most people repeat what worked
    return preferred, weights


def _player_move(rng: random.Random, match: Match, preferred: str, weights: list[float]) -> str:
    if not match.player_history:
        return preferred
    behavior = rng.choices(BEHAVIORS, weights=weights)[0]
    if behavior == "repeat":
        return match.player_history[-1]
    if behavior == "cycle":
        return _NEXT_IN_CYCLE[match.player_history[-1]]
    if behavior == "counter_ai":
        return counter(match.ai_history[-1])
    if behavior == "opening":
        return preferred
    return rng.choice(ACTIONS)


def generate_fake_rows(n_matches: int = 300, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for index in range(n_matches):
        preferred, weights = _synthetic_player(rng)
        match = Match(match_id=f"falso-{seed}-{index}")
        while not match.finished:
            player = _player_move(rng, match, preferred, weights)
            rows.append(match.play(player, rng.choice(ACTIONS)))
    return rows


def ensure_fake_data(path: Path = FAKE_CSV) -> list[dict]:
    """Return the fake rows, generating and saving them on first use."""
    rows = read_rounds(path)
    if rows:
        return rows
    rows = generate_fake_rows()
    write_rounds(path, rows)
    return read_rounds(path)


if __name__ == "__main__":
    rows = generate_fake_rows()
    write_rounds(FAKE_CSV, rows)
    print(f"{len(rows)} rounds falsos gravados em {FAKE_CSV}")
    print("Distribuição:", dict(Counter(r["acao_jogador"] for r in rows)))

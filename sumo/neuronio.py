"""The group's neuron (MLP from desafios 2/3) adapted to predict the player's next move.

Features (26 floats): last 3 player moves and last 3 AI moves, each one-hot over
(ataque, defesa, agarrao, nenhum), most recent first; then both lives / 10.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
import os
import random
from pathlib import Path
import sys
import time
from typing import Callable

import torch
from torch import nn

from regras import ACTIONS, MAX_LIVES, NONE, RoundContext, counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "desafio2"))
from desafio2_Leandro_Marques_Miguel_Cesario import ativacao, inicializar  # noqa: E402

SLOTS = ACTIONS + (NONE,)
INPUT_SIZE = 6 * len(SLOTS) + 2

# Winning recipe from desafio 3 (Muon + cosine); only WIDTH was reduced (256 -> 64)
# because the game produces hundreds of rounds, not thousands.
WIDTH = 64
DEPTH = 4
STEPS = 225
BATCH = 128
LR_MUON = 0.1
LR_ADAMW = 0.01
WEIGHT_DECAY = 0.01
REAL_ONLY_THRESHOLD = 200
SEED = 0
CHECKPOINT_VERSION = 1


def _one_hot(action: str) -> list[float]:
    return [1.0 if action == slot else 0.0 for slot in SLOTS]


def encode(ctx: RoundContext) -> list[float]:
    features: list[float] = []
    for action in ctx.player_last + ctx.ai_last:
        features += _one_hot(action)
    return features + [ctx.player_lives / MAX_LIVES, ctx.ai_lives / MAX_LIVES]


def row_to_example(row: dict) -> tuple[list[float], int]:
    """Turn a CSV row into (features, target class index)."""
    ctx = RoundContext(
        (row["jogador_t1"], row["jogador_t2"], row["jogador_t3"]),
        (row["ia_t1"], row["ia_t2"], row["ia_t3"]),
        int(row["vidas_jogador"]), int(row["vidas_ia"]),
    )
    return encode(ctx), ACTIONS.index(row["acao_jogador"])


# --- Model: same structure as desafio3/treino.py -------------------------------
class MLP(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, classes: int = len(ACTIONS),
                 width: int = WIDTH, depth: int = DEPTH):
        super().__init__()
        dimensions = [input_size] + [width] * (depth - 1) + [classes]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(depth)
        )
        for index, layer in enumerate(self.layers, start=1):
            inicializar(layer.weight, layer.bias, layer.in_features,
                        layer.out_features, index, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = ativacao(layer(x))
        return self.layers[-1](x)  # logits for cross_entropy


# --- Training --------------------------------------------------------------------
def build_training_set(fake_rows: list[dict], real_rows: list[dict]
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x, y, sampling weights). Few real rounds -> fake and real weigh 50/50."""
    if len(real_rows) >= REAL_ONLY_THRESHOLD:
        rows, weights = real_rows, [1.0] * len(real_rows)
    elif real_rows and fake_rows:
        rows = fake_rows + real_rows
        weights = [0.5 / len(fake_rows)] * len(fake_rows) + [0.5 / len(real_rows)] * len(real_rows)
    else:
        rows = fake_rows or real_rows
        weights = [1.0] * len(rows)
    if not rows:
        raise ValueError("Sem dados para treinar.")
    examples = [row_to_example(row) for row in rows]
    x = torch.tensor([features for features, _ in examples], dtype=torch.float32)
    y = torch.tensor([target for _, target in examples])
    return x, y, torch.tensor(weights, dtype=torch.float64)


def _cosine(step: int, total: int) -> float:
    """Same multiplier as desafio3 'cosseno': 1 on step 1, 0 on the last step."""
    progress = (step - 1) / max(1, total - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(fake_rows: list[dict], real_rows: list[dict],
                progress: Callable[[int, int], None] | None = None,
                steps: int = STEPS) -> tuple[MLP, dict]:
    """Train from scratch: desafio 2 init + Muon/AdamW with cosine schedule."""
    start = time.perf_counter()
    x, y, weights = build_training_set(fake_rows, real_rows)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = MLP()
    hidden = [layer.weight for layer in model.layers[:-1]]
    hidden_ids = {id(p) for p in hidden}
    others = [p for p in model.parameters() if id(p) not in hidden_ids]
    optimizers = [
        (torch.optim.Muon(hidden, lr=LR_MUON, weight_decay=WEIGHT_DECAY, momentum=0.95,
                          nesterov=True, ns_steps=5, adjust_lr_fn="original"), LR_MUON),
        (torch.optim.AdamW(others, lr=LR_ADAMW, weight_decay=WEIGHT_DECAY), LR_ADAMW),
    ]
    batch = min(BATCH, len(y))
    model.train()
    for step in range(1, steps + 1):
        index = torch.multinomial(weights, batch, replacement=True, generator=generator)
        model.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x[index]), y[index])
        loss.backward()
        factor = _cosine(step, steps)
        for optimizer, base_lr in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * factor
            optimizer.step()
        if progress is not None:
            progress(step, steps)
    model.eval()
    with torch.no_grad():
        accuracy = (model(x).argmax(1) == y).float().mean().item()
    used_fake = len(real_rows) < REAL_ONLY_THRESHOLD and bool(fake_rows)
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "fake_rows": len(fake_rows) if used_fake else 0,
        "real_rows": len(real_rows),
        "used_fake": used_fake,
        "train_accuracy": accuracy,
        "class_counts": {ACTIONS[k]: v for k, v in sorted(Counter(y.tolist()).items())},
        "seconds": time.perf_counter() - start,
    }
    return model, meta


# --- Checkpoint -----------------------------------------------------------------
def save_model(model: MLP, meta: dict, path: Path) -> None:
    """Write atomically: a crash mid-save never leaves a half-written model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"version": CHECKPOINT_VERSION, "input_size": INPUT_SIZE, "width": WIDTH,
                "depth": DEPTH, "state_dict": model.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


def load_model(path: Path) -> tuple[MLP, dict] | None:
    """Return (model, meta), or None if missing, corrupt or incompatible."""
    try:
        data = torch.load(path, weights_only=True)
        if (data["version"], data["input_size"], data["width"], data["depth"]) != (
                CHECKPOINT_VERSION, INPUT_SIZE, WIDTH, DEPTH):
            return None
        meta = data["meta"]
        if not isinstance(meta, dict):
            return None
        model = MLP()
        model.load_state_dict(data["state_dict"])
    except Exception:  # missing file, garbage bytes, wrong keys: just retrain
        return None
    model.eval()
    return model, meta


# --- AI decision ------------------------------------------------------------------
def predict_proba(model: MLP | None, ctx: RoundContext) -> list[float]:
    """Probability of each player action (ACTIONS order); uniform without a model."""
    if model is None:
        return [1 / 3] * 3
    with torch.no_grad():
        logits = model(torch.tensor([encode(ctx)], dtype=torch.float32))
        return torch.softmax(logits, dim=1)[0].tolist()


def choose_ai_action(model: MLP | None, ctx: RoundContext,
                     rng: random.Random) -> tuple[str, list[float]]:
    """Sample a predicted player move and play what beats it.

    Only `ctx` (rounds already played) is visible here, never the current move.
    Sampling instead of argmax keeps the AI from being trivially exploitable.
    """
    probabilities = predict_proba(model, ctx)
    predicted = rng.choices(ACTIONS, weights=probabilities)[0]
    return counter(predicted), probabilities

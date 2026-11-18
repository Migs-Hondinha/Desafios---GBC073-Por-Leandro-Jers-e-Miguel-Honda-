"""Persistence: one CSV line per round, plus a small JSON state file.

Rounds are appended and fsynced immediately so a crash never loses data.
Rows that fail validation (e.g. edited by hand in Excel) are skipped on read.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path

from regras import ACTIONS, MAX_LIVES, NONE

ROOT = Path(__file__).resolve().parent
# SUMO_DATA_DIR lets smoke tests run without touching the real data folder.
DATA_DIR = Path(os.environ["SUMO_DATA_DIR"]) if "SUMO_DATA_DIR" in os.environ else ROOT / "dados"
ROUNDS_CSV = DATA_DIR / "rounds.csv"
FAKE_CSV = DATA_DIR / "falsos.csv"
MODEL_PATH = DATA_DIR / "modelo.pt"
STATE_JSON = DATA_DIR / "estado.json"

CSV_COLUMNS = ["timestamp", "partida_id", "round", "jogador_t1", "jogador_t2",
               "jogador_t3", "ia_t1", "ia_t2", "ia_t3", "vidas_jogador", "vidas_ia",
               "acao_jogador", "acao_ia", "resultado"]
_HISTORY_COLUMNS = ["jogador_t1", "jogador_t2", "jogador_t3", "ia_t1", "ia_t2", "ia_t3"]
_DEFAULT_STATE = {"partidas_desde_treino": 0}


def _prepare_for_append(path: Path) -> None:
    """Keep the file appendable: back up foreign headers, close a truncated last line."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as file:
        header = file.readline().rstrip("\r\n")
    if header != ",".join(CSV_COLUMNS):
        # e.g. saved by Excel pt-BR with ';' — keep it aside, never delete user data
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.replace(path, path.with_name(f"{path.stem}.bak-{stamp}{path.suffix}"))
        return
    with path.open("rb") as file:
        file.seek(-1, os.SEEK_END)
        ends_with_newline = file.read(1) == b"\n"
    if not ends_with_newline:
        with path.open("ab") as file:
            file.write(b"\r\n")


def append_round(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_for_append(path)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        file.flush()
        os.fsync(file.fileno())


class RoundWriter:
    """Append rounds, queueing them in memory while the file is locked (e.g. open in Excel)."""

    def __init__(self, path: Path):
        self.path = path
        self._pending: list[dict] = []

    @property
    def pending(self) -> int:
        return len(self._pending)

    def write(self, row: dict | None = None) -> bool:
        """Queue `row` (if given) and flush the queue; False if rows are still pending."""
        if row is not None:
            self._pending.append(row)
        try:
            while self._pending:
                append_round(self.path, self._pending[0])
                self._pending.pop(0)
        except OSError:
            return False
        return True


def write_rounds(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _parse(row: dict) -> dict | None:
    """Return a typed copy of the row, or None if it is invalid."""
    if None in row or any(row.get(c) in (None, "") for c in CSV_COLUMNS):
        return None
    try:
        parsed = dict(row)
        parsed["round"] = int(row["round"])
        parsed["vidas_jogador"] = int(row["vidas_jogador"])
        parsed["vidas_ia"] = int(row["vidas_ia"])
    except ValueError:
        return None
    if not all(0 <= parsed[k] <= MAX_LIVES for k in ("vidas_jogador", "vidas_ia")):
        return None
    if any(row[c] not in ACTIONS + (NONE,) for c in _HISTORY_COLUMNS):
        return None
    if row["acao_jogador"] not in ACTIONS or row["acao_ia"] not in ACTIONS:
        return None
    return parsed


def read_rounds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [p for p in map(_parse, csv.DictReader(file)) if p is not None]


def load_state(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(state, dict) and isinstance(state.get("partidas_desde_treino"), int):
            return state
    except (OSError, ValueError):
        pass
    return dict(_DEFAULT_STATE)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)

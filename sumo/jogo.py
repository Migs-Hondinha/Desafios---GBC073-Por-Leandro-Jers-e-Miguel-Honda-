"""Sumô × Neurônio: pygame front-end.

States: MENU -> CHOOSING <-> REVEAL -> MATCH_OVER, plus TRAINING (background thread).
The AI picks its move when CHOOSING starts, before any player input is read.
Run with --smoke for a headless end-to-end check.
"""
from __future__ import annotations

import argparse
from enum import Enum, auto
import os
import random
import sys
import threading

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from dados_io import (FAKE_CSV, MODEL_PATH, ROUNDS_CSV, STATE_JSON, RoundWriter,  # noqa: E402
                      load_state, read_rounds, save_state)
from gerar_dados_falsos import ensure_fake_data  # noqa: E402
from neuronio import STEPS, choose_ai_action, load_model, save_model, train_model  # noqa: E402
from regras import ACTIONS, MAX_LIVES, Match  # noqa: E402

AUTO_TRAIN_EVERY = 10
WIDTH, HEIGHT = 960, 540
REVEAL_MS = 1200
LABELS = {"ataque": "Ataque", "defesa": "Defesa", "agarrao": "Agarrão"}
KEYS = {pygame.K_a: "ataque", pygame.K_s: "defesa", pygame.K_d: "agarrao"}
KEY_HINT = {"ataque": "A", "defesa": "S", "agarrao": "D"}

BG = (244, 236, 220)
INK = (40, 32, 28)
MUTED = (120, 108, 96)
SAND = (222, 200, 160)
ROPE = (160, 120, 70)
PLAYER = (52, 96, 170)
AI = (180, 56, 52)
SKIN = (240, 200, 165)
BUTTON = (255, 250, 240)
BUTTON_HOVER = (255, 232, 200)
RESULT_TEXT = {"jogador": "Você venceu o round!", "ia": "O neurônio venceu o round!",
               "empate": "Empate!"}


class State(Enum):
    MENU = auto()
    CHOOSING = auto()
    REVEAL = auto()
    MATCH_OVER = auto()
    TRAINING = auto()


class Game:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("segoeui,arial", 22)
        self.small = pygame.font.SysFont("segoeui,arial", 17)
        self.big = pygame.font.SysFont("segoeui,arial", 46, bold=True)
        self.rng = random.Random()
        self.state_data = load_state(STATE_JSON)
        self.real_rounds = len(read_rounds(ROUNDS_CSV))
        self.match: Match | None = None
        self.ai_action = ""
        self.probs = [1 / 3] * 3
        self.last: tuple[str, str, str] | None = None  # player, ai, result
        self.reveal_until = 0
        self.buttons: list[tuple[pygame.Rect, str]] = []
        self.lock = threading.Lock()
        self.training: dict = {}
        self.train_reason = ""
        self.running = True
        self.writer = RoundWriter(ROUNDS_CSV)
        self.auto_train_pending = False

        ensure_fake_data(FAKE_CSV)
        loaded = load_model(MODEL_PATH)
        self.model, self.meta = loaded if loaded else (None, None)
        self.state = State.MENU
        if self.model is None:
            self.start_training("Primeiro treino (dados falsos)")

    # --- transitions ----------------------------------------------------------
    def new_match(self) -> None:
        self.match = Match()
        self.last = None
        self.enter_choosing()

    def enter_choosing(self) -> None:
        """Decide the AI move FIRST, using only rounds already played."""
        assert self.match is not None
        self.ai_action, self.probs = choose_ai_action(self.model, self.match.context(), self.rng)
        self.state = State.CHOOSING

    def on_choose(self, action: str) -> bool:
        """Play a round; ignored unless we are waiting for the player's choice."""
        if self.state is not State.CHOOSING or self.match is None:
            return False
        row = self.match.play(action, self.ai_action)
        self.writer.write(row)
        self.real_rounds += 1
        self.last = (action, self.ai_action, row["resultado"])
        self.reveal_until = pygame.time.get_ticks() + REVEAL_MS
        self.state = State.REVEAL
        return True

    def finish_reveal(self) -> None:
        if self.state is not State.REVEAL or self.match is None:
            return
        if self.match.finished:
            self.on_match_over()
        else:
            self.enter_choosing()

    def on_match_over(self) -> None:
        self.state = State.MATCH_OVER
        self.state_data["partidas_desde_treino"] += 1
        self.save_state()
        # Train only after the player has seen the result (on "Jogar de novo"/"Menu").
        self.auto_train_pending = self.state_data["partidas_desde_treino"] >= AUTO_TRAIN_EVERY

    def save_state(self) -> None:
        try:
            save_state(STATE_JSON, self.state_data)
        except OSError as error:  # a locked file must not end the match
            print("Não foi possível salvar o estado:", error, file=sys.stderr, flush=True)

    def abandon_match(self) -> None:
        """Back to menu mid-match: rounds stay in the CSV, counter unchanged."""
        self.match = None
        self.state = State.MENU

    def start_training(self, reason: str) -> None:
        self.train_reason = reason
        with self.lock:
            self.training = {"step": 0, "total": STEPS, "done": False, "applied": False,
                             "error": None, "meta": None, "model": None}
        self.state = State.TRAINING
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        def progress(step: int, total: int) -> None:
            with self.lock:
                self.training["step"], self.training["total"] = step, total
        try:
            fake = read_rounds(FAKE_CSV)
            real = read_rounds(ROUNDS_CSV)
            model, meta = train_model(fake, real, progress=progress)
            save_model(model, meta, MODEL_PATH)
            with self.lock:
                self.training.update(model=model, meta=meta, done=True)
        except Exception as error:  # shown on screen; the previous model stays active
            with self.lock:
                self.training.update(error=str(error), done=True)

    def poll_training(self) -> None:
        with self.lock:
            info = dict(self.training)
            if not info.get("done") or info.get("applied"):
                return
            self.training["applied"] = True
        if info["error"] is None:
            self.model, self.meta = info["model"], info["meta"]
            self.state_data["partidas_desde_treino"] = 0
            self.save_state()
            print("Neurônio treinado:", info["meta"], flush=True)
        else:
            print("Falha no treino:", info["error"], file=sys.stderr, flush=True)

    @property
    def training_busy(self) -> bool:
        with self.lock:
            return self.state is State.TRAINING and not self.training.get("done")

    # --- input ------------------------------------------------------------------
    def on_button(self, name: str) -> None:
        if self.auto_train_pending and name in ("Jogar de novo", "Menu"):
            self.auto_train_pending = False
            self.start_training(f"Treino automático ({AUTO_TRAIN_EVERY} partidas)")
        elif name == "Lutar" or name == "Jogar de novo":
            self.new_match()
        elif name == "Treinar neurônio":
            self.start_training("Treino pedido pelo jogador")
        elif name == "Menu":
            self.state = State.MENU
        elif name == "Sair":
            self.running = False
        elif name in ACTIONS:
            self.on_choose(name)

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.state is State.CHOOSING:
                    self.abandon_match()
                elif self.state is State.MENU:
                    self.running = False
            elif event.key in KEYS:
                self.on_choose(KEYS[event.key])
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for rect, name in self.buttons:
                if rect.collidepoint(event.pos):
                    self.on_button(name)
                    break

    def update(self) -> None:
        self.poll_training()
        if self.state is State.REVEAL and pygame.time.get_ticks() >= self.reveal_until:
            self.finish_reveal()

    # --- drawing ----------------------------------------------------------------
    def text(self, value: str, pos: tuple[int, int], font=None, color=INK, center=True) -> None:
        surface = (font or self.font).render(value, True, color)
        rect = surface.get_rect(center=pos) if center else surface.get_rect(topleft=pos)
        self.screen.blit(surface, rect)

    def button(self, label: str, name: str, rect: pygame.Rect) -> None:
        hover = rect.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(self.screen, BUTTON_HOVER if hover else BUTTON, rect, border_radius=10)
        pygame.draw.rect(self.screen, INK, rect, width=2, border_radius=10)
        self.text(label, rect.center)
        self.buttons.append((rect, name))

    def draw_sumo(self, x: int, color: tuple[int, int, int], facing: int, lean: float) -> None:
        cx, ground = int(x + facing * lean * 40), 400
        pygame.draw.ellipse(self.screen, (196, 174, 136), (cx - 75, ground - 12, 150, 24))
        pygame.draw.rect(self.screen, SKIN, (cx - 48, ground - 70, 30, 70), border_radius=12)
        pygame.draw.rect(self.screen, SKIN, (cx + 18, ground - 70, 30, 70), border_radius=12)
        pygame.draw.circle(self.screen, SKIN, (cx, ground - 120), 72)
        pygame.draw.rect(self.screen, color, (cx - 70, ground - 92, 140, 34), border_radius=8)
        pygame.draw.rect(self.screen, color, (cx - 14, ground - 92, 28, 54), border_radius=6)
        arm_x = cx + facing * 62
        pygame.draw.circle(self.screen, SKIN, (arm_x, ground - 135), 24)
        pygame.draw.circle(self.screen, SKIN, (cx, ground - 212), 36)
        pygame.draw.circle(self.screen, INK, (cx - facing * 6, ground - 250), 14)
        pygame.draw.circle(self.screen, INK, (cx + facing * 14, ground - 216), 4)

    def draw_hearts(self, lives: int, x: int, color: tuple[int, int, int], step: int) -> None:
        for i in range(MAX_LIVES):
            hx = x + i * step
            fill = color if i < lives else (210, 200, 188)
            pygame.draw.circle(self.screen, fill, (hx - 5, 32), 7)
            pygame.draw.circle(self.screen, fill, (hx + 5, 32), 7)
            pygame.draw.polygon(self.screen, fill, [(hx - 12, 35), (hx + 12, 35), (hx, 50)])

    def draw_arena(self, lean: float = 0.0) -> None:
        pygame.draw.ellipse(self.screen, SAND, (90, 330, 780, 150))
        pygame.draw.ellipse(self.screen, ROPE, (90, 330, 780, 150), width=6)
        self.draw_sumo(300, PLAYER, 1, lean)
        self.draw_sumo(660, AI, -1, lean)

    def draw_footer(self) -> None:
        if self.auto_train_pending:
            self.text("10 partidas completas: o neurônio vai treinar ao continuar.",
                      (WIDTH // 2, 250), self.small, AI)
        left = AUTO_TRAIN_EVERY - self.state_data["partidas_desde_treino"]
        self.text(f"Rounds reais coletados: {self.real_rounds}  ·  "
                  f"Treino automático em {left} partida(s)", (WIDTH // 2, HEIGHT - 18),
                  self.small, MUTED)

    def draw(self) -> None:
        self.buttons = []
        self.screen.fill(BG)
        if self.state is State.MENU:
            self.text("Sumô × Neurônio", (WIDTH // 2, 90), self.big)
            self.text("Ataque vence agarrão · agarrão vence defesa · defesa vence ataque",
                      (WIDTH // 2, 145), self.small, MUTED)
            for i, name in enumerate(("Lutar", "Treinar neurônio", "Sair")):
                self.button(name, name, pygame.Rect(WIDTH // 2 - 130, 210 + i * 72, 260, 54))
            if self.meta:
                when = self.meta["trained_at"].replace("T", " às ")
                self.text(f"Neurônio treinado em {when} · acurácia no treino "
                          f"{self.meta['train_accuracy']:.0%}", (WIDTH // 2, 450), self.small, MUTED)
            self.draw_footer()
        elif self.state in (State.CHOOSING, State.REVEAL, State.MATCH_OVER):
            self.draw_fight()
        elif self.state is State.TRAINING:
            self.draw_training()
        if self.writer.pending:
            self.text(f"{self.writer.pending} round(s) aguardando gravação: feche o rounds.csv "
                      "no Excel (serão gravados no próximo round).", (WIDTH // 2, 60), self.small, AI)

    def draw_fight(self) -> None:
        assert self.match is not None
        lean = 0.0
        if self.state is State.REVEAL:
            remaining = max(0, self.reveal_until - pygame.time.get_ticks())
            lean = min(1.0, (REVEAL_MS - remaining) / 250)
        self.draw_arena(lean)
        self.draw_hearts(self.match.player_lives, 40, PLAYER, 30)
        self.draw_hearts(self.match.ai_lives, WIDTH - 310, AI, 30)
        self.text("Você", (40, 70), self.small, PLAYER, center=False)
        self.text("Neurônio", (WIDTH - 110, 70), self.small, AI, center=False)
        self.text(f"Round {self.match.round_number + (self.state is State.CHOOSING)}",
                  (WIDTH // 2, 36), self.font)
        if self.state is State.CHOOSING:
            self.text("Escolha seu golpe (A / S / D) · ESC volta ao menu", (WIDTH // 2, 78),
                      self.small, MUTED)
            for i, action in enumerate(ACTIONS):
                rect = pygame.Rect(WIDTH // 2 - 315 + i * 215, 470, 200, 50)
                self.button(f"{LABELS[action]} [{KEY_HINT[action]}]", action, rect)
        elif self.last is not None:
            player, ai, result = self.last
            self.text(LABELS[player], (300, 108), self.big, PLAYER)
            self.text(LABELS[ai], (660, 108), self.big, AI)
            panel = pygame.Rect(WIDTH // 2 - 330, 452, 660, 70)
            pygame.draw.rect(self.screen, BUTTON, panel, border_radius=12)
            pygame.draw.rect(self.screen, INK, panel, width=2, border_radius=12)
            self.text(RESULT_TEXT[result], (WIDTH // 2, 474), self.font)
            guess = "  ·  ".join(f"{LABELS[a].lower()} {p:.0%}" for a, p in zip(ACTIONS, self.probs))
            self.text(f"Neurônio previu: {guess}", (WIDTH // 2, 500), self.small, MUTED)
        if self.state is State.MATCH_OVER:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((244, 236, 220, 242))
            self.screen.blit(overlay, (0, 0))
            won = self.match.winner == "jogador"
            self.text("Vitória!" if won else "Derrota!", (WIDTH // 2, 170), self.big,
                      PLAYER if won else AI)
            self.text(f"Partida em {self.match.round_number} rounds", (WIDTH // 2, 225), self.font)
            self.button("Jogar de novo", "Jogar de novo", pygame.Rect(WIDTH // 2 - 280, 290, 260, 54))
            self.button("Menu", "Menu", pygame.Rect(WIDTH // 2 + 20, 290, 260, 54))
            self.draw_footer()

    def draw_training(self) -> None:
        with self.lock:
            info = dict(self.training)
        self.text(self.train_reason, (WIDTH // 2, 120), self.big)
        bar = pygame.Rect(WIDTH // 2 - 300, 230, 600, 34)
        pygame.draw.rect(self.screen, BUTTON, bar, border_radius=8)
        fill = bar.copy()
        fill.width = int(bar.width * info["step"] / max(1, info["total"]))
        pygame.draw.rect(self.screen, PLAYER, fill, border_radius=8)
        pygame.draw.rect(self.screen, INK, bar, width=2, border_radius=8)
        self.text(f"Passo {info['step']} / {info['total']}", (WIDTH // 2, 290), self.small, MUTED)
        if not info["done"]:
            self.text("Aguarde o neurônio ficar pronto…", (WIDTH // 2, 340))
            return
        if info["error"]:
            self.text(f"Falha no treino: {info['error']}", (WIDTH // 2, 340), color=AI)
        elif info["meta"]:
            meta = info["meta"]
            self.text(f"Pronto! Acurácia no treino: {meta['train_accuracy']:.0%} · rounds reais: "
                      f"{meta['real_rows']} · usou falsos: {'sim' if meta['used_fake'] else 'não'}",
                      (WIDTH // 2, 340))
        self.button("Lutar", "Lutar", pygame.Rect(WIDTH // 2 - 280, 400, 260, 54))
        self.button("Menu", "Menu", pygame.Rect(WIDTH // 2 + 20, 400, 260, 54))


def _smoke(game: Game, clock: pygame.time.Clock) -> int:
    """Headless end-to-end check: train if needed, play one full match."""
    def frames(n: int) -> None:
        for _ in range(n):
            pygame.event.pump()
            game.update()
            game.draw()
            clock.tick(240)

    for _ in range(6000):  # up to ~25 s waiting for the first training
        if not game.training_busy:
            break
        frames(1)
    frames(2)
    assert game.model is not None, "modelo não ficou pronto"
    game.on_button("Lutar")
    before = len(read_rounds(ROUNDS_CSV))
    assert game.on_choose("ataque") is True
    assert game.on_choose("ataque") is False, "segundo input no mesmo round foi aceito"
    assert len(read_rounds(ROUNDS_CSV)) == before + 1
    frames(5)
    rng = random.Random(1)
    while game.state is not State.MATCH_OVER:
        if game.state is State.REVEAL:
            game.finish_reveal()
        elif game.state is State.CHOOSING:
            game.on_choose(rng.choice(ACTIONS))
        else:
            raise AssertionError(f"estado inesperado {game.state}")
        frames(1)
    frames(30)
    print(f"SMOKE OK: {game.match.round_number} rounds, vencedor={game.match.winner}, "
          f"rounds reais no CSV={len(read_rounds(ROUNDS_CSV))}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sumô × Neurônio")
    parser.add_argument("--smoke", action="store_true", help="teste automático sem janela")
    args = parser.parse_args(argv)
    if args.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_caption("Sumô × Neurônio")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    game = Game(screen)
    try:
        if args.smoke:
            return _smoke(game, clock)
        while game.running:
            for event in pygame.event.get():
                game.handle(event)
            game.update()
            game.draw()
            pygame.display.flip()
            clock.tick(60)
        return 0
    finally:
        game.writer.write()  # last try for rounds queued while the CSV was locked
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())

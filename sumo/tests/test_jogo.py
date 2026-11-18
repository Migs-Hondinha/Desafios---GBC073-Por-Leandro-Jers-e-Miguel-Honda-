import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame  # noqa: E402

import jogo  # noqa: E402


class TestAutoTrain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        self.patches = [mock.patch.object(jogo, name, d / file) for name, file in (
            ("ROUNDS_CSV", "rounds.csv"), ("FAKE_CSV", "falsos.csv"),
            ("MODEL_PATH", "modelo.pt"), ("STATE_JSON", "estado.json"))]
        for p in self.patches:
            p.start()
        pygame.init()
        self.game = jogo.Game(pygame.display.set_mode((jogo.WIDTH, jogo.HEIGHT)))
        for _ in range(3000):  # wait for the first training
            if not self.game.training_busy:
                break
            pygame.time.wait(10)
        self.game.update()

    def tearDown(self):
        for _ in range(3000):  # let a running training release the temp files
            if not self.game.training_busy:
                break
            pygame.time.wait(10)
        pygame.quit()
        for p in self.patches:
            p.stop()
        self._tmp.cleanup()

    def _finish_match(self):
        self.game.on_button("Lutar")
        while self.game.state is not jogo.State.MATCH_OVER:
            if self.game.state is jogo.State.REVEAL:
                self.game.finish_reveal()
            elif self.game.state is jogo.State.CHOOSING:
                self.game.on_choose("ataque")
            else:
                self.fail(f"estado inesperado: {self.game.state}")

    def test_tenth_match_shows_result_before_training(self):
        self.game.state_data["partidas_desde_treino"] = jogo.AUTO_TRAIN_EVERY - 1
        self._finish_match()
        self.assertIs(self.game.state, jogo.State.MATCH_OVER)
        self.game.on_button("Menu")
        self.assertIs(self.game.state, jogo.State.TRAINING)


if __name__ == "__main__":
    unittest.main()

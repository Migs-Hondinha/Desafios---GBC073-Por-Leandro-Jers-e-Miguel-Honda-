import unittest

from neuronio import INPUT_SIZE, encode, row_to_example
from regras import ACTIONS, Match


class TestFeatures(unittest.TestCase):
    def test_encode_empty_history(self):
        x = encode(Match().context())
        self.assertEqual(len(x), INPUT_SIZE)
        self.assertEqual(INPUT_SIZE, 26)
        none_slots = [3, 7, 11, 15, 19, 23]
        for i in range(24):
            self.assertEqual(x[i], 1.0 if i in none_slots else 0.0, i)
        self.assertEqual(x[24:], [1.0, 1.0])

    def test_encode_partial(self):
        m = Match()
        m.play("defesa", "ataque")
        x = encode(m.context())
        self.assertEqual(x[0:4], [0.0, 1.0, 0.0, 0.0])   # player t1 = defesa
        self.assertEqual(x[4:8], [0.0, 0.0, 0.0, 1.0])   # player t2 = nenhum
        self.assertEqual(x[12:16], [1.0, 0.0, 0.0, 0.0])  # ai t1 = ataque

    def test_lives_are_fractions(self):
        m = Match()
        m.play("ataque", "defesa")  # player loses one life
        self.assertAlmostEqual(encode(m.context())[24], 0.9)

    def test_row_to_example_matches_encode(self):
        m = Match()
        m.play("ataque", "defesa")
        ctx_before = m.context()
        row = m.play("agarrao", "ataque")
        x, y = row_to_example(row)
        self.assertEqual(x, encode(ctx_before))
        self.assertEqual(y, ACTIONS.index("agarrao"))


def repeat_rows(n_matches: int = 60, seed: int = 0) -> list[dict]:
    """Synthetic player that always repeats its last move (random opening)."""
    import random
    rng = random.Random(seed)
    rows = []
    for _ in range(n_matches):
        m = Match()
        while not m.finished:
            move = m.player_history[-1] if m.player_history else rng.choice(ACTIONS)
            rows.append(m.play(move, rng.choice(ACTIONS)))
    return rows


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from neuronio import train_model
        cls.rows = repeat_rows()
        cls.model, cls.meta = train_model(cls.rows, [])

    def test_mlp_shape(self):
        import torch
        from neuronio import MLP
        model = MLP()
        self.assertEqual(model(torch.zeros(5, 26)).shape, (5, 3))
        dims = [(l.in_features, l.out_features) for l in model.layers]
        self.assertEqual(dims, [(26, 64), (64, 64), (64, 64), (64, 3)])

    def test_weights_balance_when_few_real(self):
        from neuronio import build_training_set
        fake = repeat_rows(100)[:1000]
        real = repeat_rows(5, seed=9)[:50]
        x, y, w = build_training_set(fake, real)
        self.assertEqual(x.shape[0], 1050)
        self.assertAlmostEqual(w[:1000].sum().item(), w[1000:].sum().item(), places=4)

    def test_real_only_after_threshold(self):
        from neuronio import build_training_set
        fake = repeat_rows(100)[:1000]
        real = repeat_rows(20, seed=9)[:200]
        x, _, _ = build_training_set(fake, real)
        self.assertEqual(x.shape[0], 200)

    def test_no_data_raises(self):
        from neuronio import build_training_set
        with self.assertRaises(ValueError):
            build_training_set([], [])

    def test_training_learns_repeat_pattern(self):
        self.assertGreater(len(self.rows), 600)
        self.assertGreater(self.meta["train_accuracy"], 0.9)
        self.assertFalse(self.meta["used_fake"] is None)

    def test_progress_called_each_step(self):
        from neuronio import train_model
        calls = []
        train_model(self.rows[:50], [], progress=lambda s, t: calls.append((s, t)), steps=10)
        self.assertEqual(calls, [(i, 10) for i in range(1, 11)])

    def test_save_load_roundtrip(self):
        import tempfile, torch
        from pathlib import Path
        from neuronio import load_model, save_model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "modelo.pt"
            save_model(self.model, self.meta, path)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            loaded, meta = load_model(path)
            x = torch.tensor([encode(Match().context())])
            self.assertTrue(torch.allclose(loaded(x), self.model(x)))
            self.assertEqual(meta["train_accuracy"], self.meta["train_accuracy"])

    def test_load_corrupt_returns_none(self):
        import tempfile, torch
        from pathlib import Path
        from neuronio import load_model
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "lixo.pt"
            bad.write_bytes(b"lixo")
            self.assertIsNone(load_model(bad))
            self.assertIsNone(load_model(Path(tmp) / "ausente.pt"))
            other = Path(tmp) / "outro.pt"
            torch.save({"version": 1, "input_size": 26, "width": 32, "depth": 4,
                        "state_dict": {}, "meta": {}}, other)
            self.assertIsNone(load_model(other))

    def test_load_checkpoint_without_meta_returns_none(self):
        import tempfile, torch
        from pathlib import Path
        from neuronio import load_model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sem_meta.pt"
            torch.save({"version": 1, "input_size": 26, "width": 64, "depth": 4,
                        "state_dict": self.model.state_dict()}, path)
            self.assertIsNone(load_model(path))
            torch.save({"version": 1, "input_size": 26, "width": 64, "depth": 4,
                        "state_dict": self.model.state_dict(), "meta": "x"}, path)
            self.assertIsNone(load_model(path))


class TestAIDecision(unittest.TestCase):
    def test_uniform_without_model(self):
        from neuronio import predict_proba
        self.assertEqual(predict_proba(None, Match().context()), [1 / 3] * 3)

    def test_counter_of_confident_prediction(self):
        import random
        from neuronio import choose_ai_action, train_model
        model, _ = train_model(repeat_rows(), [])
        m = Match()
        m.play("agarrao", "agarrao")  # player likely repeats agarrao -> AI should attack
        rng = random.Random(0)
        picks = [choose_ai_action(model, m.context(), rng)[0] for _ in range(200)]
        self.assertGreater(picks.count("ataque"), 150)

    def test_ai_decision_cannot_see_current_action(self):
        import inspect
        from dataclasses import fields
        from neuronio import choose_ai_action
        from regras import RoundContext
        self.assertEqual(list(inspect.signature(choose_ai_action).parameters),
                         ["model", "ctx", "rng"])
        self.assertEqual({f.name for f in fields(RoundContext)},
                         {"player_last", "ai_last", "player_lives", "ai_lives"})


if __name__ == "__main__":
    unittest.main()

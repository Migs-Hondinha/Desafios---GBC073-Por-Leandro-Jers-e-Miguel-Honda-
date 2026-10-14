"""Check data isolation and a real, repeatable training run."""
import csv
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class TrainingTests(unittest.TestCase):
    def test_muon_owns_only_hidden_weights_and_adamw_owns_the_rest(self):
        import treino
        self.assertTrue(hasattr(treino, "make_optimizers"), "Muon factory is missing")
        import torch
        model = treino.MLP(10, 3, 16, 4)
        args = argparse.Namespace(otimizador="muon", lr=0.02, lr_adamw=0.001,
                                  weight_decay=0.01)
        optimizers = treino.make_optimizers(model, args)
        self.assertIsInstance(optimizers["muon"], torch.optim.Muon)
        self.assertEqual(optimizers["muon"].defaults["ns_steps"], 5)
        owned = {name: {id(p) for group in optimizer.param_groups for p in group["params"]}
                 for name, optimizer in optimizers.items()}
        self.assertEqual(owned["muon"], {id(layer.weight) for layer in model.layers[:-1]})
        self.assertFalse(owned["muon"] & owned["adamw"])
        self.assertEqual(owned["muon"] | owned["adamw"], {id(p) for p in model.parameters()})
        before = [p.detach().clone() for p in model.parameters()]
        model(torch.randn(8, 10)).square().mean().backward()
        for optimizer in optimizers.values():
            optimizer.step()
        self.assertTrue(all(not torch.equal(p, old) for p, old in zip(model.parameters(), before)))

    def test_schedule_shapes(self):
        import treino
        self.assertTrue(hasattr(treino, "schedule_factor"), "Schedules are missing")
        curves = {name: [treino.schedule_factor(name, step, 100, 0.1)
                         for step in range(1, 101)]
                  for name in ("constante", "cosseno", "warmup_cosseno", "degraus")}
        self.assertEqual(curves["constante"], [1.0] * 100)
        cosine = curves["cosseno"]
        self.assertEqual(cosine[0], 1.0)
        self.assertAlmostEqual(cosine[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(cosine, cosine[1:])))
        warmup = curves["warmup_cosseno"]
        self.assertAlmostEqual(warmup[0], 0.1)
        self.assertEqual(warmup[9], 1.0)
        self.assertGreater(warmup[9], warmup[10])
        self.assertAlmostEqual(warmup[-1], 0.0)
        self.assertAlmostEqual(curves["degraus"][49], 1.0)
        self.assertAlmostEqual(curves["degraus"][50], 0.1)
        self.assertAlmostEqual(curves["degraus"][75], 0.01)

    def test_muon_cli_supports_all_schedules_and_logs_both_rates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            for schedule in ("constante", "cosseno", "warmup_cosseno", "degraus"):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", str(ROOT / "treino.py"),
                     "--otimizador", "muon", "--lr", "0.02", "--lr-adamw", "0.001",
                     "--agenda", schedule, "--semente", "7", "--passos", "20",
                     "--largura", "16", "--saida", temporary, "--nome", schedule],
                    capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                with (Path(temporary) / "logs" / f"{schedule}.csv").open() as file:
                    rows = list(csv.DictReader(file))
                expected = {"constante": (0.02, 0.02), "cosseno": (0.02, 0.0),
                            "warmup_cosseno": (0.01, 0.0), "degraus": (0.02, 0.0002)}
                self.assertAlmostEqual(float(rows[0]["lr"]), expected[schedule][0])
                self.assertAlmostEqual(float(rows[-1]["lr"]), expected[schedule][1])
                for row in rows:
                    self.assertAlmostEqual(float(row["lr_adamw"]), float(row["lr"]) / 20)
                metadata = json.loads((Path(temporary) / "logs" / f"{schedule}.json").read_text())
                self.assertEqual(metadata["semente"], 7)
                self.assertEqual(metadata["lr_adamw"], 0.001)
            with (Path(temporary) / "experimentos.csv").open() as file:
                self.assertTrue(all(row["diverged"] == "False" for row in csv.DictReader(file)))

    def test_csv_schema_extension_preserves_old_results(self):
        from treino import append_csv
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "results.csv"
            append_csv(path, {"run": "old", "lr": 0.05})
            append_csv(path, {"run": "new", "lr": 0.02, "lr_adamw": 0.001})
            with path.open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0], {"run": "old", "lr": "0.05", "lr_adamw": ""})
            self.assertEqual(rows[1]["lr_adamw"], "0.001")

    def test_training_is_repeatable_and_writes_diagnostics(self):
        self.assertTrue((ROOT / "treino.py").exists(), "Training entry point is missing")
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary)
            for run in ("first", "second"):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", str(ROOT / "treino.py"), "--passos", "20",
                     "--largura", "16", "--saida", str(output), "--nome", run],
                    capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            logs = []
            for run in ("first", "second"):
                with (output / "logs" / f"{run}.csv").open() as file:
                    logs.append(list(csv.DictReader(file)))
            self.assertEqual(logs[0], logs[1])
            self.assertEqual([int(row["step"]) for row in logs[0]], [1, 20])
            self.assertGreater(float(logs[0][-1]["global_update_ratio"]), 0)
            self.assertIn("layer_3_grad_norm", logs[0][-1])
            with (output / "experimentos.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            metadata = json.loads((output / "logs" / "first.json").read_text())
            self.assertFalse(metadata["test_evaluated"])

    def test_split_is_disjoint_and_scaling_uses_training_only(self):
        self.assertTrue((ROOT / "treino.py").exists(), "Training entry point is missing")
        import torch
        from treino import load_data, DEFAULT_DATA

        data = load_data(DEFAULT_DATA)
        train, validation, test = [set(data[key].tolist()) for key in
                                   ("train_indices", "validation_indices", "test_indices")]
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(len(train | validation | test), data["row_count"])
        self.assertTrue(torch.allclose(data["train_x"].mean(0),
                                       torch.zeros(data["train_x"].shape[1]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()

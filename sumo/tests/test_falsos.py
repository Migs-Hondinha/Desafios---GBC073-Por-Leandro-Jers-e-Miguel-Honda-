import unittest

from gerar_dados_falsos import generate_fake_rows
from neuronio import row_to_example


class TestFakeData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_fake_rows(300)

    def test_fake_rows_valid_and_deterministic(self):
        a = generate_fake_rows(20, seed=1)
        b = generate_fake_rows(20, seed=1)
        strip = lambda rows: [{k: v for k, v in r.items() if k != "timestamp"} for r in rows]
        self.assertEqual(strip(a), strip(b))
        self.assertGreaterEqual(len(a), 200)
        for row in a:
            row_to_example(row)

    def test_fake_rows_have_learnable_pattern(self):
        rows = [r for r in self.rows if r["jogador_t1"] != "nenhum"]
        repeats = sum(r["acao_jogador"] == r["jogador_t1"] for r in rows) / len(rows)
        self.assertGreater(repeats, 0.40)

    def test_fake_opening_bias(self):
        openings = [r for r in self.rows if r["round"] == 1]
        attack = sum(r["acao_jogador"] == "ataque" for r in openings) / len(openings)
        self.assertGreater(attack, 0.40)


if __name__ == "__main__":
    unittest.main()

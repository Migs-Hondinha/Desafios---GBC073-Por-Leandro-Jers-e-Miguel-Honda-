from pathlib import Path
import tempfile
import unittest

from dados_io import CSV_COLUMNS, append_round, load_state, read_rounds, save_state
from regras import Match


class TestDataIO(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_then_read_roundtrip(self):
        path = self.dir / "sub" / "rounds.csv"
        m = Match()
        append_round(path, m.play("ataque", "defesa"))
        append_round(path, m.play("agarrao", "ataque"))
        rows = read_rounds(path)
        self.assertEqual(len(rows), 2)
        self.assertIsInstance(rows[1]["vidas_jogador"], int)
        self.assertEqual(rows[1]["vidas_jogador"], 9)
        self.assertEqual(rows[1]["round"], 2)

    def test_header_written_once(self):
        path = self.dir / "rounds.csv"
        m = Match()
        append_round(path, m.play("ataque", "defesa"))
        append_round(path, m.play("ataque", "defesa"))
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum(line.startswith("timestamp,") for line in lines), 1)

    def test_missing_file_is_empty(self):
        self.assertEqual(read_rounds(self.dir / "x.csv"), [])

    def test_invalid_rows_skipped(self):
        path = self.dir / "rounds.csv"
        good = Match().play("ataque", "defesa")
        bad = dict(good, acao_jogador="chute")
        header = ",".join(CSV_COLUMNS)
        to_line = lambda row: ",".join(str(row[c]) for c in CSV_COLUMNS)
        path.write_text("\n".join([header, to_line(good), to_line(bad),
                                   "2026-11-18,abc"]) + "\n", encoding="utf-8")
        self.assertEqual(len(read_rounds(path)), 1)

    def test_append_after_truncated_line_keeps_new_row(self):
        path = self.dir / "rounds.csv"
        m = Match()
        append_round(path, m.play("ataque", "defesa"))
        text = path.read_text(encoding="utf-8")
        path.write_text(text[:-8], encoding="utf-8")  # crash mid-line: no final newline
        append_round(path, m.play("agarrao", "ataque"))
        rows = read_rounds(path)
        self.assertEqual([r["round"] for r in rows], [2])

    def test_foreign_header_is_backed_up_and_new_rows_kept(self):
        path = self.dir / "rounds.csv"
        path.write_text("timestamp;partida_id;round\nx;y;1\n", encoding="utf-8")  # Excel pt-BR
        append_round(path, Match().play("ataque", "defesa"))
        self.assertEqual(len(read_rounds(path)), 1)
        backups = list(self.dir.glob("rounds.bak-*.csv"))
        self.assertEqual(len(backups), 1)
        self.assertIn("timestamp;partida_id", backups[0].read_text(encoding="utf-8"))

    def test_writer_queues_rows_while_file_is_locked(self):
        from dados_io import RoundWriter
        path = self.dir / "rounds.csv"
        path.mkdir()  # opening a directory fails like a file locked by Excel
        writer = RoundWriter(path)
        m = Match()
        self.assertFalse(writer.write(m.play("ataque", "defesa")))
        self.assertEqual(writer.pending, 1)
        path.rmdir()
        self.assertTrue(writer.write(m.play("ataque", "defesa")))
        self.assertEqual(writer.pending, 0)
        self.assertEqual([r["round"] for r in read_rounds(path)], [1, 2])

    def test_state_default_and_roundtrip(self):
        path = self.dir / "estado.json"
        self.assertEqual(load_state(path), {"partidas_desde_treino": 0})
        save_state(path, {"partidas_desde_treino": 7})
        self.assertEqual(load_state(path), {"partidas_desde_treino": 7})
        path.write_text("{lixo", encoding="utf-8")
        self.assertEqual(load_state(path), {"partidas_desde_treino": 0})


if __name__ == "__main__":
    unittest.main()

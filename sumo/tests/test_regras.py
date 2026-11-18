import unittest

from regras import Match, counter, round_result


class TestRules(unittest.TestCase):
    def test_counter_cycle(self):
        self.assertEqual(counter("ataque"), "defesa")
        self.assertEqual(counter("defesa"), "agarrao")
        self.assertEqual(counter("agarrao"), "ataque")

    def test_round_result(self):
        self.assertEqual(round_result("ataque", "agarrao"), "jogador")
        self.assertEqual(round_result("agarrao", "defesa"), "jogador")
        self.assertEqual(round_result("defesa", "ataque"), "jogador")
        self.assertEqual(round_result("ataque", "defesa"), "ia")
        self.assertEqual(round_result("defesa", "defesa"), "empate")

    def test_empty_history_context(self):
        ctx = Match().context()
        self.assertEqual(ctx.player_last, ("nenhum",) * 3)
        self.assertEqual(ctx.ai_last, ("nenhum",) * 3)
        self.assertEqual((ctx.player_lives, ctx.ai_lives), (10, 10))

    def test_partial_history_most_recent_first(self):
        m = Match()
        m.play("ataque", "ataque")
        m.play("defesa", "ataque")
        self.assertEqual(m.context().player_last, ("defesa", "ataque", "nenhum"))
        self.assertEqual(m.context().ai_last, ("ataque", "ataque", "nenhum"))

    def test_history_keeps_only_last_three(self):
        m = Match()
        for action in ("ataque", "defesa", "agarrao", "defesa"):
            m.play(action, action)
        self.assertEqual(m.context().player_last, ("defesa", "agarrao", "defesa"))

    def test_draw_keeps_lives_and_loser_loses_one(self):
        m = Match()
        m.play("ataque", "ataque")
        self.assertEqual((m.player_lives, m.ai_lives), (10, 10))
        m.play("ataque", "defesa")
        self.assertEqual((m.player_lives, m.ai_lives), (9, 10))

    def test_match_ends_at_zero(self):
        m = Match()
        for _ in range(10):
            self.assertFalse(m.finished)
            m.play("ataque", "agarrao")
        self.assertTrue(m.finished)
        self.assertEqual(m.winner, "jogador")
        with self.assertRaises(ValueError):
            m.play("ataque", "agarrao")

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            Match().play("chute", "ataque")

    def test_play_row_uses_context_before_round(self):
        m = Match()
        m.play("ataque", "defesa")
        row = m.play("agarrao", "ataque")
        self.assertEqual(row["jogador_t1"], "ataque")
        self.assertEqual(row["ia_t1"], "defesa")
        self.assertEqual(row["vidas_jogador"], 9)
        self.assertEqual(row["acao_jogador"], "agarrao")
        self.assertEqual(row["resultado"], "ia")
        self.assertEqual(row["round"], 2)
        self.assertEqual(row["partida_id"], m.match_id)


if __name__ == "__main__":
    unittest.main()

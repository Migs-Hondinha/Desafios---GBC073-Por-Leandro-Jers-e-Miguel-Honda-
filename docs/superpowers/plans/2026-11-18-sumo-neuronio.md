# Sumô × Neurônio — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Jogo pygame de sumô (ataque/defesa/agarrão, 10 vidas) contra o MLP do grupo, que prevê o próximo golpe do jogador com base nos 3 anteriores. O jogo coleta os rounds em CSV, retreina por botão ou a cada 10 partidas e é distribuído como `.zip` com `jogar.bat`.

**Architecture:** `regras.py` contém as regras puras, sem torch e sem pygame. `dados_io.py` faz a persistência em CSV/JSON. `neuronio.py` reúne features, o MLP (com `ativacao`/`inicializar` do desafio 2), o treino Muon+cosseno, a predição e a decisão da IA. `gerar_dados_falsos.py` cria o dataset de partida. `jogo.py` é a interface pygame, uma máquina de estados com treino em thread.

**Tech Stack:** Python 3.12 (via `uv`), PyTorch ≥ 2.9 CPU, pygame 2.6, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-18-sumo-neuronio-design.md`

## Global Constraints

- PyTorch apenas; nunca Keras/TensorFlow.
- Ações: `"ataque"`, `"defesa"`, `"agarrao"`; vazio = `"nenhum"`. Índices de classe: ataque=0, defesa=1, agarrao=2.
- Ataque vence agarrão, agarrão vence defesa e defesa vence ataque. Ações iguais empatam e ninguém perde vida.
- `MAX_LIVES = 10`. Histórico de 3 golpes. Features: 26 floats.
- MLP: 26 → 64 → 64 → 64 → 3, `ativacao`/`inicializar` importados de `../desafio2/desafio2_Leandro_Marques_Miguel_Cesario.py`.
- Treino: Muon(lr 0,1, momentum 0,95, nesterov, ns_steps 5, adjust_lr_fn "original", wd 0,01) nas 3 matrizes ocultas, incluindo 26→64. AdamW(lr 0,01, wd 0,01) nos vieses e na camada de saída. Agenda cosseno (fórmula de `desafio3/treino.py`), 225 passos, lote `min(128, n)`, semente 0, sempre do zero.
- Mistura: falsos + reais com peso total 50/50 se houver menos de 200 rounds reais (`REAL_ONLY_THRESHOLD = 200`); caso contrário, só reais.
- Treino automático a cada 10 partidas completas (`AUTO_TRAIN_EVERY = 10`). O botão Treinar zera o contador. Partida abandonada não conta.
- A IA decide antes de ler o input do jogador do round atual.
- Identificadores em inglês, textos da interface em português.
- `torch>=2.9`, CPU index do PyTorch. Arquivos de runtime em `sumo/dados/` (ignorado pelo git e excluído do zip).
- Não fazer commit sem pedido explícito do Leandro. Os passos "Commit" abaixo ficam **pendentes até ele autorizar**.

## Review Focus

1. **Janela fechada durante o treino:** o processo encerra sem travar, e `modelo.pt` nunca fica meio escrito (gravar em `.tmp` e depois `os.replace`). Coberto na Task 4.
2. **`modelo.pt` incompatível ou corrompido** (outro formato, outra versão): `load_model` retorna `None` e o jogo retreina, sem crash. Coberto na Task 4.
3. **`rounds.csv` com linha truncada ou editada no Excel** (ação inválida, coluna faltando): a linha é ignorada e as demais são lidas. Coberto na Task 2.
4. **Pasta extraída com espaços ou acentos** (`C:\Users\João\Downloads\sumo neuronio\`): `jogar.bat` funciona, com todos os caminhos entre aspas e paths de dados relativos a `__file__`. Coberto na Task 7.
5. **Clique duplo ou tecla durante a revelação do round:** não pode gerar dois rounds. O input é ignorado fora do estado `CHOOSING`. Coberto na Task 6.

---

## File Structure

```
entrega-gbc073/
  .gitignore                 # Modify: add sumo/dados/, sumo/.venv/, dist/
  sumo/
    pyproject.toml           # Task 1
    .python-version          # Task 1: "3.12"
    regras.py                # Task 1
    dados_io.py              # Task 2
    neuronio.py              # Tasks 3–5
    gerar_dados_falsos.py    # Task 3
    jogo.py                  # Task 6
    jogar.bat                # Task 7
    README.md                # Task 7
    tests/__init__.py
    tests/test_regras.py     # Task 1
    tests/test_dados_io.py   # Task 2
    tests/test_neuronio.py   # Tasks 3–5
    tests/test_falsos.py     # Task 3
  empacotar.ps1              # Task 7
```

Executar os testes sempre a partir de `sumo/`: `uv run python -m unittest -v`.

---

### Task 1: Projeto + regras do jogo

**Files:**
- Create: `sumo/pyproject.toml`, `sumo/.python-version`, `sumo/regras.py`, `sumo/tests/__init__.py`, `sumo/tests/test_regras.py`
- Modify: `.gitignore` (raiz do repo; criar se não existir)

**Interfaces:**
- Produces (em `regras.py`):
  - `ACTIONS: tuple[str, str, str] = ("ataque", "defesa", "agarrao")`, `NONE = "nenhum"`, `MAX_LIVES = 10`, `HISTORY = 3`
  - `counter(action: str) -> str`: a ação que vence `action`
  - `round_result(player: str, ai: str) -> str`: `"jogador"` | `"ia"` | `"empate"`
  - `@dataclass(frozen=True) RoundContext`: `player_last: tuple[str, str, str]`, `ai_last: tuple[str, str, str]` (mais recente primeiro, completado com `NONE`), `player_lives: int`, `ai_lives: int`
  - `class Match`: `__init__(self, match_id: str | None = None)` (gera um `uuid4().hex` quando não recebe id); atributos `match_id`, `round_number` (começa em 0), `player_lives`, `ai_lives`, `player_history: list[str]`, `ai_history: list[str]`; `context() -> RoundContext`; `play(player: str, ai: str) -> dict` (aplica o round e retorna a linha CSV, ver Task 2 `CSV_COLUMNS`); `finished: bool` (property); `winner: str | None` (property, `"jogador"`/`"ia"`).

- [ ] **Step 1: `pyproject.toml`**

```toml
[project]
name = "sumo-neuronio"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = ["torch>=2.9", "pygame>=2.6"]

[tool.uv]
package = false

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

`.python-version`: `3.12`. Adicionar ao `.gitignore`: `sumo/dados/`, `sumo/.venv/`, `dist/`, `__pycache__/`.

- [ ] **Step 2: Testes que falham** (`tests/test_regras.py`)

```python
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
    self.assertEqual((ctx.player_lives, ctx.ai_lives), (10, 10))

def test_partial_history_most_recent_first(self):
    m = Match(); m.play("ataque", "ataque"); m.play("defesa", "ataque")
    self.assertEqual(m.context().player_last, ("defesa", "ataque", "nenhum"))
    self.assertEqual(m.context().ai_last, ("ataque", "ataque", "nenhum"))

def test_draw_keeps_lives_and_loser_loses_one(self):
    m = Match(); m.play("ataque", "ataque")
    self.assertEqual((m.player_lives, m.ai_lives), (10, 10))
    m.play("ataque", "defesa")
    self.assertEqual((m.player_lives, m.ai_lives), (9, 10))

def test_match_ends_at_zero(self):
    m = Match()
    for _ in range(10): m.play("ataque", "agarrao")
    self.assertTrue(m.finished); self.assertEqual(m.winner, "jogador")
    with self.assertRaises(ValueError): m.play("ataque", "agarrao")

def test_invalid_action_rejected(self):
    with self.assertRaises(ValueError): Match().play("chute", "ataque")

def test_play_row_uses_context_before_round(self):
    m = Match(); m.play("ataque", "defesa")
    row = m.play("agarrao", "ataque")
    self.assertEqual(row["jogador_t1"], "ataque")
    self.assertEqual(row["vidas_jogador"], 9)   # lives at round START
    self.assertEqual(row["acao_jogador"], "agarrao")
    self.assertEqual(row["round"], 2)
```

- [ ] **Step 3: Rodar** `uv run python -m unittest tests.test_regras -v`. Esperado: FAIL (ImportError).
- [ ] **Step 4: Implementar `regras.py`.** A linha de `play` é montada a partir de `context()` **antes** de aplicar o round. `timestamp` em `datetime.now().isoformat(timespec="seconds")`.
- [ ] **Step 5: Rodar de novo.** Esperado: todos os testes passam.
- [ ] **Step 6: Commit** (pendente de autorização): `feat(sumo): game rules and project scaffold`

---

### Task 2: Persistência (CSV + estado)

**Files:**
- Create: `sumo/dados_io.py`, `sumo/tests/test_dados_io.py`

**Interfaces:**
- Consumes: `regras.ACTIONS`, `regras.NONE`
- Produces:
  - `ROOT = Path(__file__).resolve().parent`; `DATA_DIR = Path(os.environ["SUMO_DATA_DIR"]) if "SUMO_DATA_DIR" in os.environ else ROOT / "dados"` (a variável serve aos smoke tests); `ROUNDS_CSV`, `FAKE_CSV = DATA_DIR / "falsos.csv"`, `MODEL_PATH = DATA_DIR / "modelo.pt"`, `STATE_JSON`
  - `CSV_COLUMNS = ["timestamp", "partida_id", "round", "jogador_t1", "jogador_t2", "jogador_t3", "ia_t1", "ia_t2", "ia_t3", "vidas_jogador", "vidas_ia", "acao_jogador", "acao_ia", "resultado"]`
  - `append_round(path: Path, row: dict) -> None`: cria a pasta e o cabeçalho se for preciso; append; `flush` + `os.fsync`.
  - `write_rounds(path: Path, rows: list[dict]) -> None`: sobrescreve (usado só para falsos).
  - `read_rounds(path: Path) -> list[dict]`: arquivo ausente → `[]`; ignora linhas inválidas (ação fora de `ACTIONS`/`NONE`, vidas não inteiras fora de 0..10, coluna faltando) e converte `round`/`vidas_*` para `int`.
  - `load_state(path: Path) -> dict` (padrão `{"partidas_desde_treino": 0}`; JSON corrompido → padrão); `save_state(path: Path, state: dict) -> None` (escrita atômica via `.tmp` + `os.replace`).

- [ ] **Step 1: Testes que falham**, usando `tempfile.TemporaryDirectory`:
  - `test_append_then_read_roundtrip`: duas linhas de `Match.play` → `read_rounds` retorna 2 linhas, com `vidas_jogador` do tipo `int`.
  - `test_header_written_once`: dois appends → o arquivo tem exatamente 1 linha começando com `timestamp,`.
  - `test_missing_file_is_empty`: `read_rounds(tmp/"x.csv") == []`.
  - `test_invalid_rows_skipped` (Review Focus 3): escrever à mão o cabeçalho + 1 linha válida + 1 linha com `acao_jogador=chute` + 1 linha truncada `2026-11-18,abc` → `len(read_rounds(p)) == 1`.
  - `test_state_default_and_roundtrip`: o estado ausente vem com o padrão; `save_state` e depois `load_state` preservam `{"partidas_desde_treino": 7}`; arquivo com texto `{lixo` → padrão.
- [ ] **Step 2: Rodar.** Esperado: FAIL.
- [ ] **Step 3: Implementar `dados_io.py`.** Usar `csv.DictWriter`/`DictReader` com `encoding="utf-8", newline=""`.
- [ ] **Step 4: Rodar.** Esperado: PASS.
- [ ] **Step 5: Commit** (pendente): `feat(sumo): round CSV and state persistence`

---

### Task 3: Features + dados falsos

**Files:**
- Create: `sumo/neuronio.py` (somente a parte de features nesta task), `sumo/gerar_dados_falsos.py`, `sumo/tests/test_neuronio.py`, `sumo/tests/test_falsos.py`

**Interfaces:**
- Consumes: `regras.RoundContext`, `regras.Match`, `regras.counter`, `dados_io.write_rounds`, `dados_io.FAKE_CSV`
- Produces:
  - `neuronio.INPUT_SIZE = 26`, `neuronio.SLOTS = ("ataque", "defesa", "agarrao", "nenhum")`
  - `neuronio.encode(ctx: RoundContext) -> list[float]`: na ordem player t1, t2, t3, ai t1, t2, t3, cada bloco one-hot em `SLOTS`, seguido de `player_lives/10` e `ai_lives/10`.
  - `neuronio.row_to_example(row: dict) -> tuple[list[float], int]`: reconstrói o `RoundContext` a partir da linha; o alvo é `ACTIONS.index(row["acao_jogador"])`.
  - `gerar_dados_falsos.generate_fake_rows(n_matches: int = 300, seed: int = 0) -> list[dict]`
  - `gerar_dados_falsos.ensure_fake_data(path: Path = FAKE_CSV) -> list[dict]`: gera e grava se não existir; caso exista, lê.

O jogador sintético é sorteado por partida com `random.Random(seed)`. Cada um tem pesos aleatórios sobre 5 comportamentos, e cada round sorteia um comportamento conforme esses pesos:
  - `opening`: no round 1, ação preferida (ataque com prob. 0,5; as demais com 0,25 cada); nos outros rounds, a mesma preferência.
  - `repeat`: repete `player_history[-1]`.
  - `cycle`: ataque → agarrao → defesa → ataque…, a partir do último golpe.
  - `counter_ai`: `counter(ai_history[-1])`.
  - `noise`: uniforme.

  Quando não há histórico, `repeat`, `cycle` e `counter_ai` caem em `opening`. A IA dos falsos é uniforme. Executar como script (`python gerar_dados_falsos.py`) grava `FAKE_CSV` e imprime o nº de rounds e a distribuição.

- [ ] **Step 1: Testes que falham**
  - `test_encode_empty_history`: `encode(Match().context())` tem len 26; os índices 3, 7, 11, 15, 19 e 23 (posição "nenhum") valem 1.0, os 18 valores one-hot restantes valem 0, e as duas últimas posições valem `[1.0, 1.0]`.
  - `test_encode_partial`: após `play("defesa","ataque")`, o bloco t1 do jogador é `[0,1,0,0]` e o t2 é `[0,0,0,1]`.
  - `test_row_to_example_matches_encode`: para uma linha de `Match.play`, `row_to_example(row)[0] == encode(ctx_before)` e o alvo corresponde à ação.
  - `test_fake_rows_valid_and_deterministic`: `generate_fake_rows(20, seed=1)` é igual em duas chamadas; todas as linhas passam por `row_to_example`; há pelo menos 200 linhas.
  - `test_fake_rows_have_learnable_pattern`: em `generate_fake_rows(300)`, a frequência de `acao_jogador == jogador_t1` (com t1 ≠ nenhum) é maior que 0,40, acima do acaso de 0,33.
  - `test_fake_opening_bias`: nas linhas com `round == 1`, ataque aparece em mais de 40% dos casos.
- [ ] **Step 2: Rodar.** Esperado: FAIL.
- [ ] **Step 3: Implementar.** `neuronio.py` ainda **não** importa torch nesta task. As features são listas puras.
- [ ] **Step 4: Rodar.** Esperado: PASS.
- [ ] **Step 5: Commit** (pendente): `feat(sumo): feature encoding and fake dataset`

---

### Task 4: MLP + treino + salvar/carregar

**Files:**
- Modify: `sumo/neuronio.py`
- Modify: `sumo/tests/test_neuronio.py`

**Interfaces:**
- Consumes: `encode`, `row_to_example` (Task 3); `ativacao`, `inicializar` de `desafio2` (via `sys.path.insert(0, str(ROOT.parent / "desafio2"))`, como em `desafio3/treino.py`)
- Produces:
  - Constantes: `WIDTH = 64`, `DEPTH = 4`, `STEPS = 225`, `BATCH = 128`, `LR_MUON = 0.1`, `LR_ADAMW = 0.01`, `WEIGHT_DECAY = 0.01`, `REAL_ONLY_THRESHOLD = 200`, `SEED = 0`
  - `class MLP(nn.Module)`: `__init__(self, input_size: int = INPUT_SIZE, classes: int = 3, width: int = WIDTH, depth: int = DEPTH)`; mesmo corpo de `desafio3/treino.py` (`nn.ModuleList` de `nn.Linear`, `inicializar` por camada, `ativacao` nas ocultas e logits na saída).
  - `build_training_set(fake_rows: list[dict], real_rows: list[dict]) -> tuple[Tensor, Tensor, Tensor]`: retorna `(x, y, weights)`. Com `len(real) >= 200`, usa só reais e pesos 1. Com reais > 0 e < 200, os pesos somam 0,5 para todo o conjunto de falsos e 0,5 para todo o de reais. Sem reais, só falsos com pesos 1. Sem nenhum dado, `ValueError`.
  - `train_model(fake_rows: list[dict], real_rows: list[dict], progress: Callable[[int, int], None] | None = None, steps: int = STEPS) -> tuple[MLP, dict]`. A cada passo, sorteia o lote com `torch.multinomial(weights, batch, replacement=True, generator=g)`. Os otimizadores seguem as Global Constraints, e `factor = 0.5 * (1 + cos(pi * (step-1)/max(1, steps-1)))` é aplicado às duas lr base **antes** de `step()`. Chama `progress(step, steps)` a cada passo. O `meta` retornado traz `{"trained_at", "fake_rows", "real_rows", "used_fake": bool, "train_accuracy", "class_counts", "seconds"}`. Fixa `torch.manual_seed(SEED)` antes de construir a rede.
  - `save_model(model: MLP, meta: dict, path: Path) -> None`: `torch.save({"version": 1, "input_size": INPUT_SIZE, "width": WIDTH, "depth": DEPTH, "state_dict": ..., "meta": meta}, tmp)` e depois `os.replace(tmp, path)`.
  - `load_model(path: Path) -> tuple[MLP, dict] | None`: devolve `None` se o arquivo não existir, se houver qualquer exceção na carga (`torch.load(..., weights_only=True)`) ou se `version`/`input_size`/`width`/`depth` não baterem. Em seguida chama `model.eval()`.

- [ ] **Step 1: Testes que falham**
  - `test_mlp_shape`: `MLP()(torch.zeros(5, 26)).shape == (5, 3)`; as camadas têm dimensões 26→64→64→64→3.
  - `test_weights_balance_when_few_real`: 1.000 falsos e 50 reais → os pesos dos falsos somam ≈ os dos reais (tolerância relativa 1e-5).
  - `test_real_only_after_threshold`: 1.000 falsos e 200 reais → `x.shape[0] == 200`.
  - `test_no_data_raises`: `build_training_set([], [])` lança `ValueError`.
  - `test_training_learns_repeat_pattern`: 600 linhas sintéticas em que `acao_jogador == jogador_t1` sempre (geradas com `Match` e um jogador que repete, abrindo com ação aleatória); `train_model(rows, [], steps=225)` → `meta["train_accuracy"] > 0.9`.
  - `test_progress_called_each_step`: `steps=10` → o callback recebe `(1, 10)` … `(10, 10)`.
  - `test_save_load_roundtrip`: depois de salvar e carregar, os logits do modelo carregado em `encode(Match().context())` são iguais aos do original (`torch.allclose`).
  - `test_load_corrupt_returns_none` (Review Focus 2): arquivo com bytes `b"lixo"` → `None`; checkpoint salvo com `width` 32 → `None`; caminho ausente → `None`.
  - `test_save_is_atomic` (Review Focus 1): depois de `save_model`, não sobra `*.tmp` na pasta.
- [ ] **Step 2: Rodar.** Esperado: FAIL.
- [ ] **Step 3: Implementar.** `torch.set_num_threads(1)` **não** é usado aqui, para o treino ser rápido.
- [ ] **Step 4: Rodar.** Esperado: PASS.
- [ ] **Step 5: Medir o tempo.** `uv run python -c "from gerar_dados_falsos import generate_fake_rows; from neuronio import train_model; m,meta=train_model(generate_fake_rows(),[]); print(meta['seconds'], meta['train_accuracy'])"`. Esperado: menos de 15 s. Se passar disso, registrar o valor e avisar antes de seguir.
- [ ] **Step 6: Commit** (pendente): `feat(sumo): group MLP with Muon+cosine training`

---

### Task 5: Predição + decisão da IA (sem trapaça)

**Files:**
- Modify: `sumo/neuronio.py`, `sumo/tests/test_neuronio.py`

**Interfaces:**
- Consumes: `MLP`, `encode`, `regras.counter`, `regras.ACTIONS`
- Produces:
  - `predict_proba(model: MLP | None, ctx: RoundContext) -> list[float]`: softmax dos logits sob `torch.no_grad()`; com `model is None`, devolve `[1/3]*3`.
  - `choose_ai_action(model: MLP | None, ctx: RoundContext, rng: random.Random) -> tuple[str, list[float]]`: sorteia `predicted` com `rng.choices(ACTIONS, weights=p)` e retorna `(counter(predicted), p)`. As probabilidades vão para o HUD.

- [ ] **Step 1: Testes que falham**
  - `test_uniform_without_model`: `predict_proba(None, ctx) == [1/3]*3`.
  - `test_counter_of_confident_prediction`: um modelo treinado no padrão "repete" (pode reutilizar o helper da Task 4 com `steps=225`), com contexto em que `player_last[0] == "agarrao"` → em 200 sorteios com `random.Random(0)`, `"ataque"` aparece mais de 150 vezes.
  - `test_ai_decision_cannot_see_current_action`: `inspect.signature(choose_ai_action).parameters` é exatamente `["model", "ctx", "rng"]`, e `RoundContext` não tem campo com a ação do round atual (`{f.name for f in fields(RoundContext)} == {"player_last", "ai_last", "player_lives", "ai_lives"}`).
- [ ] **Step 2: Rodar.** Esperado: FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar a suíte inteira.** Esperado: PASS.
- [ ] **Step 5: Commit** (pendente): `feat(sumo): AI decision from predicted player move`

---

### Task 6: Interface pygame

**Files:**
- Create: `sumo/jogo.py`

**Interfaces:**
- Consumes: tudo acima. `dados_io.{ROUNDS_CSV, FAKE_CSV, MODEL_PATH, STATE_JSON, append_round, read_rounds, load_state, save_state}`, `gerar_dados_falsos.ensure_fake_data`, `neuronio.{train_model, save_model, load_model, choose_ai_action}`, `regras.Match`
- Produces: `main(argv: list[str] | None = None) -> int`; flag `--smoke` (roda cerca de 60 frames com `SDL_VIDEODRIVER=dummy`, simula 1 partida inteira por chamadas diretas e encerra com código 0).

**Estados** (`enum`): `MENU`, `CHOOSING`, `REVEAL`, `MATCH_OVER`, `TRAINING`.
- Janela 960×540, 60 FPS, título "Sumô × Neurônio". Dois sumôs desenhados com `pygame.draw` (corpo em círculo grande, mawashi em retângulo, cabeça com coque). Jogador à esquerda em azul, IA à direita em vermelho. Linha do dohyō no chão.
- **Início:** `ensure_fake_data()`; `load_model(MODEL_PATH)`; se o resultado for `None`, vai para `TRAINING` com motivo "Primeiro treino (dados falsos)".
- **MENU:** botões "Lutar", "Treinar neurônio" e "Sair". Rodapé mostra "Rounds reais coletados: N · Treino automático em K partidas".
- **CHOOSING:** ao entrar no estado, **antes de aceitar input**, chama `choose_ai_action(model, match.context(), rng)` e guarda `(ai_action, probs)` (garantia anti-trapaça). Botões e teclas: A = Ataque, S = Defesa, D = Agarrão; ESC volta ao menu (partida abandonada, contador intacto). HUD com 10 corações por lado e "Round N".
- **REVEAL** (~1,2 s): `row = match.play(player, ai_action)`; `append_round(ROUNDS_CSV, row)` imediatamente. Mostra as duas ações sobre os sumôs, o texto do resultado ("Você venceu o round!" / "O neurônio venceu o round!" / "Empate!") e uma linha pequena "Neurônio previu: ataque 52% · defesa 30% · agarrão 18%". Todo input é ignorado neste estado (Review Focus 5). Ao fim, vai para `MATCH_OVER` se `match.finished`; caso contrário, volta a `CHOOSING`.
- **MATCH_OVER:** incrementa `partidas_desde_treino` e salva o estado. Mostra "Vitória!" ou "Derrota!" e os botões "Jogar de novo" e "Menu". Se o contador chegar a 10, entra direto em `TRAINING` com o motivo "Treino automático (10 partidas)".
- **TRAINING:** dispara `threading.Thread(daemon=True)`, que executa `train_model(read_rounds(FAKE_CSV), read_rounds(ROUNDS_CSV), progress=...)` e depois `save_model`. O `progress` escreve em atributos protegidos por `threading.Lock`. A tela mostra o motivo, uma barra `step/225`, "Aguarde o neurônio ficar pronto…" e, ao terminar, "Pronto! Acurácia no treino: X% · rounds reais: N · usou falsos: sim/não" com o botão "Lutar". Ao concluir, troca `self.model`, zera o contador, salva o estado e imprime `meta` no console. Uma exceção na thread aparece na tela ("Falha no treino: …"), e o jogo continua com o modelo anterior.
- **Fechar a janela:** `pygame.quit()` + `return 0`. A thread daemon não bloqueia a saída, e o save atômico protege o `modelo.pt` (Review Focus 1).

- [ ] **Step 1: Implementar `jogo.py` conforme acima.** A lógica de transição de estados fica em métodos pequenos (`on_choose(action)`, `finish_reveal()`, `on_match_over()`, `start_training(reason)`), para que o `--smoke` consiga chamá-los diretamente.
- [ ] **Step 2: Smoke test.** `uv run python jogo.py --smoke` com `SUMO_DATA_DIR` apontando para uma pasta no scratchpad. Esperado: código de saída 0, `rounds.csv` temporário com pelo menos 10 linhas e console mostrando o `meta` do primeiro treino.
- [ ] **Step 3: Teste do duplo input.** No `--smoke`, chamar `on_choose("ataque")` duas vezes seguidas dentro do mesmo round. O CSV deve ganhar só 1 linha (a segunda chamada é ignorada porque o estado já é `REVEAL`).
- [ ] **Step 4: Teste manual pelo Leandro.** `uv run python jogo.py`: jogar 1 partida, apertar Treinar e conferir que a barra avança e o jogo volta.
- [ ] **Step 5: Commit** (pendente): `feat(sumo): pygame game loop with training screen`

---

### Task 7: Distribuição (`jogar.bat`, README, zip)

**Files:**
- Create: `sumo/jogar.bat`, `sumo/README.md`, `empacotar.ps1`

**Interfaces:**
- Consumes: `sumo/` inteiro e `desafio2/desafio2_Leandro_Marques_Miguel_Cesario.py`
- Produces: `dist/sumo-neuronio.zip`, contendo `sumo-neuronio/sumo/...` e `sumo-neuronio/desafio2/desafio2_Leandro_Marques_Miguel_Cesario.py`, sem `dados/`, `.venv/`, `__pycache__/` e `tests/`.

- [ ] **Step 1: `jogar.bat`**

```bat
@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Instalando uv (so na primeira vez)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
echo Preparando o jogo (a primeira vez baixa ~200 MB)...
uv run --project "%~dp0." python "%~dp0jogo.py"
if errorlevel 1 pause
```

- [ ] **Step 2: `README.md`.** Explicar como jogar (teclas A/S/D), onde os dados ficam (`sumo/dados/rounds.csv`), como juntar dados de outro computador (copiar as linhas do `rounds.csv`, já que o `partida_id` é único) e que a primeira execução precisa de internet. Incluir a seção "Rede neural": mesma MLP/ativação/init do desafio 2 e receita do desafio 3, com a tabela de adaptações do spec.
- [ ] **Step 3: `empacotar.ps1`.** Copia os arquivos para `dist/staging/sumo-neuronio/`, gera o zip com `Compress-Archive -Force` e remove a pasta de staging.
- [ ] **Step 4: Smoke test do zip** (Review Focus 4). Extrair `dist/sumo-neuronio.zip` numa pasta `teste sumô` (com espaço e acento) **dentro do scratchpad** e rodar `cmd /c "<pasta>\sumo-neuronio\sumo\jogar.bat"` com `SUMO_SMOKE=1`, que faz o `.bat` repassar `--smoke`. Para isso, acrescentar ao `.bat` `if defined SUMO_SMOKE set ARGS=--smoke` e passar `%ARGS%`. Esperado: código de saída 0 e `dados/rounds.csv` criado dentro da cópia extraída.
- [ ] **Step 5: Commit** (pendente): `feat(sumo): packaging for friends`

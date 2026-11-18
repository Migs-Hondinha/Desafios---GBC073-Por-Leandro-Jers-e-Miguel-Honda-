# Sumô × Neurônio: design

Data: 2026-11-18. Autores: Leandro e Miguel (GBC073).

## Intenção

No dataset Tekken, a previsão do próximo golpe era contaminada por
movimentação, distância e outros fatores do jogo. Queremos isolar a pergunta:
**com os 2–3 golpes anteriores, a rede consegue prever o próximo golpe do
jogador?** Para isso, criamos um jogo mínimo, no formato de pedra-papel-tesoura,
que gera os próprios dados. Jogamos contra a rede, ela coleta os rounds e
retreina, e o jogo pode ser enviado em `.zip` a um amigo.

## Objetivos

1. Jogo de dois sumôs parados, frente a frente. Um é o jogador e o outro é a IA.
2. Ações: **ataque**, **defesa** e **agarrão**. Ataque vence agarrão, agarrão
   vence defesa e defesa vence ataque. Ações iguais empatam e ninguém perde vida.
3. Jogo por rounds. O jogo espera a escolha do jogador, revela as duas ações e
   aplica o resultado. Quem perde o round perde 1 de 10 vidas. Quem chega a
   0 vidas perde a partida.
4. A IA prevê a próxima ação do jogador com o MLP do grupo e joga a ação que
   vence a previsão.
5. Funciona sem histórico. No 1º round, e no 2º e 3º com histórico parcial, os
   golpes inexistentes entram como categoria "nenhum". Assim, a rede aprende
   com qual golpe o jogador costuma abrir.
6. Dados falsos permitem o primeiro treino antes de existir qualquer dado real.
7. Cada round jogado é salvo em CSV no disco local.
8. Um botão "Treinar neurônio" mostra uma tela de espera até o modelo ficar
   pronto e depois volta para a luta.
9. A cada 10 partidas completas, o treino roda sozinho com a mesma tela de espera.
10. O `.zip` roda em outro Windows com duplo clique em `jogar.bat`. Na primeira
    execução, o `uv` baixa Python e as dependências, o que exige internet uma vez.

## Não-objetivos

- Movimentação, distância, animação elaborada, som ou multiplayer em rede.
- Sincronizar dados entre computadores. Cada instalação tem seus próprios dados.
  Para juntá-los, copiamos os CSVs manualmente.
- Protocolo científico com busca de hiperparâmetros (isso foi o desafio 3).
- Executável `.exe` ou build para Linux/macOS (o `.bat` é só Windows; em
  Linux/macOS, `uv run` funciona manualmente).

## Anti-objetivos

- Uma IA que **trapaceia**, lendo a escolha atual do jogador. A previsão só pode
  usar rounds anteriores. A ação do jogador é capturada antes, mas a IA decide
  sem acesso a ela, e o código precisa garantir isso.
- Uma rede diferente da do grupo sem aviso explícito.
- Perder dados reais por sobrescrita ou crash. O CSV é aberto em modo append
  e cada round é gravado imediatamente.

## Restrições

- PyTorch, nunca Keras/TensorFlow (regra da disciplina).
- Reaproveitar `desafio2/desafio2_Leandro_Marques_Miguel_Cesario.py`
  (`ativacao`: `0.5·tanh`; `inicializar`: ortogonal com ganhos por camada).
- Reaproveitar a receita vencedora do desafio 3: **Muon (lr 0,1) nas matrizes
  ocultas + AdamW (lr 0,01) nos vieses e na saída, weight decay 0,01, agenda
  cosseno, 225 passos, lote 128**.
- Distribuição com `uv` e `pyproject.toml`, usando o índice CPU do PyTorch para
  reduzir o download.
- Código com identificadores em inglês e textos de interface em português.

## Abordagens consideradas (distribuição)

| Abordagem | Trade-off | Decisão |
|---|---|---|
| 1. Zip pequeno + `jogar.bat`, com `uv` baixando Python/torch na 1ª execução | Zip de ~1 MB; exige internet uma vez; código idêntico ao do grupo | **Escolhida por Leandro** |
| 2. `.exe` via PyInstaller | Offline, mas zip de ~250 MB, só Windows e build frágil com torch | Descartada |
| 3. Reescrever rede em NumPy/JS | Leve, mas não é mais o código/receita do grupo | Descartada |

Interface: **pygame**, por ser leve, instalável pelo `uv` e sem servidor web.
A alternativa seria uma página HTML servida por Python local, que exige
servidor e navegador; pygame é mais simples (decisão do Claude).

## Adaptações da rede (avisadas a Leandro)

| Item | Desafio 3 | Sumô | Por quê |
|---|---|---|---|
| Entradas | 10 colunas Tekken | 26 (abaixo) | Sem distância nem HP contínuo; histórico explícito |
| Largura | 256 | **64** | Poucas centenas de rounds; 256 decora o dataset |
| Profundidade | 4 camadas | 4 camadas | Mantida |
| Ativação/init | desafio 2 | desafio 2 | Mantida |
| Otimizador/agenda | Muon 0,1 + AdamW 0,01, cosseno, wd 0,01 | idem | Mantida |
| Passos/lote | 225 / 128 | 225 / min(128, n) | Lote limitado pelo tamanho do dataset |
| Normalização | média/desvio do treino | nenhuma (one-hot e frações em [0,1]) | Entradas já limitadas |

### Entradas (26 valores)

- Últimas 3 ações do **jogador**: 3 blocos one-hot de 4 posições
  `[ataque, defesa, agarrão, nenhum]`, sendo t-1, t-2 e t-3 (12 valores).
- Últimas 3 ações da **IA**: mesmo formato (12 valores). O jogador reage ao que
  a IA fez.
- `vidas_jogador / 10` e `vidas_ia / 10` (2 valores).

Saída: 3 logits (0 = ataque, 1 = defesa, 2 = agarrão), com `cross_entropy`.
A codificação de classes segue a ordem `atk, def, grb` das colunas do Tekken.

### Decisão da IA (decisão do Claude)

A rede dá probabilidades `p` para a próxima ação do jogador. A IA **sorteia** a
previsão segundo `p`, em vez de sempre pegar o argmax, e joga o golpe que vence
o golpe sorteado. Com argmax, a IA ficaria determinística e fácil de explorar
quando o jogador percebesse o padrão. Sem modelo treinado, usa `p` uniforme.

## Dados

### Dataset falso (`gerar_dados_falsos.py`)

Simula jogadores sintéticos, cada um com uma mistura de vieses, para que exista
padrão aprendível:
- abertura preferida, por exemplo 50% ataque no 1º round;
- tendência a repetir o golpe anterior;
- ciclo (ataque→agarrão→defesa…);
- contra-ataque ao último golpe da IA;
- ruído uniforme.

São cerca de 300 partidas simuladas contra uma IA aleatória, com o mesmo formato
do CSV real. O arquivo vai para `dados/falsos.csv` e é regenerável com semente
fixa.

### Dados reais (`dados/rounds.csv`)

Há uma linha por round, gravada em append no momento do resultado:

`timestamp, partida_id, round, jogador_t1, jogador_t2, jogador_t3, ia_t1, ia_t2,
ia_t3, vidas_jogador, vidas_ia, acao_jogador, acao_ia, resultado`

As ações são gravadas como texto (`ataque`/`defesa`/`agarrao`/`nenhum`), para
serem legíveis no Excel. `acao_jogador` é o alvo. Também guardamos
`dados/estado.json` com o contador de partidas completas desde o último treino.

### Mistura no treino (decisão do Claude)

- Com menos de 200 rounds reais, o treino usa falsos + reais, **ponderados
  50/50**. Os lotes são amostrados com `WeightedRandomSampler`, de modo que os
  rounds reais, mesmo em minoria (~5 mil falsos contra dezenas de reais), formem
  metade de cada lote. Assim, a IA começa a se adaptar ao jogador já no primeiro
  treino automático. Isso traz risco de decorar os poucos rounds reais, que
  aceitamos para esse estágio.
- Com 200 ou mais rounds reais, usa só reais. Os falsos serviram apenas para
  dar partida à rede.
- **Todo treino recomeça do zero**: inicialização do desafio 2 seguida dos 225
  passos Muon+cosseno completos, como na receita validada. Não fazemos ajuste
  fino sobre o `modelo.pt` anterior. Assim, o estado do otimizador não precisa
  ser salvo, e os falsos deixam de influenciar a rede quando saem do dataset.
  Semente fixa (0) para reprodutibilidade.

Não há divisão treino/validação no jogo. O console mostra a acurácia no próprio
dataset e a distribuição das classes, como diagnóstico, sem afirmar
generalização.

### Modelo

`dados/modelo.pt` guarda o `state_dict` e os metadados (data, nº de exemplos,
acurácia). O jogo carrega esse arquivo ao iniciar. Se ele não existir, treina
nos falsos antes da primeira luta, com tela de espera.

## Fluxo do jogo

```
[Menu] → Lutar | Treinar neurônio | Sair
[Luta]  HUD: vidas (10 corações cada), nº da partida, "treino automático em N partidas"
        Botões/teclas: [A] Ataque  [S] Defesa  [D] Agarrão
        → IA decide (antes de ler a escolha do jogador) → revela as duas → resultado → próximo round
[Fim de partida] Vitória/Derrota → Jogar de novo | Menu
        se partidas_desde_treino == 10 → [Treinando...] → zera contador
[Treinando...] barra de progresso por passo; o treino roda em thread separada e a janela continua respondendo
```

## Estrutura de arquivos

```
entrega-gbc073/
  desafio2/desafio2_Leandro_Marques_Miguel_Cesario.py   (reaproveitado)
  sumo/
    pyproject.toml        # pygame, torch (índice CPU), python>=3.11
    jogar.bat             # instala uv se faltar → uv run jogo.py
    README.md
    jogo.py               # pygame: telas, input, HUD, sprites desenhados
    regras.py             # ações, quem vence, estado da partida (sem torch, testável)
    neuronio.py           # MLP + features + treino (Muon/AdamW, cosseno) + predição
    dados_io.py           # append CSV, leitura, estado.json
    gerar_dados_falsos.py
    tests/                # regras, features (incl. histórico vazio), treino roda e reduz perda
    dados/                # criado em runtime; ignorado pelo git
  empacotar.ps1           # gera sumo.zip com sumo/ + desafio2/ (sem dados/, .venv)
```

`neuronio.py` importa `desafio2` pelo caminho relativo `../desafio2`, como o
desafio 3 faz. Por isso, o zip leva as duas pastas.

## Detalhes fechados (decisões do Claude)

- O botão "Treinar neurônio" também zera o contador de 10 partidas.
- Uma partida abandonada (volta ao menu no meio) não conta para o contador,
  mas os rounds dela continuam no CSV.
- As vidas nas features são as do **início** do round.
- O Muon cobre todas as matrizes ocultas, incluindo a 1ª camada (26→64), como
  em `treino.py`. O AdamW cobre os vieses e a camada de saída.
- `partida_id` é um UUID, para não colidir quando juntarmos CSVs de máquinas
  diferentes.
- A IA escolhe a ação no início do round, antes de qualquer input do jogador.
  Um teste garante que a função de decisão não recebe a ação atual.
- `torch>=2.9` (por causa de `torch.optim.Muon`) fica fixado no `pyproject.toml`.

## Deixado ao implementador

- Visual exato dos sumôs, cores e layout, desde que leia bem em 960×540.
- Texto da tela de espera e formato da barra de progresso.
- Detalhes do `jogar.bat`: mensagens e tratamento de falta de internet.

## Verificação

- Testes unitários de regras, features (histórico vazio e parcial) e treino
  (perda cai num padrão sintético simples, como "sempre repete").
- Treino com 225 passos e largura 64 em CPU deve levar poucos segundos. Vamos
  medir.
- Smoke test do zip: extrair numa pasta limpa e rodar `jogar.bat`.

# Sumô × Neurônio

Dois sumôs ficam frente a frente. Um deles é você; o outro é o neurônio do grupo
(MLP dos desafios 2 e 3). O neurônio usa os seus **3 golpes anteriores** para
prever o próximo e joga o golpe que vence essa previsão.

- **Ataque** vence **agarrão**
- **Agarrão** vence **defesa**
- **Defesa** vence **ataque**
- Golpes iguais empatam, e ninguém perde vida.

Cada lado começa com 10 vidas. Quem perde todas perde a partida.

## Como jogar (Windows)

1. Descompacte o `.zip`.
2. Dê dois cliques em `sumo\jogar.bat`.
3. Na primeira execução, o `uv` é instalado e baixa Python, PyTorch (CPU) e
   pygame, cerca de 200 MB. Isso **precisa de internet uma vez**. Depois, o jogo
   abre direto.

Teclas: **A** = Ataque, **S** = Defesa, **D** = Agarrão, **ESC** = voltar ao menu.
Também é possível clicar nos botões.

Em Linux/macOS, com o [uv](https://docs.astral.sh/uv/) instalado: `cd sumo && uv run python jogo.py`.

## Treino do neurônio

- No primeiro uso não há dados reais. O jogo então gera `dados/falsos.csv`
  (jogadores sintéticos com hábitos: abertura preferida, repetir golpe, ciclo,
  contra-atacar o último golpe da IA, ruído) e treina com ele.
- Cada round jogado é gravado na hora em `dados/rounds.csv`.
- O botão **Treinar neurônio** treina com os dados atuais e mostra a barra de
  progresso. A luta continua quando o treino termina.
- A cada **10 partidas completas**, o treino roda sozinho. Partidas abandonadas
  com ESC não contam, mas os rounds delas ficam salvos.
- Com menos de 200 rounds reais, o treino mistura falsos e reais com peso 50/50.
  A partir de 200 rounds reais, usa só os reais.
- Cada treino recomeça do zero e o resultado é salvo em `dados/modelo.pt`.

### Juntar dados de vários computadores

Cada partida tem um `partida_id` único. Para juntar os dados, copie as linhas
(sem o cabeçalho) do `rounds.csv` de um computador para o final do `rounds.csv`
do outro.

## A rede neural

O jogo usa a mesma rede dos trabalhos anteriores, com as adaptações abaixo:

| Item | Desafio 3 | Sumô | Motivo |
|---|---|---|---|
| Ativação / inicialização | `0.5·tanh` / ortogonal (desafio 2) | igual | importadas de `../desafio2` |
| Camadas | 4 `Linear` | 4 `Linear` | igual |
| Otimizador | Muon 0,1 + AdamW 0,01, wd 0,01, cosseno, 225 passos, lote 128 | igual | receita vencedora do desafio 3 |
| Entradas | 10 colunas Tekken | 26 | sem distância/HP contínuo; histórico explícito |
| Largura | 256 | 64 | poucas centenas de rounds; 256 decoraria os dados |
| Normalização | média/desvio | nenhuma | entradas já são 0/1 ou frações |

As 26 entradas são os últimos 3 golpes do jogador e os últimos 3 da IA, cada um
em one-hot `[ataque, defesa, agarrão, nenhum]`, mais vidas/10 de cada lado.
"Nenhum" preenche o início da partida, e assim a rede aprende **com que golpe o
jogador costuma abrir**.

A IA **sorteia** a previsão de acordo com as probabilidades da rede, em vez de
sempre pegar a mais provável, e joga o golpe que vence o golpe sorteado. Ela
decide antes de ler o seu input, então não há trapaça.

## Desenvolvimento

```powershell
cd sumo
uv run python -m unittest -v      # testes
uv run python jogo.py --smoke     # partida automática sem janela
uv run python gerar_dados_falsos.py
cd ..; ./empacotar.ps1            # gera dist/sumo-neuronio.zip
```

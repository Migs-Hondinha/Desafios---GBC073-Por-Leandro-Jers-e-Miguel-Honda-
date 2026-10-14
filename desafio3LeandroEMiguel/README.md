# Trabalho 3 — treinamento configurável

Esta versão faz **um treinamento instrumentado**, com cinco opções de
otimizador e quatro agendas. Você escolhe taxa, agenda e semente por argumentos.
`protocolo.py` executa as etapas do enunciado, e `relatorio.py` gera os PDFs
com os resultados. Os experimentos oficiais ainda precisam ser executados.
Veja `PASSO_A_PASSO.md` para a sequência completa de comandos.

## Preparar após baixar o repositório

Pré-requisitos: Python compatível com PyTorch e `uv`. No PowerShell, partindo
da raiz do repositório clonado ou extraído do ZIP:

```powershell
cd desafio3LeandroEMiguel
uv venv --python 3.13 .venv
uv pip install --python ./.venv/Scripts/python.exe -r requirements.txt
```

Execute com o ambiente recém-criado:

```powershell
./.venv/Scripts/python.exe -X utf8 treino.py
```

Os caminhos são relativos à pasta `desafio3LeandroEMiguel`; não há dependência do computador
do autor ou do Codex. `-X utf8` permite imprimir acentos corretamente.

No Linux/macOS, substitua `./.venv/Scripts/python.exe` por `./.venv/bin/python`
tanto na instalação quanto na execução. O dataset e o inicializador necessários
estão versionados em `desafio2/`, ao lado de `desafio3LeandroEMiguel/`. Baixe o repositório inteiro.

O comando padrão usa SGD com momento 0,9, taxa 0,05, lote 128, semente 0,
100 passos e taxa constante. A rede tem quatro camadas lineares, com largura
256 nas três ocultas. Esses valores se baseiam no harness anterior, mas o
orçamento de 100 passos é apenas um exemplo inicial, não a baseline oficial.

Outra execução, mudando o otimizador e a taxa:

```powershell
./.venv/Scripts/python.exe -X utf8 treino.py --otimizador adamw --lr 0.001
```

Exemplo com todas as escolhas principais explícitas (cole em uma linha):

```powershell
./.venv/Scripts/python.exe -X utf8 treino.py --otimizador muon --lr 0.02 --lr-adamw 0.0003 --agenda warmup_cosseno --warmup-fracao 0.1 --semente 7 --passos 100 --weight-decay 0.01
```

| Opção | O que você escolhe |
|---|---|
| `--otimizador` | `sgd`, `nesterov`, `adam`, `adamw` ou `muon` |
| `--lr` | Taxa base; no warmup, é o pico atingido no fim do aquecimento |
| `--lr-adamw` | Taxa da parte AdamW **apenas quando o otimizador é Muon** |
| `--agenda` | `constante`, `cosseno`, `warmup_cosseno` ou `degraus` |
| `--warmup-fracao` | Fração dos passos usada para aquecimento; padrão 0,1 |
| `--semente` | Inteiro não negativo que controla inicialização e lotes |
| `--passos` | Quantidade de atualizações; padrão 100 |
| `--weight-decay` | Intensidade do decaimento de pesos; padrão zero |

Quando `--lr` é omitida, usamos 0,05 para SGD/Nesterov, 0,001 para Adam/AdamW
e 0,02 para Muon. São pontos de partida, não taxas já otimizadas. Você pode
substituí-los por qualquer taxa finita positiva. No Muon, o AdamW auxiliar
usa 0,0003 se `--lr-adamw` não for informada. No protocolo final, essa taxa
auxiliar deve ser a melhor encontrada na busca do AdamW.

`--semente` muda a inicialização e os lotes. A divisão dos dados continua fixa
para comparar treinos sobre os mesmos conjuntos.

Isso demonstra a configuração; duas execuções isoladas não constituem uma
comparação justa ou comprovam qual otimizador é melhor. O `weight_decay`
padrão é zero para todos. Com zero, não estamos explorando a regularização
desacoplada que diferencia AdamW. A escolha será fixada e declarada no protocolo.

```powershell
./.venv/Scripts/python.exe -X utf8 treino.py --help
./.venv/Scripts/python.exe -X utf8 -m unittest -v
```

## Leia o código nesta ordem

### 1. `main`: configura a execução

Está no fim de `treino.py`. Lê opções como `--lr`, `--passos` e `--semente`,
valida valores e chama `train`. O bloco `if __name__ == "__main__"` executa
essa função quando você roda o arquivo, sem começar um treino ao importá-lo.

### 2. `load_data`: prepara os exemplos

Lê `../desafio2/dataset_tekken.csv`. As dez colunas de entrada formam `x`;
`y_target` forma `y`, o código da classe que queremos prever. Não atribuímos
nomes de ações às classes, pois seu significado ainda precisa ser confirmado.

A separação é estratificada: cada classe é dividida entre os conjuntos.
Preservamos os 20% de teste do adaptador anterior (semente 42) e separamos
20% dos exemplos restantes para validação (semente 2026):

| Conjunto | Exemplos neste CSV | Uso |
|---|---:|---|
| Treino | 9.600 | Calcular gradientes e atualizar pesos |
| Validação | 2.400 | Acompanhar resultados e escolher configurações |
| Teste | 3.000 | Reservado; esta versão não calcula métricas nele |

Em cada atributo, subtraímos a média do treino e dividimos pelo desvio padrão
do treino. A validação usa essas mesmas estatísticas. Isso evita usar a
distribuição da validação para preparar o treinamento.

Os dados são separados por linhas. Sem identificadores de partida, não é
possível garantir independência entre partidas. Além disso, o teste já foi
utilizado no trabalho anterior; preservá-lo evita reutilizá-lo nesta seleção,
mas não transforma esse conjunto em um teste historicamente inédito.

### 3. `MLP`: define a rede

O fluxo padrão neste CSV é:

```text
10 entradas → 256 → 256 → 256 → 3 saídas
```

Cada `Linear` calcula uma transformação com pesos e vieses. Nas camadas ocultas,
aplicamos `0.5 * tanh(x)`. A última produz pontuações chamadas logits.
`CrossEntropyLoss` recebe esses logits e a classe correta para calcular o erro;
não devemos aplicar tanh ou softmax na saída antes dessa função.

As funções `ativacao` e `inicializar` são importadas do arquivo de vocês no
trabalho 2. Portanto, mantenha as pastas `desafio2` e `desafio3LeandroEMiguel` lado a lado.
`treino.py` não depende do harness do professor nem de torchvision.

### 4. `make_optimizers`: escolhe como atualizar os pesos

SGD, Nesterov, Adam e AdamW usam um otimizador para todos os parâmetros.
Muon usa dois otimizadores complementares:

- **Muon:** matrizes de pesos das camadas ocultas, com momento 0,95, Nesterov
  e cinco iterações de Newton–Schulz para ortogonalizar a atualização.
- **AdamW:** todos os vieses e os pesos da camada de saída.

Cada parâmetro pertence a exatamente um dos dois. A implementação é a nativa
`torch.optim.Muon`; o ambiente local tem PyTorch 2.14. O ajuste de escala
`original` é explícito: Muon também escala a atualização conforme o formato
da matriz. A coluna `lr` registra a taxa do grupo após a agenda; não representa
sozinha o deslocamento dos pesos, que é medido pela razão de atualização.

### 4.1. `schedule_factor` e `apply_schedule`: controlam as taxas

Para facilitar a leitura, calculamos um multiplicador diretamente a cada
passo, sem manter um objeto scheduler separado. `apply_schedule` define
`taxa_do_passo = taxa_base * multiplicador` **antes** de atualizar os pesos.

| Agenda | Comportamento |
|---|---|
| `constante` | Multiplicador 1 durante todo o treino |
| `cosseno` | Começa na taxa base e decresce suavemente até zero |
| `warmup_cosseno` | Aumenta linearmente até a taxa base e depois decresce em cosseno até zero |
| `degraus` | Reduz a taxa para 10% após metade dos passos e para 1% após 75% |

No warmup, usamos aproximadamente `passos * warmup_fracao` passos, limitado
a pelo menos um e no máximo `passos - 1`. A primeira taxa é a taxa base
dividida pelo número de passos de warmup. Essa agenda exige pelo menos três
passos. Cosseno com apenas um passo usa a taxa base. Nas agendas cosseno
com mais passos, a última atualização tem taxa zero por definição.

Para Muon + AdamW, aplicamos **o mesmo multiplicador** às duas taxas base.
Os logs registram ambas: `lr` para Muon e `lr_adamw` para seu AdamW auxiliar.
Quando o otimizador é AdamW sozinho, sua taxa está em `lr`.

### 5. `train`: repete o aprendizado

Um passo processa um lote de 128 exemplos:

```python
model.zero_grad(set_to_none=True)  # Clear every parameter's previous gradient.
loss = nn.functional.cross_entropy(model(x), y)  # Predict and measure error.
loss.backward()  # Compute the gradient for each trainable parameter.
rates = apply_schedule(optimizers, args, step)  # Set this step's learning rates.
for optimizer in optimizers.values():
    optimizer.step()  # Each optimizer updates only its own parameters.
```

`DataLoader` embaralha os exemplos. Quando esgotamos os lotes, começamos uma
nova passagem. `drop_last=True` descarta o lote incompleto dessa passagem,
mantendo o tamanho de cada atualização igual. O treino termina pelo número
de passos, não pelo número de épocas.

A semente controla a inicialização e o embaralhamento. As sementes de divisão
dos dados permanecem fixas entre execuções. Usamos CPU, uma thread e algoritmos
determinísticos para facilitar a reprodução nesta primeira versão.

### 6. `vector_norm` e `evaluate`: observam o treino

`vector_norm` calcula a raiz da soma dos quadrados: uma medida do tamanho de
um conjunto de números. Registramos globalmente e nas camadas 0, 2 e 3:

- Norma do gradiente antes da atualização. Não há clipping nesta versão.
- Norma dos parâmetros antes da atualização, incluindo pesos e vieses.
- Razão entre o tamanho da atualização e o tamanho anterior dos parâmetros.

Também registramos perda do lote, perda e acurácia de validação e taxa usada.
O intervalo padrão é 20 passos, com registros adicionais no primeiro e último.
A perda de treino é calculada **antes** da atualização; a validação é calculada
**depois**. Uma perda por lote pode oscilar porque cada lote tem exemplos distintos.

`evaluate` usa `torch.no_grad()`: calcula as métricas sem gerar gradientes ou
atualizar a rede. A acurácia é a fração de classes previstas corretamente.

### 7. `append_csv`: salva as observações

```text
desafio3LeandroEMiguel/
  treino.py
  test_treino.py
  README.md
  experimentos.csv
  logs/
    exemplo_sgd.csv
    exemplo_sgd.json
```

- `logs/*.csv`: evolução dos sinais ao longo dos passos.
- `logs/*.json`: configuração completa, índices da divisão, versões e hashes
  dos dados, do treino e do inicializador.
- `logs/*.pt`: pesos finais de cada execução finita, usados no teste final.
  Os exemplos anteriores à implementação de checkpoints não têm esse arquivo.
- `experimentos.csv`: uma linha por treino, com métrica final de validação,
  configuração, tempo e indicação de divergência.

Valores não finitos na perda, gradientes ou parâmetros interrompem a execução
e são sinalizados no resumo. Um nome já existente é recusado para não substituir
resultados. Sem `--nome`, um identificador com data/hora é criado automaticamente.

## Execução de exemplo já realizada

Com SGD, 100 passos e os parâmetros padrão, a execução `exemplo_sgd` terminou
com perda de validação **0,9909** e acurácia **51,04%**. É um teste funcional,
com uma única semente. Não é evidência de superioridade nem resultado final.
O conjunto de teste não foi avaliado.

## Próximas etapas

1. Entender este treino e confirmar a receita de referência do grupo.
2. Executar as etapas de `protocolo.py`, seguindo `PASSO_A_PASSO.md`.
3. Gerar os PDFs, revisar o diagnóstico e enviar a pasta ao repositório do grupo.

## Seleção e reprodução do protocolo

- Prefixo `p3_`: execuções oficiais deste protocolo. Exemplos anteriores ficam
  no CSV e nos logs, mas não participam da seleção.
- Baseline: SGD com momento 0,9, taxa 0,05, decay zero e taxa constante.
  É a referência proposta a partir do harness anterior; confirme que representa
  a receita do grupo. Quatro camadas e largura 256 são uma escolha fixa do estudo.
- Candidatas: decay 0,01 fixo. Portanto, comparamos receitas e declaramos essa
  diferença de regularização em relação à baseline.
- Busca: acurácia final de validação, desempate por menor perda. Grade inicial
  com cinco taxas, estendida automaticamente se a melhor cair na borda. Dez
  extensões sem sair da borda interrompem a busca para revisão, sem aceitar a borda.
- Agendas: dois melhores otimizadores e Muon se necessário. Três melhores
  combinações seguem para confirmação nas sementes 0, 1 e 2.
- Uma candidata que divergir em qualquer semente fica reportada, mas é excluída
  da seleção final. Não calculamos sua média usando só execuções bem-sucedidas.
- Ganho: diferenças de acurácia entre candidata e baseline na mesma semente.
  Reportamos média e desvio amostral desses três ganhos em pontos percentuais.
  Consideramos ganho consistente apenas se a média superar esse desvio. É uma
  regra descritiva explícita, não uma declaração de significância estatística.
- Teste final: checkpoint da semente 0 da melhor candidata consistente; se não
  houver ganho consistente, usa baseline semente 0. A semente não é escolhida
  por acurácia. Resultado é salvo e reutilizado; não se avalia novamente.
- O código bloqueia mudanças de orçamento, dados e implementação no meio do
  estudo. Execuções concluídas são reutilizadas. Uma execução interrompida
  deixa arquivos preservados e exige revisão em vez de sobrescrita silenciosa.
- Depois do teste final, novos treinos desse protocolo são bloqueados. Isso
  não impede alguém de executar `treino.py` manualmente: não use o teste para
  orientar novas escolhas.

`verificar_fluxo.py` é uma verificação técnica com dados sintéticos, 40 passos
e rede pequena. Suas saídas ficam em `tmp/qa_*/`, são ignoradas na entrega e
não são resultados do Tekken. Não execute para cumprir o protocolo científico.

# Do treinamento à entrega

Use o PowerShell. Execute os comandos em ordem, esperando cada um terminar.
Não execute duas etapas simultaneamente. O roteiro usa 225 passos, lote 128,
quatro camadas lineares e largura 256 em todas as comparações. São escolhas
do estudo, não números impostos pelo professor. No CSV atual, são três épocas.

Antes de começar, confirme a baseline: SGD com momento 0,9, lr 0,05, agenda
constante, inicializador do trabalho 2 e decay zero. É a referência proposta
a partir do código anterior. Se a receita atual do grupo for outra, ajuste-a
antes da primeira execução. As candidatas usam decay 0,01, declarado no painel.

## 1. Abrir a pasta

```powershell
cd desafio3
uv venv --python 3.13 .venv
$pythonTreino = "./.venv/Scripts/python.exe"
uv pip install --python $pythonTreino -r requirements.txt
```

## 2. Conferir o programa

```powershell
& $pythonTreino -X utf8 -m unittest -v
```

Espere `OK`. Os comandos acima partem da raiz do repositório baixado e usam
apenas arquivos versionados. No Linux/macOS, use `./.venv/bin/python` e execute
os comandos Python sem o operador `&` do PowerShell; use `pythonTreino=./.venv/bin/python`
para definir a variável no Bash. `Get-Content arquivo` pode ser substituído
por abrir o arquivo em um editor, e `Start-Process` pelo visualizador de PDF.

## 3. Baseline - três sementes

```powershell
& $pythonTreino -X utf8 -u protocolo.py baseline
```

Executa as sementes 0, 1 e 2. Gera `baseline.json`, logs, checkpoints e linhas
no `experimentos.csv`. Não precisa rodar os três comandos manuais antigos.

## 4. Taxas - cinco ou mais valores por otimizador

```powershell
& $pythonTreino -X utf8 -u protocolo.py taxas
```

Ajusta SGD, Nesterov, Adam e AdamW; depois ajusta Muon usando a melhor taxa
do AdamW na parte auxiliar. Usa warmup + cosseno e semente 0. Amplia a grade
se a melhor taxa estiver na borda. Gera `melhores_taxas.json`.

```powershell
Get-Content melhores_taxas.json
```

Muon demora mais na CPU. Não feche o terminal durante uma execução.

## 5. Agendas - quatro por otimizador selecionado

```powershell
& $pythonTreino -X utf8 -u protocolo.py agendas
```

Seleciona os dois melhores otimizadores, inclui Muon se necessário e compara
constante, cosseno, warmup + cosseno e degraus. Usa a taxa ajustada de cada um.
Gera `finalistas.json` com três combinações.

## 6. Confirmação - três combinações, três sementes

```powershell
& $pythonTreino -X utf8 -u protocolo.py confirmacao
```

São nove execuções. Gera `resumo.json` com médias, desvios, ganhos pareados e
a receita selecionada para teste. Não declara ganho consistente se a diferença
não superar a variação pelo critério descritivo explicado no README.

```powershell
Get-Content resumo.json
```

## 7. Selecionar o caso instável para o painel

```powershell
& $pythonTreino -X utf8 -u protocolo.py diagnostico
```

Procura uma execução com valores não finitos ou perda registrada maior que
duas vezes a perda inicial. Caso necessário, executa ensaios extras de SGD com
taxas altas, identificados como diagnóstico e fora da seleção. Tudo é reportado.
Gera `instavel.json`. Se não observar instabilidade, para para revisão em vez
de inventar um caso. Um aumento de perda é um sinal a interpretar, não prova
isolada de explosão de gradientes.

## 8. Avaliar o teste uma única vez

```powershell
& $pythonTreino -X utf8 -u protocolo.py teste
```

Só faça isso após fechar as escolhas pela validação. Usa o checkpoint da
semente 0 da receita selecionada, sem novo treino. Gera `teste_final.json`.
Se repetir o comando, apenas mostra o resultado salvo. O teste já foi usado
no trabalho 2; essa limitação histórica é declarada no relatório.

## 9. Gerar os PDFs

ReportLab foi instalado pelo `requirements.txt`. Use o mesmo ambiente:

```powershell
& $pythonTreino -X utf8 relatorio.py
```

Gera `painel.pdf` (sinais alinhados, taxas, Muon × AdamW e resultados) e
`slide.pdf` (uma página). Eles usam os resultados reais do estudo. Os PDFs em
`tmp/qa_*/` são de verificação com dados sintéticos e NÃO devem ser entregues.

## 10. Abrir e revisar

```powershell
Start-Process .\painel.pdf
Start-Process .\slide.pdf
```

Leia as curvas e confira o diagnóstico. O texto automático descreve os sinais;
você precisa entendê-los e sustentar a interpretação na apresentação. Podemos
revisar essa parte juntos depois dos experimentos oficiais.

## 11. Conferir a entrega

```powershell
Get-Item treino.py, protocolo.py, relatorio.py, experimentos.csv, painel.pdf, slide.pdf
Get-ChildItem logs
```

Entregue a pasta `desafio3/`, incluindo código, CSV, logs, arquivos JSON do
protocolo, PDFs e instruções. Mantenha acessíveis no repositório o dataset e
o inicializador do trabalho 2, pois o código os importa. Não inclua `tmp/`,
`__pycache__/` nem ambientes virtuais. Os checkpoints `.pt` permitem avaliar
o teste e reproduzir a receita selecionada sem refazer o treinamento.

## 12. Enviar ao repositório do grupo

O repositório contém o código e as dependências locais; os resultados são
gerados pelos comandos acima. PDFs e outros artefatos gerados estão ignorados
no Git por decisão dos autores. Se a entrega formal exigir esses arquivos,
combine seu envio separadamente. O prazo do enunciado é 14/10/2026 às 12h.

## Se um comando parar

Não avance com erro. Copie a mensagem e revise a causa. Comandos repetidos
reutilizam execuções concluídas; arquivos de uma execução interrompida são
preservados e sinalizados, sem apagamento automático. Uma busca que permanece
na borda após dez extensões também para para revisão.

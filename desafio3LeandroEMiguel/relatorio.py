"""Generate painel.pdf and a single slide.pdf from completed protocol results.

Requires reportlab. No invented measurements or selection using test results.
"""
import argparse
import math
from pathlib import Path
import statistics
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen.canvas import Canvas

from protocolo import read_json, read_rows, best, valid

PALETTE = ['#17324D', '#007F86', '#E16A34', '#7258A6']


def paragraph(canvas, text, x, y, width=105, size=10, leading=14):
    canvas.setFillColor(colors.HexColor('#17324D'))
    canvas.setFont('Helvetica', size)
    for line in textwrap.wrap(text, width=width):
        canvas.drawString(x, y, line)
        y -= leading
    return y


def plot(canvas, series, x, y, width, height, title, log_y=False, log_x=False, x_end=None):
    """Draw vector lines; nonfinite values break a curve instead of hiding failure."""
    canvas.setFillColor(colors.HexColor('#17324D'))
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(x, y + height + 17, title)
    finite = [(a, b) for _, points in series for a, b in points
              if math.isfinite(a) and math.isfinite(b) and (not log_x or a > 0)]
    if not finite:
        canvas.drawString(x, y + height / 2, 'Sem medidas finitas disponíveis')
        return
    tx = lambda value: math.log10(max(value, 1e-12)) if log_x else value
    ty = lambda value: math.log10(max(value, 1e-12)) if log_y else value
    x_min, x_max = min(tx(a) for a, _ in finite), max(tx(a) for a, _ in finite)
    if x_end is not None:
        x_max = max(x_max, tx(x_end))
    y_min, y_max = min(ty(b) for _, b in finite), max(ty(b) for _, b in finite)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    px = lambda value: x + (tx(value) - x_min) / (x_max - x_min) * width
    py = lambda value: y + (ty(value) - y_min) / (y_max - y_min) * height
    canvas.setFont('Helvetica', 7)
    for index in range(3):
        tick = y_min + (y_max - y_min) * index / 2
        yy = y + height * index / 2
        canvas.setStrokeColor(colors.HexColor('#DDE4EB'))
        canvas.line(x, yy, x + width, yy)
        canvas.setFillColor(colors.HexColor('#536579'))
        canvas.drawRightString(x - 5, yy - 2, f'{10 ** tick if log_y else tick:.3g}')
    for index in range(3):
        tick = x_min + (x_max - x_min) * index / 2
        canvas.drawCentredString(x + index * width / 2, y - 11,
                                f'{10 ** tick if log_x else tick:.3g}')
    for index, (label, points) in enumerate(series):
        color = colors.HexColor(PALETTE[index % len(PALETTE)])
        canvas.setFillColor(color)
        canvas.setStrokeColor(color)
        canvas.setLineWidth(1.1)
        canvas.drawString(x + index * width / max(1, len(series)), y + height + 5, label)
        previous = None
        for a, b in points:
            if not math.isfinite(a) or not math.isfinite(b) or (log_x and a <= 0):
                previous = None
                continue
            current = (px(a), py(b))
            if previous is not None:
                canvas.line(*previous, *current)
            canvas.circle(*current, 1.25, fill=1, stroke=0)
            previous = current


def logs_for(directory, row):
    rows = read_rows(directory / 'logs' / (row['run'] + '.csv'))
    if not rows:
        raise RuntimeError(f"Logs ausentes: {row['run']}")
    return rows


def signal(rows, key):
    return [(float(row['step']), float(row[key])) for row in rows if row.get(key)]


def generate(directory):
    summary = read_json(directory / 'resumo.json')
    unstable = read_json(directory / 'instavel.json')
    final_test = read_json(directory / 'teste_final.json')
    all_rows = read_rows(directory / 'experimentos.csv')
    chosen = summary['best_candidate']
    baseline = summary['baseline']
    winner_row = chosen['runs'][0]
    baseline_row = baseline['runs'][0]
    # Compare Muon and AdamW at their tuned rates under the SAME warmup schedule.
    rates = read_json(directory / 'melhores_taxas.json')
    muon_best = best([row for row in all_rows if row['run'].startswith('p3_')
                      and row['optimizer'] == 'muon' and not row['run'].startswith('p3_diagnostico_')])
    diagnostic_rows = logs_for(directory, winner_row)
    gradient = [value for _, value in signal(diagnostic_rows, 'global_grad_norm') if math.isfinite(value)]
    val = signal(diagnostic_rows, 'validation_loss')
    observation = (f"Na semente 0 da melhor candidata, a perda de validação passou de {val[0][1]:.4f} "
                   f"para {val[-1][1]:.4f}. O gradiente global máximo foi {max(gradient):.3g}, "
                   f"e a mediana foi {statistics.median(gradient):.3g}. "
                   'Esses sinais descrevem a trajetória; não isolam a causa do resultado.')
    status = ('Ganho maior que o desvio dos ganhos pareados.' if chosen['exceeds_variation']
              else 'Não há ganho consistente pelo critério declarado; manter baseline.')
    width, height = landscape(A4)
    panel = Canvas(str(directory / 'painel.pdf'), pagesize=(width, height))
    panel.setTitle('Desafio 3 - sinais de treinamento e comparação')
    for label, row in [('Melhor candidata - semente 0', winner_row),
                       ('Baseline - semente 0', baseline_row),
                       ('Melhor execução com Muon na validação', muon_best),
                       ('Caso instável observado', unstable['run'])]:
        rows = logs_for(directory, row)
        panel.setFont('Helvetica-Bold', 17)
        panel.drawString(48, height - 32, label)
        paragraph(panel, f"{row['optimizer']} | lr={row['lr']} | {row['schedule']} | semente={row['seed']} "
                  f"| divergiu={row['diverged']}", 48, height - 50, size=9)
        paragraph(panel, row['run'], 48, height - 64, size=7)
        group_names = [key.removesuffix('_grad_norm') for key in rows[0] if key.endswith('_grad_norm')]
        plots = [('Perda (lote antes do passo; validação após)',
                  [('treino', signal(rows, 'train_loss')), ('validação', signal(rows, 'validation_loss'))], False),
                 ('Norma do gradiente antes da atualização (log)',
                  [(name, signal(rows, name + '_grad_norm')) for name in group_names], True),
                 ('Norma dos parâmetros antes da atualização (log)',
                  [(name, signal(rows, name + '_parameter_norm')) for name in group_names], True),
                 ('Taxa usada na atualização', [('principal', signal(rows, 'lr'))] +
                  ([('AdamW auxiliar', signal(rows, 'lr_adamw'))] if row['optimizer'] == 'muon' else []), False),
                 ('Razão de atualização (log; zeros limitados a 1e-12 para desenho)',
                  [(name, signal(rows, name + '_update_ratio')) for name in group_names], True)]
        for index, (title, series, logarithmic) in enumerate(plots):
            plot(panel, series, 82, height - 155 - index * 91, width - 125, 53,
                 title, log_y=logarithmic, x_end=float(row['requested_steps']))
        paragraph(panel, 'Eixo horizontal comum: passos. Camadas numeradas a partir de zero. Valores não finitos interrompem a curva.',
                  48, 24, width=140, size=8)
        panel.showPage()
    panel.setFont('Helvetica-Bold', 17)
    panel.drawString(48, height - 32, 'Busca de taxas - seleção somente na validação')
    for index, optimizer in enumerate(('sgd', 'nesterov', 'adam', 'adamw', 'muon')):
        rows = [row for row in all_rows if row['run'].startswith('p3_taxa_') and row['optimizer'] == optimizer]
        points = sorted((float(row['lr']), 100 * float(row['validation_accuracy'])) for row in rows if valid(row))
        failures = sum(not valid(row) for row in rows)
        x, y = 80 + (index % 2) * 390, height - 180 - (index // 2) * 163
        plot(panel, [('acurácia (%)', points)], x, y, 310, 95, optimizer, log_x=True)
        paragraph(panel, f'{len(rows)} execuções; {failures} divergentes (preservadas no CSV).',
                  x, y - 30, width=53, size=8)
    panel.showPage()
    panel.setFont('Helvetica-Bold', 17)
    panel.drawString(48, height - 32, 'Muon × AdamW - razão de atualização por camada')
    muon_logs, adamw_logs = logs_for(directory, rates['muon']), logs_for(directory, rates['adamw'])
    layers = [key for key in muon_logs[0] if key.startswith('layer_') and key.endswith('_update_ratio')]
    for index, layer in enumerate(layers):
        plot(panel, [('Muon + AdamW', signal(muon_logs, layer)), ('AdamW', signal(adamw_logs, layer))],
             90, height - 200 - index * 150, width - 145, 93,
             layer, log_y=True, x_end=float(rates['muon']['requested_steps']))
    paragraph(panel, 'Mesma agenda warmup + cosseno e semente 0; cada otimizador com sua taxa ajustada.',
              48, 29, size=9)
    panel.showPage()
    panel.setFont('Helvetica-Bold', 17)
    panel.drawString(48, height - 32, 'Confirmação, interpretação e limites')
    y = paragraph(panel, f"Baseline: {100 * baseline['mean']:.2f}% ± {100 * baseline['std']:.2f}% (desvio amostral, 3 sementes).",
                  48, height - 65)
    for item in summary['candidates']:
        recipe = item['recipe']
        text = f"{recipe['optimizer']} | lr={recipe['lr']} | {recipe['schedule']}: "
        text += (f"{100 * item['mean']:.2f}% ± {100 * item['std']:.2f}%; "
                 f"ganho pareado {item['gain_mean_pp']:.2f} ± {item['gain_std_pp']:.2f} p.p."
                 if item['eligible'] else 'Divergência em uma ou mais sementes; excluída da seleção, mantida no relatório.')
        y = paragraph(panel, text, 48, y - 15)
    for text in [status, observation,
                 'Critério descritivo: média dos ganhos pareados maior que seu desvio amostral. Três sementes não constituem prova de significância estatística.',
                 f"Teste final: {100 * final_test['test_accuracy']:.2f}% no checkpoint da semente 0 da receita selecionada. Não usado na seleção.",
                 f"Orçamento: {summary['configuration']['passos']} passos; lote {summary['configuration']['lote']}; "
                 f"camadas {summary['configuration']['camadas']}; largura {summary['configuration']['largura']}. "
                 'Baseline SGD 0.9, taxa constante, decay 0; candidatas decay 0.01. Warmup de 10%.',
                 'Limites: divisão por linhas sem identificação de partidas; teste preservado do trabalho 2, não historicamente inédito; classes sem nomes de ações confirmados.',
                 'Os exemplos e todas as execuções, inclusive divergentes, permanecem em experimentos.csv. Comparações oficiais usam apenas o prefixo p3_.']:
        y = paragraph(panel, text, 48, y - 16, width=115, size=9, leading=13)
    panel.save()
    # A single landscape slide with the measured result and an evidence-based observation.
    slide = Canvas(str(directory / 'slide.pdf'), pagesize=(960, 540))
    slide.setTitle('Desafio 3 - resultado da comparação')
    slide.setFillColor(colors.HexColor('#F3F6FA'))
    slide.rect(0, 0, 960, 540, fill=1, stroke=0)
    slide.setFillColor(colors.HexColor('#17324D'))
    slide.setFont('Helvetica-Bold', 26)
    slide.drawString(36, 498, 'Qual receita treinou melhor no problema do grupo?')
    recipe = chosen['recipe']
    paragraph(slide, f"Melhor candidata: {recipe['optimizer']} | taxa {recipe['lr']} | {recipe['schedule']}",
              36, 466, size=14)
    metrics = [('BASELINE', f"{100 * baseline['mean']:.2f}% ± {100 * baseline['std']:.2f}%"),
               ('MELHOR CANDIDATA', f"{100 * chosen['mean']:.2f}% ± {100 * chosen['std']:.2f}%"),
               ('GANHO PAREADO', f"{chosen['gain_mean_pp']:+.2f} ± {chosen['gain_std_pp']:.2f} p.p.")]
    for index, (label, value) in enumerate(metrics):
        x = 36 + index * 304
        slide.setFillColor(colors.white)
        slide.roundRect(x, 357, 285, 86, 8, fill=1, stroke=0)
        paragraph(slide, label, x + 15, 420, size=10)
        paragraph(slide, value, x + 15, 381, size=21)
    plot(slide, [('baseline', signal(logs_for(directory, baseline_row), 'validation_loss')),
                 ('candidata', signal(diagnostic_rows, 'validation_loss'))],
         75, 120, 385, 167, 'Perda de validação - semente 0', x_end=float(winner_row['requested_steps']))
    y = paragraph(slide, status, 508, 298, width=48, size=14, leading=18)
    paragraph(slide, observation, 508, y - 15, width=55, size=12, leading=17)
    paragraph(slide, 'Média ± desvio amostral; sementes 0, 1 e 2. Ganho em pontos percentuais. Comparação na validação.',
              36, 59, width=125, size=10)
    paragraph(slide, 'Critério descritivo, não teste de significância. Mesmo orçamento e arquitetura; decay declarado no painel.',
              36, 39, width=125, size=10)
    slide.save()
    print('Gerados:', directory / 'painel.pdf', 'e', directory / 'slide.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saida', type=Path, default=Path(__file__).resolve().parent)
    generate(parser.parse_args().saida.resolve())

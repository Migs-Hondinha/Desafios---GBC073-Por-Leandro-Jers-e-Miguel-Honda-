"""Run the assignment one stage at a time; never select on test data.

Stages: baseline -> taxas -> agendas -> confirmacao -> teste.
Each experiment calls the same treino.py that can be used manually.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'desafio2' / 'dataset_tekken.csv'
SEEDS = (0, 1, 2)
SCHEDULES = ('constante', 'cosseno', 'warmup_cosseno', 'degraus')


def read_json(path):
    if not path.exists():
        raise RuntimeError(f'Etapa anterior pendente: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def valid(row):
    return (row['diverged'] == 'False' and bool(row['validation_accuracy'])
            and math.isfinite(float(row['validation_accuracy']))
            and math.isfinite(float(row['validation_loss'])))


def ranking(row):
    return (-float(row['validation_accuracy']), float(row['validation_loss']))


def best(rows):
    candidates = [row for row in rows if valid(row)]
    if not candidates:
        raise RuntimeError('Nenhuma execução finita. Inspecione os logs antes de continuar.')
    return min(candidates, key=ranking)


def next_rate(rates, chosen):
    rates = sorted(set(rates))
    if chosen == rates[0]:
        return rates[0] / (rates[1] / rates[0])
    if chosen == rates[-1]:
        return rates[-1] * (rates[-1] / rates[-2])
    return None


def gain_statistics(baseline, candidate):
    if len(baseline) != 3 or len(candidate) != 3:
        raise ValueError('São necessárias três sementes pareadas.')
    gains = [100 * (new - old) for old, new in zip(baseline, candidate)]
    mean, deviation = statistics.mean(gains), statistics.stdev(gains)
    return {'gain_mean_pp': mean, 'gain_std_pp': deviation,
            'exceeds_variation': mean > deviation}


class Protocol:
    def __init__(self, args):
        self.args = args
        self.output = args.saida.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.table = self.output / 'experimentos.csv'
        # Bind the budget, input files and training implementation before any run.
        configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                         for key, value in vars(args).items() if key not in ('etapa', 'saida')}
        for label, path in [('data', args.dados), ('training', ROOT / 'treino.py'),
                            ('initializer', ROOT.parent / 'desafio2' / 'desafio2_Leandro_Marques_Miguel_Cesario.py')]:
            configuration[label + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        configuration['selection'] = 'final validation accuracy; ties: lower validation loss'
        configuration['seeds'] = list(SEEDS)
        path = self.output / 'protocolo_config.json'
        if path.exists() and read_json(path) != configuration:
            raise RuntimeError('Configuração/código/dados mudaram. Restaure-os ou use outra --saida para um novo estudo.')
        if not path.exists():
            save_json(path, configuration)

    def run(self, stage, optimizer, lr, schedule='warmup_cosseno', seed=0,
            adamw_lr=0.0003, decay=0.01):
        name = f'p3_{stage}_{optimizer}_{lr:.10g}_{schedule}_s{seed}'
        existing = [row for row in read_rows(self.table) if row['run'] == name]
        if existing:
            if len(existing) != 1:
                raise RuntimeError(f'Registro duplicado: {name}')
            return existing[0]
        if (self.output / 'teste_iniciado.json').exists():
            raise RuntimeError('O teste final já foi iniciado: não treine novas configurações neste estudo.')
        # An interrupted run remains visible rather than being silently overwritten.
        if (self.output / 'logs' / f'{name}.json').exists():
            raise RuntimeError(f'Execução interrompida: {name}. Preserve seus arquivos e revise antes de retomar.')
        command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'treino.py'),
                   '--dados', str(self.args.dados.resolve()), '--saida', str(self.output),
                   '--nome', name, '--otimizador', optimizer, '--lr', str(lr),
                   '--lr-adamw', str(adamw_lr), '--agenda', schedule, '--semente', str(seed),
                   '--passos', str(self.args.passos), '--lote', str(self.args.lote),
                   '--camadas', str(self.args.camadas), '--largura', str(self.args.largura),
                   '--weight-decay', str(decay), '--registrar-cada', '20']
        print('\nEXECUTANDO:', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
        return next(row for row in read_rows(self.table) if row['run'] == name)

    def baseline(self):
        rows = [self.run('baseline', 'sgd', self.args.baseline_lr, 'constante', seed, decay=0)
                for seed in SEEDS]
        if not all(valid(row) for row in rows):
            raise RuntimeError('Baseline divergiu. Revise a receita antes de comparar médias.')
        save_json(self.output / 'baseline.json', rows)

    def sweep(self, optimizer, adamw_lr=0.0003):
        rates = ([10 ** exponent for exponent in (-3, -2.5, -2, -1.5, -1)]
                 if optimizer == 'muon' else [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        # Ten extensions is a safety stop, not permission to accept a boundary winner.
        for extension in range(11):
            rows = [self.run('taxa', optimizer, lr, adamw_lr=adamw_lr) for lr in rates]
            winner = best(rows)
            new_rate = next_rate(rates, float(winner['lr']))
            if new_rate is None:
                return winner
            if extension == 10:
                raise RuntimeError(f'{optimizer}: melhor taxa ainda na borda após 10 extensões. Revise a busca.')
            print(f'{optimizer}: melhor taxa na borda; ampliando para {new_rate:g}.', flush=True)
            rates.append(new_rate)

    def rates(self):
        read_json(self.output / 'baseline.json')
        winners = {name: self.sweep(name) for name in ('sgd', 'nesterov', 'adam', 'adamw')}
        winners['muon'] = self.sweep('muon', float(winners['adamw']['lr']))
        save_json(self.output / 'melhores_taxas.json', winners)
        for name, row in winners.items():
            print(f"{name}: lr={row['lr']}; acurácia={float(row['validation_accuracy']):.2%}")

    def schedules(self):
        winners = read_json(self.output / 'melhores_taxas.json')
        names = sorted(winners, key=lambda name: ranking(winners[name]))[:2]
        if 'muon' not in names:
            names.append('muon')
        rows = [self.run('agenda', name, float(winners[name]['lr']), schedule,
                         adamw_lr=float(winners['adamw']['lr']))
                for name in names for schedule in SCHEDULES]
        candidates = sorted([row for row in rows if valid(row)], key=ranking)[:3]
        if len(candidates) < 3:
            raise RuntimeError('Menos de três combinações finitas para confirmar.')
        save_json(self.output / 'finalistas.json', candidates)
        print('Três finalistas salvas em finalistas.json.')

    def confirmation(self):
        baseline_rows = read_json(self.output / 'baseline.json')
        baseline_values = [float(row['validation_accuracy']) for row in baseline_rows]
        candidates = []
        for recipe in read_json(self.output / 'finalistas.json'):
            rows = [self.run('confirmacao', recipe['optimizer'], float(recipe['lr']),
                             recipe['schedule'], seed, float(recipe['lr_adamw'] or 0.0003))
                    for seed in SEEDS]
            if not all(valid(row) for row in rows):
                candidates.append({'recipe': recipe, 'runs': rows, 'eligible': False})
                continue
            values = [float(row['validation_accuracy']) for row in rows]
            candidates.append({'recipe': recipe, 'runs': rows, 'eligible': True,
                               'mean': statistics.mean(values), 'std': statistics.stdev(values),
                               **gain_statistics(baseline_values, values)})
        eligible = [item for item in candidates if item['eligible']]
        if not eligible:
            raise RuntimeError('Todas as finalistas tiveram divergência na confirmação; não há resultado confiável.')
        chosen = max(eligible, key=lambda item: (item['mean'], -statistics.mean(
            float(row['validation_loss']) for row in item['runs'])))
        # Keep seed 0 fixed in advance: do not cherry-pick a lucky checkpoint.
        selected = chosen['runs'][0] if chosen['exceeds_variation'] else baseline_rows[0]
        summary = {'baseline': {'runs': baseline_rows, 'mean': statistics.mean(baseline_values),
                                'std': statistics.stdev(baseline_values)},
                   'candidates': candidates, 'best_candidate': chosen, 'selected_for_test': selected,
                   'criterion': 'mean paired gain > sample std of paired gains; descriptive, n=3; not a significance test',
                   'configuration': read_json(self.output / 'protocolo_config.json')}
        save_json(self.output / 'resumo.json', summary)
        print(f"Melhor candidata: {chosen['mean']:.2%} ± {chosen['std']:.2%}")
        print(f"Ganho pareado: {chosen['gain_mean_pp']:.2f} ± {chosen['gain_std_pp']:.2f} pontos percentuais")
        print('Ganho supera o desvio entre sementes.' if chosen['exceeds_variation']
              else 'Sem ganho consistente pelo critério declarado; baseline mantida para o teste.')

    def diagnostic(self):
        read_json(self.output / 'resumo.json')
        # Search actual logs; never fabricate an unstable curve.
        for rate in (None, 1.0, 10.0, 100.0):
            if rate is not None:
                self.run('diagnostico', 'sgd', rate, 'constante')
            for row in read_rows(self.table):
                if not row['run'].startswith('p3_'):
                    continue
                logs = read_rows(self.output / 'logs' / (row['run'] + '.csv'))
                losses = [float(point['train_loss']) for point in logs
                          if math.isfinite(float(point['train_loss']))]
                if row['diverged'] == 'True' or (losses and max(losses) > 2 * losses[0]):
                    save_json(self.output / 'instavel.json', {
                        'run': row, 'criterion': 'nonfinite values or recorded loss > 2x first loss',
                        'controlled_extra': row['run'].startswith('p3_diagnostico_')})
                    print('Caso de instabilidade observado:', row['run'])
                    return
        raise RuntimeError('Não foi observado o critério de instabilidade. Inspecione os sinais antes de completar o painel.')

    def final_test(self):
        result_path = self.output / 'teste_final.json'
        if result_path.exists():
            print(json.dumps(read_json(result_path), ensure_ascii=False, indent=2))
            return
        marker = self.output / 'teste_iniciado.json'
        if marker.exists():
            raise RuntimeError('Teste já iniciado, sem resultado final. Revise a interrupção; não reavalie automaticamente.')
        read_json(self.output / 'instavel.json')
        summary = read_json(self.output / 'resumo.json')
        row = summary['selected_for_test']
        import torch
        from treino import MLP, load_data, evaluate
        torch.set_num_threads(1)
        metadata = read_json(self.output / 'logs' / (row['run'] + '.json'))
        data = load_data(self.args.dados)
        model = MLP(len(data['features']), data['classes'], metadata['largura'], metadata['camadas'])
        model.load_state_dict(torch.load(self.output / 'logs' / (row['run'] + '.pt'),
                                        map_location='cpu', weights_only=True))
        with marker.open('x', encoding='utf-8') as file:
            json.dump({'run': row['run'], 'seed': 0}, file)
        loss, accuracy = evaluate(model, data['test_x'], data['test_y'])
        result = {'run': row['run'], 'seed': 0, 'test_loss': loss, 'test_accuracy': accuracy,
                  'test_examples': len(data['test_y']), 'used_for_selection': False,
                  'limitation': 'Holdout preserved from assignment 2; not historically unseen.'}
        save_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('etapa', choices=['baseline', 'taxas', 'agendas', 'confirmacao', 'diagnostico', 'teste'])
    parser.add_argument('--saida', type=Path, default=ROOT)
    parser.add_argument('--dados', type=Path, default=DATA)
    parser.add_argument('--passos', type=int, default=225)
    parser.add_argument('--lote', type=int, default=128)
    parser.add_argument('--camadas', type=int, default=4)
    parser.add_argument('--largura', type=int, default=256)
    parser.add_argument('--baseline-lr', type=float, default=0.05)
    args = parser.parse_args()
    if args.passos < 3 or min(args.lote, args.largura) < 1 or args.camadas < 3:
        parser.error('Orçamento e arquitetura inválidos.')
    if not math.isfinite(args.baseline_lr) or args.baseline_lr <= 0:
        parser.error('Taxa da baseline deve ser finita e positiva.')
    protocol = Protocol(args)
    functions = {'baseline': protocol.baseline, 'taxas': protocol.rates,
                 'agendas': protocol.schedules, 'confirmacao': protocol.confirmation,
                 'diagnostico': protocol.diagnostic, 'teste': protocol.final_test}
    functions[args.etapa]()


if __name__ == '__main__':
    main()

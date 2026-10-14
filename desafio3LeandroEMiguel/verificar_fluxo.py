"""Integration check on SYNTHETIC data; never evaluates the Tekken holdout.

Uses a small network and runs treino.main in-process to avoid interpreter
startup overhead. Outputs stay under tmp/qa_*/ and are not assignment results.
"""
import argparse
import contextlib
import csv
import io
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from unittest.mock import patch

import protocolo
import treino


def main():
    root = Path(__file__).resolve().parent
    temporary_root = root / 'tmp'
    temporary_root.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='qa_', dir=temporary_root))
    source = output / 'synthetic.csv'
    generator = random.Random(2718)
    with source.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'x{i}' for i in range(10)] + ['y_target'])
        for _ in range(500):
            features = [generator.uniform(-1, 1) for _ in range(10)]
            writer.writerow(features + [int(features[0] + features[1] > 0)])
    args = argparse.Namespace(etapa='baseline', saida=output, dados=source,
                              passos=40, lote=32, camadas=4, largura=16, baseline_lr=0.05)
    protocol = protocolo.Protocol(args)

    def execute(command, check):
        start = command.index(str(root / 'treino.py'))
        with patch.object(sys, 'argv', command[start:]):
            treino.main()
        return subprocess.CompletedProcess(command, 0)

    with (output / 'qa_console.txt').open('w', encoding='utf-8') as console:
        with contextlib.redirect_stdout(console), patch.object(protocolo.subprocess, 'run', execute):
            for stage in (protocol.baseline, protocol.rates, protocol.schedules,
                          protocol.confirmation, protocol.diagnostic, protocol.final_test):
                stage()
            before = (output / 'teste_final.json').read_bytes()
            protocol.final_test()
            assert (output / 'teste_final.json').read_bytes() == before
            protocol.baseline()  # Existing runs must be reused, even after final test.
            try:
                protocol.run('forbidden_after_test', 'sgd', 0.1)
            except RuntimeError:
                pass
            else:
                raise AssertionError('New training was allowed after final test')
    print('QA com dados sintéticos concluída:', output)
    print('Execuções:', len(protocolo.read_rows(output / 'experimentos.csv')))


if __name__ == '__main__':
    main()

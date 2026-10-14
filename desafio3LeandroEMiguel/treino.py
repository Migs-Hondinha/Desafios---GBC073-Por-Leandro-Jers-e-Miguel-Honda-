"""Desafio 3: choose an optimizer and schedule, then record learning signals.

Read main() first, then follow the numbered sections below.
Experiment sweeps and final test evaluation are separate steps.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "desafio2"
DEFAULT_DATA = PREVIOUS / "dataset_tekken.csv"
INITIALIZER = PREVIOUS / "desafio2_Leandro_Marques_Miguel_Cesario.py"
sys.path.insert(0, str(PREVIOUS))
from desafio2_Leandro_Marques_Miguel_Cesario import ativacao, inicializar


# 1. Data: preserve the previous 20% holdout, then reserve validation data.
def load_data(path: Path) -> dict:
    """Split by class; fit normalization on training rows only."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        if "y_target" not in columns or len(columns) != len(set(columns)):
            raise ValueError("CSV precisa de colunas únicas e y_target.")
        features = [name for name in columns if name != "y_target"]
        if not features:
            raise ValueError("CSV sem atributos de entrada.")
        rows, labels = [], []
        for row in reader:
            values = [float(row[name]) for name in features]
            target = float(row["y_target"])
            if None in row or not all(math.isfinite(v) for v in values + [target]):
                raise ValueError("CSV contém valores ausentes, extras ou não finitos.")
            if not target.is_integer() or target < 0:
                raise ValueError("y_target precisa ser um inteiro não negativo.")
            rows.append(values)
            labels.append(int(target))
    classes = sorted(set(labels))
    if len(classes) < 2 or classes != list(range(len(classes))):
        raise ValueError("Classes precisam ser 0, 1, ..., sem lacunas.")
    x, y = torch.tensor(rows, dtype=torch.float32), torch.tensor(labels)
    if not torch.isfinite(x).all():
        raise ValueError("Valores excedem a capacidade de float32.")
    holdout_generator = torch.Generator().manual_seed(42)
    validation_generator = torch.Generator().manual_seed(2026)
    train_parts, validation_parts, test_parts = [], [], []
    for label in classes:
        indices = (y == label).nonzero(as_tuple=True)[0]
        if len(indices) < 5:
            raise ValueError("Cada classe precisa ter pelo menos 5 exemplos.")
        indices = indices[torch.randperm(len(indices), generator=holdout_generator)]
        test_count = max(1, round(0.2 * len(indices)))
        test_parts.append(indices[:test_count])
        remaining = indices[test_count:]
        remaining = remaining[torch.randperm(len(remaining), generator=validation_generator)]
        validation_count = max(1, round(0.2 * len(remaining)))
        validation_parts.append(remaining[:validation_count])
        train_parts.append(remaining[validation_count:])
    train, validation, test = map(torch.cat, (train_parts, validation_parts, test_parts))
    mean = x[train].mean(0)
    std = x[train].std(0, correction=0).clamp_min(1e-6)
    return {
        "train_x": (x[train] - mean) / std, "train_y": y[train],
        "validation_x": (x[validation] - mean) / std, "validation_y": y[validation],
        "test_x": (x[test] - mean) / std, "test_y": y[test],
        "train_indices": train, "validation_indices": validation, "test_indices": test,
        "row_count": len(y), "classes": len(classes), "features": features,
    }


# 2. Model: four Linear layers by default, with the previous initialization.
class MLP(nn.Module):
    def __init__(self, input_size: int, classes: int, width: int, depth: int):
        super().__init__()
        dimensions = [input_size] + [width] * (depth - 1) + [classes]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[i], dimensions[i + 1]) for i in range(depth)
        )
        for index, layer in enumerate(self.layers, start=1):
            inicializar(layer.weight, layer.bias, layer.in_features,
                        layer.out_features, index, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = ativacao(layer(x))
        # CrossEntropyLoss expects logits, not probabilities or tanh outputs.
        return self.layers[-1](x)


# 3. Optimizers: Muon owns hidden weight matrices; AdamW owns the rest.
def make_optimizers(model: MLP, args: argparse.Namespace) -> dict[str, torch.optim.Optimizer]:
    options = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.otimizador == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon requer PyTorch com torch.optim.Muon (>= 2.9).")
        hidden_weights = [layer.weight for layer in model.layers[:-1]]
        hidden_ids = {id(p) for p in hidden_weights}
        other_parameters = [p for p in model.parameters() if id(p) not in hidden_ids]
        return {
            "muon": torch.optim.Muon(hidden_weights, momentum=0.95, nesterov=True,
                                      ns_steps=5, adjust_lr_fn="original", **options),
            "adamw": torch.optim.AdamW(other_parameters, lr=args.lr_adamw,
                                        weight_decay=args.weight_decay),
        }
    if args.otimizador in ("sgd", "nesterov"):
        return {args.otimizador: torch.optim.SGD(
            model.parameters(), momentum=0.9,
            nesterov=args.otimizador == "nesterov", **options)}
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[args.otimizador]
    return {args.otimizador: optimizer_class(model.parameters(), **options)}


def schedule_factor(schedule: str, step: int, total: int, warmup_fraction: float) -> float:
    """Return a multiplier for the rate used NOW (step is one-based).

    Both Muon and its companion AdamW receive the same multiplier.
    Cosine schedules reach zero on the final update; a one-step cosine uses 1.
    """
    if schedule == "constante":
        return 1.0
    if schedule == "degraus":
        # Reduce by ten after 50% and 75% of the update budget.
        completed = step - 1
        reductions = int(completed >= math.ceil(0.5 * total))
        reductions += int(completed >= math.ceil(0.75 * total))
        return 0.1 ** reductions
    if schedule == "cosseno":
        progress = (step - 1) / max(1, total - 1)
    elif schedule == "warmup_cosseno":
        warmup_steps = min(total - 1, max(1, round(total * warmup_fraction)))
        if step <= warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total - warmup_steps)
    else:
        raise ValueError(f"Agenda desconhecida: {schedule}")
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_schedule(optimizers: dict[str, torch.optim.Optimizer],
                   args: argparse.Namespace, step: int) -> dict[str, float]:
    """Set and return the rates before the optimizer updates the parameters."""
    factor = schedule_factor(args.agenda, step, args.passos, args.warmup_fracao)
    rates = {}
    for name, optimizer in optimizers.items():
        base_lr = args.lr_adamw if args.otimizador == "muon" and name == "adamw" else args.lr
        rates[name] = base_lr * factor
        for group in optimizer.param_groups:
            group["lr"] = rates[name]
    return rates


# 4. Diagnostics: aggregate weights AND biases; gradients are measured before step.
def vector_norm(tensors) -> float:
    return math.sqrt(sum(tensor.detach().double().square().sum().item() for tensor in tensors))


@torch.no_grad()
def evaluate(model: MLP, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    accuracy = (logits.argmax(1) == y).float().mean().item()
    model.train()
    return loss, accuracy


def append_csv(path: Path, row: dict) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    fields = list(row)
    if not needs_header:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_fields = reader.fieldnames or []
            added_fields = [key for key in row if key not in existing_fields]
            fields = existing_fields + added_fields
            old_rows = list(reader) if added_fields else []
        if added_fields:
            # Extend older summaries without discarding or inventing old values.
            temporary_path = path.with_suffix(".csv.tmp")
            with temporary_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows)
            temporary_path.replace(path)
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


# 5. Training: forward -> loss -> backward -> diagnostics -> optimizer step.
def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(args.semente)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.dados)
    model = MLP(len(data["features"]), data["classes"], args.largura, args.camadas)
    optimizers = make_optimizers(model, args)
    loader = DataLoader(TensorDataset(data["train_x"], data["train_y"]),
                        batch_size=args.lote, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.semente))
    if len(loader) == 0:
        raise ValueError("O lote não pode exceder o número de exemplos de treino.")
    args.saida.mkdir(parents=True, exist_ok=True)
    log_directory = args.saida / "logs"
    log_directory.mkdir(exist_ok=True)
    run_name = args.nome or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_directory / f"{run_name}.csv"
    metadata_path = log_directory / f"{run_name}.json"
    if log_path.exists() or metadata_path.exists():
        raise ValueError("Nome de execução já existe. Escolha outro --nome.")
    metadata = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    metadata.update({"torch_version": torch.__version__, "python_version": sys.version,
                     "device": "cpu", "test_evaluated": False,
                     "dataset_sha256": hashlib.sha256(args.dados.read_bytes()).hexdigest(),
                     "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "initializer_sha256": hashlib.sha256(INITIALIZER.read_bytes()).hexdigest(),
                     "features": data["features"],
                     "split_seeds": {"holdout": 42, "validation": 2026}})
    metadata["optimizer_defaults"] = {name: optimizer.defaults for name, optimizer in optimizers.items()}
    metadata["optimizer_parameters"] = {
        name: [parameter_name for parameter_name, parameter in model.named_parameters()
               if any(parameter is p for group in optimizer.param_groups for p in group["params"])]
        for name, optimizer in optimizers.items()
    }
    for key in ("train_indices", "validation_indices", "test_indices"):
        metadata[key] = data[key].tolist()
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    groups = {"global": list(model.parameters())}
    for index in sorted({0, args.camadas // 2, args.camadas - 1}):
        groups[f"layer_{index}"] = list(model.layers[index].parameters())
    print(f"Treino: {len(data['train_y'])} | Validação: {len(data['validation_y'])} | "
          f"Teste reservado: {len(data['test_indices'])}")
    print(f"Otimizador: {args.otimizador} | agenda: {args.agenda} | "
          f"taxa base: {args.lr:g} | semente: {args.semente}")
    if args.otimizador == "muon":
        print(f"Taxa base do AdamW auxiliar: {args.lr_adamw:g}")
    iterator = iter(loader)
    started = time.perf_counter()
    diverged = False
    validation_loss, accuracy = evaluate(model, data["validation_x"], data["validation_y"])
    for step in range(1, args.passos + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        model.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        finite_gradients = all(p.grad is None or torch.isfinite(p.grad).all()
                               for p in model.parameters())
        diverged = not math.isfinite(loss.item()) or not finite_gradients
        record = step == 1 or step % args.registrar_cada == 0 or step == args.passos or diverged
        before, signals = {}, {}
        if record:
            for name, parameters in groups.items():
                before[name] = [p.detach().clone() for p in parameters]
                signals[f"{name}_grad_norm"] = vector_norm(p.grad for p in parameters if p.grad is not None)
                signals[f"{name}_parameter_norm"] = vector_norm(parameters)
        rates = apply_schedule(optimizers, args, step)
        if not diverged:
            for optimizer in optimizers.values():
                optimizer.step()
            diverged = not all(torch.isfinite(p).all() for p in model.parameters())
        # Always retain diagnostics when a non-finite update is detected.
        if diverged and not record:
            append_csv(log_directory / f"{run_name}_failure.csv", {"step": step, "reason": "nonfinite_update"})
        if record:
            for name, parameters in groups.items():
                change = vector_norm(p.detach() - old for p, old in zip(parameters, before[name]))
                signals[f"{name}_update_ratio"] = change / max(signals[f"{name}_parameter_norm"], 1e-12)
            validation_loss, accuracy = evaluate(model, data["validation_x"], data["validation_y"])
            diverged = diverged or not math.isfinite(validation_loss)
            append_csv(log_path, {"step": step, "train_loss": loss.item(),
                                  "validation_loss": validation_loss, "validation_accuracy": accuracy,
                                  "lr": rates[args.otimizador],
                                  "lr_adamw": rates["adamw"] if args.otimizador == "muon" else "",
                                  **signals})
            print(f"Passo {step:4d} | perda={loss.item():.4f} | "
                  f"validação={validation_loss:.4f} | acurácia={accuracy:.2%} | "
                  f"lr={rates[args.otimizador]:.6g}")
        if diverged:
            break
    if not diverged:
        torch.save(model.state_dict(), log_directory / f"{run_name}.pt")
    append_csv(args.saida / "experimentos.csv", {
        "run": run_name, "optimizer": args.otimizador, "lr": args.lr,
        "schedule": args.agenda, "seed": args.semente, "requested_steps": args.passos,
        "completed_steps": step, "batch_size": args.lote, "width": args.largura,
        "depth": args.camadas, "weight_decay": args.weight_decay,
        "validation_loss": validation_loss if not diverged else "",
        "validation_accuracy": accuracy if not diverged else "", "diverged": diverged,
        "seconds": time.perf_counter() - started, "metadata": str(metadata_path.resolve()),
        "lr_adamw": args.lr_adamw if args.otimizador == "muon" else "",
        "warmup_fraction": args.warmup_fracao if args.agenda == "warmup_cosseno" else "",
    })
    print(f"Estado: {'DIVERGIU' if diverged else 'concluído'} | Registro: {log_path}")


# 6. Entry point: validate user options before running an experiment.
def main() -> None:
    parser = argparse.ArgumentParser(description="Trabalho 3 — treino configurável com registro de sinais.")
    parser.add_argument("--dados", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--otimizador", choices=["sgd", "nesterov", "adam", "adamw", "muon"], default="sgd")
    parser.add_argument("--lr", type=float, help="Taxa base (pico no warmup). Padrão: SGD/Nesterov=0.05, Adam/AdamW=0.001, Muon=0.02.")
    parser.add_argument("--lr-adamw", type=float, default=0.0003,
                        help="Taxa base do AdamW auxiliar do Muon; padrão: 0.0003.")
    parser.add_argument("--agenda", choices=["constante", "cosseno", "warmup_cosseno", "degraus"],
                        default="constante", help="Como a taxa varia durante o treino.")
    parser.add_argument("--warmup-fracao", type=float, default=0.1,
                        help="Fração de passos de aquecimento em warmup_cosseno; padrão: 0.1.")
    parser.add_argument("--semente", type=int, default=0, help="Semente da inicialização e dos lotes; padrão: 0.")
    parser.add_argument("--passos", type=int, default=100)
    parser.add_argument("--lote", type=int, default=128)
    parser.add_argument("--largura", type=int, default=256)
    parser.add_argument("--camadas", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--registrar-cada", type=int, default=20)
    parser.add_argument("--saida", type=Path, default=ROOT)
    parser.add_argument("--nome", help="Identificador único; automático quando omitido.")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = {"sgd": 0.05, "nesterov": 0.05, "adam": 0.001,
                   "adamw": 0.001, "muon": 0.02}[args.otimizador]
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr deve ser finita e positiva.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay deve ser finito e não negativo.")
    if not math.isfinite(args.lr_adamw) or args.lr_adamw <= 0:
        parser.error("--lr-adamw deve ser finita e positiva.")
    if not 0 < args.warmup_fracao < 1:
        parser.error("--warmup-fracao deve estar entre 0 e 1, sem incluir os extremos.")
    if min(args.passos, args.lote, args.largura) < 1 or args.camadas < 3:
        parser.error("Passos, lote e largura positivos; pelo menos 3 camadas.")
    if not 10 <= args.registrar_cada <= 50:
        parser.error("--registrar-cada deve estar entre 10 e 50.")
    if args.agenda == "warmup_cosseno" and args.passos < 3:
        parser.error("warmup_cosseno requer pelo menos 3 passos.")
    if not 0 <= args.semente < 2 ** 63:
        parser.error("--semente deve ser um inteiro entre 0 e 2**63 - 1.")
    if args.nome and (args.nome in (".", "..") or any(c in args.nome for c in '/\\:*?"<>|')):
        parser.error("--nome deve ser um nome de arquivo simples.")
    args.dados = args.dados.resolve()
    args.saida = args.saida.resolve()
    train(args)


if __name__ == "__main__":
    main()

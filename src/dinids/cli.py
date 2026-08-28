"""Command-line interface for training and evaluating DI-NIDS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dinids import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinids",
        description="Train and evaluate the DI-NIDS research implementation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-dann", help="Train source-only and DANN models")
    _add_domain_arguments(train)
    train.add_argument("--output-dir", type=Path, default=Path("runs/dann"))
    train.add_argument("--source-epochs", type=_positive_int, default=10)
    train.add_argument("--dann-epochs", type=_positive_int, default=10)
    train.add_argument("--batch-size", type=_positive_int, default=1024)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument(
        "--shuffle-batches",
        action="store_true",
        help="shuffle each epoch; the supplied implementation retained a fixed split order",
    )
    train.add_argument(
        "--optimisation",
        choices=("legacy", "scheduled", "constant"),
        default="legacy",
        help="legacy schedules source pretraining but keeps DANN learning rate constant",
    )
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate OSVM or DI-NIDS")
    _add_domain_arguments(evaluate)
    evaluate.add_argument(
        "--encoder",
        type=Path,
        help="DANN encoder state dictionary; omit it to evaluate the raw-feature OSVM baseline",
    )
    evaluate.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    evaluate.add_argument("--nu", type=float, default=0.001)
    evaluate.add_argument("--kernel", default="poly")
    evaluate.add_argument("--feature-batch-size", type=_positive_int, default=8192)
    evaluate.add_argument(
        "--max-benign-train-rows",
        type=_positive_int,
        help="optionally cap the benign source rows used to fit the kernel OSVM",
    )
    evaluate.set_defaults(handler=_evaluate)
    return parser


def _add_domain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-name", help="Display name; defaults to the source filename")
    parser.add_argument("--target-name", help="Display name; defaults to the target filename")
    parser.add_argument("--test-size", type=_unit_interval, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument(
        "--preprocessing",
        choices=("legacy-independent", "source-train"),
        default="legacy-independent",
        help=(
            "legacy-independent matches the supplied implementation; "
            "source-train avoids test fitting"
        ),
    )
    parser.add_argument("--max-source-rows", type=_positive_int)
    parser.add_argument("--max-target-rows", type=_positive_int)
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="allow trusted .pkl/.pickle input; pickle files can execute code while loading",
    )


def _train(args: argparse.Namespace) -> int:
    from dinids.data import load_domain, prepare_training_data
    from dinids.pipeline import resolve_device
    from dinids.training import TrainConfig, train_dann_models

    source = load_domain(
        args.source,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        max_rows=args.max_source_rows,
        seed=args.seed,
    )
    target = load_domain(
        args.target,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        max_rows=args.max_target_rows,
        seed=args.seed,
    )
    data = prepare_training_data(
        source,
        target,
        test_size=args.test_size,
        seed=args.seed,
        preprocessing=args.preprocessing,
    )
    config = TrainConfig(
        source_epochs=args.source_epochs,
        dann_epochs=args.dann_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        seed=args.seed,
        num_workers=args.num_workers,
        optimisation=args.optimisation,
        shuffle_batches=args.shuffle_batches,
    )
    summary = train_dann_models(
        data,
        output_dir=args.output_dir,
        device=resolve_device(args.device),
        config=config,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from dinids.data import load_domain, prepare_evaluation_data
    from dinids.pipeline import evaluate_pair, resolve_device, write_results

    source = load_domain(
        args.source,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        max_rows=args.max_source_rows,
        seed=args.seed,
    )
    target = load_domain(
        args.target,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        max_rows=args.max_target_rows,
        seed=args.seed,
    )
    data = prepare_evaluation_data(
        source,
        target,
        test_size=args.test_size,
        seed=args.seed,
        preprocessing=args.preprocessing,
    )
    results = evaluate_pair(
        data,
        source_name=args.source_name or args.source.name,
        target_name=args.target_name or args.target.name,
        encoder_path=args.encoder,
        device=resolve_device(args.device),
        feature_batch_size=args.feature_batch_size,
        nu=args.nu,
        kernel=args.kernel,
        preprocessing=args.preprocessing,
        max_benign_train_rows=args.max_benign_train_rows,
        seed=args.seed,
    )
    destination = write_results(results, args.output)
    print(json.dumps(results, indent=2))
    print(f"Results written to {destination}", file=sys.stderr)
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

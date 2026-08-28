from dinids.cli import build_parser


def test_evaluate_parser_accepts_public_command_shape() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--source",
            "source.csv",
            "--target",
            "target.csv",
            "--encoder",
            "encoder.pt",
        ]
    )
    assert args.command == "evaluate"
    assert args.preprocessing == "legacy-independent"
    assert args.encoder.name == "encoder.pt"

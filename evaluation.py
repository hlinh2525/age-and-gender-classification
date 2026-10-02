import json
from pathlib import Path

import torch

from training import MultitaskLoss, run_epoch


def load_best_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def evaluate_test(
    model,
    test_loader,
    device,
    output_dir,
    age_class_weights=None,
    checkpoint_config=None,
):
    criterion = MultitaskLoss(
        age_class_weights=age_class_weights,
    )
    metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "checkpoint_config": checkpoint_config,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return metrics


def print_metrics(metrics):
    for name, value in metrics.items():
        print(f"{name}: {value}")

import torch
from training import (AGE_LOSS_WEIGHT, GENDER_LOSS_WEIGHT, 
                      MultitaskLoss, run_epoch, save_json)

def evaluate_test(model, test_loader, device, output_dir,
                  age_class_weights=None, checkpoint_config=None):
    """Evaluate a loaded best checkpoint once on the untouched test split."""
    metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=MultitaskLoss(
            age_class_weights=age_class_weights,
            age_weight=AGE_LOSS_WEIGHT,
            gender_weight=GENDER_LOSS_WEIGHT),
        device=device,
        optimizer=None
    )
    save_json(
        {
            "metrics": metrics,
            "optimizer": (checkpoint_config or {}).get("optimizer"),
            "age_loss_weight": AGE_LOSS_WEIGHT,
            "gender_loss_weight": GENDER_LOSS_WEIGHT
        },
        output_dir / "test_metrics.json"
    )
    return metrics


def print_metrics(metrics):
    for name, value in metrics.items():
        if name == "age_confusion_matrix":
            matrix = value
            cell_width = max(5, max(len(str(cell)) for row in matrix for cell in row))
            for row in matrix:
                print(" ".join(f"{cell:>{cell_width}}" for cell in row))
            continue
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, (int, float)) for item in value)
        ):
            formatted_values = ", ".join(f"{item:.4f}" for item in value)
            print(f"{name}: [{formatted_values}]")
        else:
            print(f"{name}: {value}")


def load_best_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
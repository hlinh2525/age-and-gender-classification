import argparse
from pathlib import Path
import torch

from evaluation import evaluate_test, load_best_checkpoint, print_metrics
from explainability import (
    save_gradcam_visualization,
    select_gradcam_indices
)
from models.complex_cnn import ComplexCNN
from preprocessing import build_loaders
from training import compute_age_class_weights, train_model


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "complex_cnn"
GRADCAM_SELECTION_PATH = OUTPUT_DIR / "gradcam_samples.json"
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3   # dùng LR như SimpleCNN
WEIGHT_DECAY = 1e-4
PATIENCE = 5
GRADCAM_SAMPLE_COUNT = 8


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def close_datasets(loaders):
    for loader in loaders.values():
        close_method = getattr(loader.dataset, "close", None)
        if callable(close_method):
            close_method()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()
    loaders = build_loaders(
        model_type="scratch",
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size
    )
    model = ComplexCNN().to(device)
    age_class_weights = compute_age_class_weights(loaders["train"], device)

    try:
        _, checkpoint_path = train_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            device=device,
            output_dir=OUTPUT_DIR,
            model_name="ComplexCNN",
            max_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            age_class_weights=age_class_weights
        )
        checkpoint = load_best_checkpoint(model, checkpoint_path, device)
        test_metrics = evaluate_test(
            model,
            loaders["test"],
            device,
            OUTPUT_DIR,
            age_class_weights=age_class_weights,
            checkpoint_config=checkpoint.get("config")
        )
        print("Test metrics:")
        print_metrics(test_metrics)

        GRADCAM_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        gradcam_indices = select_gradcam_indices(
            loader=loaders["val"],
            selection_path=GRADCAM_SELECTION_PATH,
            sample_count=GRADCAM_SAMPLE_COUNT
        )
        for target_name in ["age", "gender"]:
            save_gradcam_visualization(
                model=model,
                loader=loaders["val"],
                device=device,
                output_path=OUTPUT_DIR / f"gradcam_{target_name}.png",
                target_name=target_name,
                selected_indices=gradcam_indices,
                target_layer=model.get_gradcam_layer(target_name),
                model_type="scratch"
            )
    finally:
        close_datasets(loaders)


if __name__ == "__main__":
    main()
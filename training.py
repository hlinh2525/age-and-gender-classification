import json
import time
from pathlib import Path
import torch
from torch import nn
from age_config import AGE_LABELS, NUM_AGE_CLASSES

AGE_LOSS_WEIGHT = 1.5
GENDER_LOSS_WEIGHT = 0.5

class MultitaskLoss(nn.Module):
    def __init__(
        self,
        age_class_weights=None,
        age_weight=AGE_LOSS_WEIGHT,
        gender_weight=GENDER_LOSS_WEIGHT
    ):
        super().__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.age_criterion = nn.CrossEntropyLoss(weight=age_class_weights)
        self.gender_criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, true_age_class, true_gender):
        age_loss = self.age_criterion(outputs["age"], true_age_class)
        gender_loss = self.gender_criterion(outputs["gender"],true_gender)
        total_loss = (self.age_weight * age_loss + self.gender_weight * gender_loss)
        
        return total_loss, age_loss, gender_loss


class EpochMetrics:
    def __init__(self):
        self.samples = 0
        self.total_loss = 0.0
        self.age_loss = 0.0
        self.gender_loss = 0.0
        self.correct_age = 0
        self.age_confusion = torch.zeros(
            NUM_AGE_CLASSES,
            NUM_AGE_CLASSES,
            dtype=torch.long
        )
        self.correct_gender = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, batch_size, total_loss, age_loss, gender_loss,
               predicted_age_class, true_age_class, predicted_gender, true_gender
    ):
        self.samples += batch_size
        self.total_loss += total_loss.item() * batch_size
        self.age_loss += age_loss.item() * batch_size
        self.gender_loss += gender_loss.item() * batch_size
        self.correct_age += (
            predicted_age_class == true_age_class
        ).sum().item()
        confusion_indices = (
            true_age_class.detach().cpu() * NUM_AGE_CLASSES
            + predicted_age_class.detach().cpu()
        )
        self.age_confusion += torch.bincount(
            confusion_indices,
            minlength=NUM_AGE_CLASSES * NUM_AGE_CLASSES
        ).reshape(NUM_AGE_CLASSES, NUM_AGE_CLASSES)
        self.correct_gender += (predicted_gender == true_gender).sum().item()
        self.true_positive += (
            (predicted_gender == 1) & (true_gender == 1)
        ).sum().item()
        self.false_positive += (
            (predicted_gender == 1) & (true_gender == 0)
        ).sum().item()
        self.false_negative += (
            (predicted_gender == 0) & (true_gender == 1)
        ).sum().item()

    def compute(self):
        if self.samples == 0:
            raise RuntimeError("Cannot calculate metrics from an empty loader")
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = (
            self.true_positive / precision_denominator
            if precision_denominator else 0.0
        )
        recall = (
            self.true_positive / recall_denominator
            if recall_denominator else 0.0
        )
        f1_denominator = precision + recall
        f1 = (
            2.0 * precision * recall / f1_denominator
            if f1_denominator else 0.0
        )
        true_positive_age = self.age_confusion.diag().float()
        actual_age = self.age_confusion.sum(dim=1).float()
        predicted_age = self.age_confusion.sum(dim=0).float()
        age_precision = true_positive_age / predicted_age.clamp_min(1)
        age_recall = true_positive_age / actual_age.clamp_min(1)
        age_f1 = (
            2 * age_precision * age_recall
            / (age_precision + age_recall).clamp_min(1e-8)
        )

        return {
            "loss": self.total_loss / self.samples,
            "age_cross_entropy": self.age_loss / self.samples,
            "gender_cross_entropy": self.gender_loss / self.samples,
            "age_accuracy": self.correct_age / self.samples,
            "age_balanced_accuracy": age_recall.mean().item(),
            "age_macro_f1": age_f1.mean().item(),
            "age_recall_per_class": age_recall.tolist(),
            "age_confusion_matrix": self.age_confusion.tolist(),
            "gender_accuracy": self.correct_gender / self.samples,
            "gender_precision": precision,
            "gender_recall": recall,
            "gender_f1": f1
        }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metrics = EpochMetrics()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            true_age_class = batch["age_class"].to(device)
            true_gender = batch["gender"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            total_loss, age_loss, gender_loss = criterion(
                outputs, true_age_class, true_gender
            )
            if training:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            predicted_age_class = outputs["age"].argmax(dim=1)
            predicted_gender = outputs["gender"].argmax(dim=1)
            metrics.update(
                images.size(0), total_loss, age_loss, gender_loss,
                predicted_age_class, true_age_class,
                predicted_gender, true_gender
            )
    return metrics.compute()


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad)


def compute_age_class_weights(train_loader, device):
    labels = torch.as_tensor(
        train_loader.dataset.data["age_class"].to_numpy(copy=True),
        dtype=torch.long
    )
    counts = torch.bincount(labels, minlength=NUM_AGE_CLASSES).float()
    inverse_frequency = counts.sum() / (
        NUM_AGE_CLASSES * counts.clamp_min(1)
    )
    weights = inverse_frequency.sqrt()
    return (weights / weights.mean()).to(device)


def save_json(data, output_path):
    Path(output_path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8")


def save_checkpoint(output_path, model, optimizer, epoch, metrics, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": metrics,
        "config": config
    }, output_path)


def train_model(model, train_loader, val_loader, device, output_dir, model_name,
                max_epochs, learning_rate, weight_decay, patience, age_class_weights=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = MultitaskLoss(
        age_class_weights=age_class_weights,
        age_weight=AGE_LOSS_WEIGHT,
        gender_weight=GENDER_LOSS_WEIGHT
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )
    config = {
        "model": model_name,
        "parameter_count": count_parameters(model),
        "max_epochs": max_epochs,
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay
        },
        "early_stopping_patience": patience,
        "age_labels": AGE_LABELS,
        "age_class_weights": age_class_weights,
        "age_loss": "weighted CrossEntropyLoss",
        "age_loss_weight": AGE_LOSS_WEIGHT,
        "gender_loss": "CrossEntropyLoss",
        "gender_loss_weight": GENDER_LOSS_WEIGHT,
        "checkpoint_metric": "age_macro_f1"
    }
    checkpoint_path = output_dir / "best_model.pt"
    history = []
    best_checkpoint_score = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        start_time = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["age_macro_f1"])
        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start_time,
            "train": train_metrics,
            "val": val_metrics
        })
        save_json(history, output_dir / "history.json")
        checkpoint_score = val_metrics["age_macro_f1"]
        if checkpoint_score > best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, val_metrics, config)
        else:
            epochs_without_improvement += 1
        print(
            f"{model_name} epoch {epoch:02d}/{max_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_age_macro_f1={val_metrics['age_macro_f1']:.4f} | "
            f"val_gender_acc={val_metrics['gender_accuracy']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
        )
        
        if epochs_without_improvement >= patience:
            break
    return history, checkpoint_path
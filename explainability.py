import json
import math
import torch
from preprocessing import IMAGENET_MEAN, IMAGENET_STD
from age_config import AGE_LABELS


def compute_gradcam(model, images, target, target_layer):
    activations = {}

    def save_activations(_module, _inputs, features):
        activations["features"] = features
        features.retain_grad()

    hook = target_layer.register_forward_hook(save_activations)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        target(outputs).sum().backward()
        feature_maps = activations["features"]
        gradients = feature_maps.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1).relu()
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(1)
        cam_min = cam.flatten(1).min(dim=1).values[:, None, None]
        cam_max = cam.flatten(1).max(dim=1).values[:, None, None]
        return (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)
    finally:
        hook.remove()


def select_gradcam_indices(loader, selection_path, sample_count):
    members = loader.dataset.data["member"].astype(str).tolist()
    member_to_index = {member: index for index, member in enumerate(members)}
    sample_count = min(sample_count, len(members))

    if selection_path.is_file():
        selected_members = json.loads(
            selection_path.read_text(encoding="utf-8")
        )["members"]
        selected_members = [str(member) for member in selected_members]
        if (
            len(selected_members) == sample_count
            and len(set(selected_members)) == sample_count
            and all(member in member_to_index for member in selected_members)
        ):
            return [member_to_index[member] for member in selected_members]

    selected_indices = torch.randperm(len(members))[:sample_count].tolist()
    selected_members = [members[index] for index in selected_indices]

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps({"members": selected_members}, indent=2),
        encoding="utf-8"
    )
    return [member_to_index[member] for member in selected_members]


def _prediction_target(outputs, target_name):
    if target_name not in ("age", "gender"):
        raise ValueError("target_name must be age or gender")
    predicted = outputs[target_name].argmax(dim=1)
    target = lambda predictions: predictions[target_name].gather(
        1, predicted[:, None]
    ).squeeze(1)
    if target_name == "age":
        titles = [
            f"Age: {AGE_LABELS[value.item()]}"
            for value in predicted.detach().cpu()
        ]
    else:
        titles = [
            f"Gender: {value.item()}"
            for value in predicted.detach().cpu()
        ]
    return target, titles


def save_gradcam_visualization(
    model,
    loader,
    device,
    output_path,
    target_name,
    selected_indices,
    target_layer,
    model_type="scratch"
):
    import matplotlib.pyplot as plt

    if model_type not in ("scratch", "resnet18"):
        raise ValueError("model_type must be either scratch or resnet18.")

    images = torch.stack([
        loader.dataset[index]["image"] for index in selected_indices
    ]).to(device)
    model.eval()
    with torch.enable_grad():
        outputs = model(images)
        target, titles = _prediction_target(outputs, target_name)
        heatmaps = compute_gradcam(model, images, target, target_layer)
        heatmaps = heatmaps.detach().cpu()

    columns = 4
    rows = math.ceil(len(images) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(12, 3 * rows), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(images):
            continue
        image = images[index].detach().cpu().permute(1, 2, 0)
        if model_type == "resnet18":
            image = (
                image * torch.tensor(IMAGENET_STD)
                + torch.tensor(IMAGENET_MEAN)
            )
        image = image.clamp(0, 1)
        axis.imshow(image)
        axis.imshow(heatmaps[index], cmap="jet", alpha=0.45)
        axis.set_title(titles[index])
    figure.suptitle(f"Grad-CAM: {target_name}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

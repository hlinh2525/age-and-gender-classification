import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def select_gradcam_indices(loader, selection_path, sample_count=8):
    count = min(sample_count, len(loader.dataset))
    indices = list(range(count))

    selection_path = Path(selection_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps({"indices": indices}, indent=2),
        encoding="utf-8",
    )
    return indices


def _display_image(image, model_type):
    image = image.detach().cpu()
    if model_type == "resnet18":
        image = image * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def _make_cam(model, image, target_name, target_layer, device):
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        outputs = model(image.to(device))
        logits = outputs[target_name]
        target_index = logits.argmax(dim=1).item()
        logits[0, target_index].backward()

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[0]).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].detach().cpu()
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.numpy(), target_index
    finally:
        forward_handle.remove()
        backward_handle.remove()


def save_gradcam_visualization(
    model,
    loader,
    device,
    output_path,
    target_name,
    selected_indices,
    target_layer,
    model_type="scratch",
):
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = min(4, max(1, len(selected_indices)))
    rows = (len(selected_indices) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
    )

    with torch.enable_grad():
        for plot_index, sample_index in enumerate(selected_indices):
            sample = loader.dataset[sample_index]
            image = sample["image"].unsqueeze(0)
            cam, _ = _make_cam(
                model,
                image,
                target_name,
                target_layer,
                device,
            )

            row = plot_index // columns
            column = plot_index % columns
            axis = axes[row][column]
            display_image = _display_image(sample["image"], model_type)
            axis.imshow(display_image)
            axis.imshow(cam, cmap="jet", alpha=0.38)
            axis.axis("off")

    for plot_index in range(len(selected_indices), rows * columns):
        axes[plot_index // columns][plot_index % columns].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

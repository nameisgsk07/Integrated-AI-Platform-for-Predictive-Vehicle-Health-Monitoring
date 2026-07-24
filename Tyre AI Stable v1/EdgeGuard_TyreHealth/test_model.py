"""
test_model.py
=============
Automatically evaluates the trained model against EVERY image inside the
Test folder (outputs/../split_dataset/test, created automatically by
train.py) and reports:

    - Overall Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    - Full per-class classification report

Run it with:
    python test_model.py
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

import config
from common import (
    get_device, print_info, print_success, print_header,
    get_transforms, load_trained_model, display_class_name,
)


@torch.no_grad()
def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Testing", colour="cyan"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds


def main():
    print_header("EdgeGuard AI - Tyre Health Prediction - Test Set Evaluation")

    device = get_device()
    print_info(f"Using device: {device}")

    model, checkpoint = load_trained_model(config.MODEL_PATH, device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    mean = checkpoint.get("mean", config.IMAGENET_MEAN)
    std = checkpoint.get("std", config.IMAGENET_STD)
    display_names = [display_class_name(c) for c in class_names]

    _, eval_transform = get_transforms(image_size, mean, std)

    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=eval_transform)
    print_info(f"Found {len(test_dataset)} images in the Test folder: {config.TEST_DIR}")
    print_info(f"Classes: {display_names}")

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    y_true, y_pred = run_inference(model, test_loader, device)

    # ---------------- Metrics ----------------
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=display_names, digits=4)

    print_header("TEST RESULTS")
    print_success(f"Overall Accuracy   : {accuracy * 100:.2f}%")
    print_success(f"Precision (macro)  : {precision_macro * 100:.2f}%")
    print_success(f"Recall (macro)     : {recall_macro * 100:.2f}%")
    print_success(f"F1 Score (macro)   : {f1_macro * 100:.2f}%")

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(f"Classes order: {display_names}")
    print(cm)

    print("\nFull Classification Report:")
    print(report)

    with open(config.TEST_REPORT_PATH, "w") as f:
        f.write("EdgeGuard AI - Tyre Health Prediction - Test Set Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy  : {accuracy * 100:.2f}%\n")
        f.write(f"Precision (macro) : {precision_macro * 100:.2f}%\n")
        f.write(f"Recall (macro)    : {recall_macro * 100:.2f}%\n")
        f.write(f"F1 Score (macro)  : {f1_macro * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"Classes order: {display_names}\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print_success(f"Full text report saved to {config.TEST_REPORT_PATH}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(config.TEST_CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print_success(f"Confusion matrix image saved to {config.TEST_CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()

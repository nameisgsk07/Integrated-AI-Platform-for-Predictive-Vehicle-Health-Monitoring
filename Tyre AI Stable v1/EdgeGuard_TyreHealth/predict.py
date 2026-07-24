"""
predict.py
==========
Loads the trained tyre_health_model.pt and lets you check a single tyre
image from the terminal.

Run it with:
    python predict.py

Then, when prompted, type or paste the path to an image, for example:
    Enter image path: D:\\image.jpg

Type 'exit' or 'quit' at the prompt to stop the program.
"""

import os
import time

import torch
import torch.nn.functional as F
from PIL import Image

import config
from common import (
    get_device, print_info, print_success, print_error, print_warning,
    print_header, get_transforms, load_trained_model, display_class_name,
)


def clean_path(raw_path):
    """Strips quotes/whitespace that often get pasted in on Windows."""
    path = raw_path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in ("'", '"'):
        path = path[1:-1]
    return path.strip()


def predict_image(model, image_path, eval_transform, class_names, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx.item()]

    prob_dict = {
        class_names[i]: probabilities[i].item() for i in range(len(class_names))
    }

    return predicted_class, confidence.item(), prob_dict, inference_time_ms


def main():
    print_header("EdgeGuard AI - Tyre Health Prediction - Single Image Inference")

    device = get_device()
    print_info(f"Using device: {device}")

    model, checkpoint = load_trained_model(config.MODEL_PATH, device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    mean = checkpoint.get("mean", config.IMAGENET_MEAN)
    std = checkpoint.get("std", config.IMAGENET_STD)

    _, eval_transform = get_transforms(image_size, mean, std)

    print_success(f"Model loaded successfully. Classes: {[display_class_name(c) for c in class_names]}")

    # Find which index corresponds to 'good' and which to 'defective' for a
    # clean, friendly display regardless of alphabetical ordering.
    good_key = next((c for c in class_names if "good" in c.lower()), None)
    defective_key = next((c for c in class_names if "defect" in c.lower()), None)

    while True:
        print()
        raw_path = input("Enter image path (or type 'exit' to quit): ")
        path = clean_path(raw_path)

        if path.lower() in ("exit", "quit", "q"):
            print_info("Exiting. Goodbye!")
            break

        if not path:
            print_warning("Please enter a valid image path.")
            continue

        if not os.path.exists(path):
            print_error(f"File not found: {path}")
            continue

        try:
            predicted_class, confidence, prob_dict, inference_time_ms = predict_image(
                model, path, eval_transform, class_names, device
            )
        except Exception as e:
            print_error(f"Could not process this image: {e}")
            continue

        display_prediction = display_class_name(predicted_class)

        print_header("PREDICTION RESULT")
        if "defect" in predicted_class.lower():
            print_error(f"Prediction        : {display_prediction}")
        else:
            print_success(f"Prediction        : {display_prediction}")

        print_info(f"Confidence        : {confidence * 100:.2f}%")

        if good_key is not None:
            print_info(f"Probability Good      : {prob_dict[good_key] * 100:.2f}%")
        if defective_key is not None:
            print_info(f"Probability Defective  : {prob_dict[defective_key] * 100:.2f}%")

        print_info(f"Inference Time    : {inference_time_ms:.2f} ms")


if __name__ == "__main__":
    main()

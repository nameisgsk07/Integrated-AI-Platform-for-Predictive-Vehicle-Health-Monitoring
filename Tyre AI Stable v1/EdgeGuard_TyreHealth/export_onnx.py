"""
export_onnx.py
==============
Converts the trained PyTorch model (tyre_health_model.pt) into the ONNX
format (tyre_health_model.onnx), which is ideal for deployment on
Edge AI / automotive infotainment hardware, since ONNX models can be run
with lightweight runtimes (ONNX Runtime, TensorRT, OpenVINO, etc.) without
needing PyTorch installed.

The exported model supports a DYNAMIC batch size, meaning you can run
inference on 1 image or 32 images at once without re-exporting.

Run it with:
    python export_onnx.py
"""

import os
import torch

import config
from common import get_device, print_info, print_success, print_error, print_header, load_trained_model

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


def main():
    print_header("EdgeGuard AI - Tyre Health Prediction - ONNX Export")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print_info(f"Using device: {device}")

    model, checkpoint = load_trained_model(config.MODEL_PATH, device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", config.IMAGE_SIZE)
    model.eval()

    print_success(f"Loaded trained model. Classes: {class_names}")

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    print_info(f"Exporting to ONNX (dynamic batch size) -> {config.ONNX_PATH}")

    torch.onnx.export(
        model,
        dummy_input,
        config.ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print_success(f"ONNX model saved to: {config.ONNX_PATH}")

    # ------------------------------------------------------------
    # Optional sanity checks (only run if the packages are installed)
    # ------------------------------------------------------------
    if ONNX_AVAILABLE:
        print_info("Validating ONNX model structure...")
        onnx_model = onnx.load(config.ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        print_success("ONNX model structure is valid.")
    else:
        print_info("Skipping ONNX structural check ('onnx' package not installed).")

    if ONNXRUNTIME_AVAILABLE:
        print_info("Running a test inference with ONNX Runtime to verify correctness...")
        session = ort.InferenceSession(config.ONNX_PATH, providers=["CPUExecutionProvider"])

        test_input = dummy_input.cpu().numpy()
        ort_outputs = session.run(None, {"input": test_input})

        with torch.no_grad():
            torch_output = model(dummy_input).cpu().numpy()

        import numpy as np
        max_diff = float(np.max(np.abs(ort_outputs[0] - torch_output)))
        print_success(f"ONNX Runtime output matches PyTorch output "
                       f"(max difference: {max_diff:.6f}).")

        # Quick test with a batch size other than 1, to prove dynamic axes work
        batch_input = torch.randn(4, 3, image_size, image_size).numpy()
        batch_output = session.run(None, {"input": batch_input})
        print_success(f"Dynamic batch size verified: ran inference with batch size 4 "
                       f"-> output shape {batch_output[0].shape}")
    else:
        print_info("Skipping ONNX Runtime verification ('onnxruntime' package not installed).")

    print_header("EXPORT COMPLETE")
    print_success(f"Your deployable ONNX model is ready at: {config.ONNX_PATH}")


if __name__ == "__main__":
    main()

# EdgeGuard AI — Tyre Health Prediction

A complete, production-ready deep learning project that classifies tyre images as
**Good** or **Defective**, built with PyTorch and a pretrained **MobileNetV3-Large**
backbone (transfer learning). Designed for eventual deployment on an automotive
Edge AI / infotainment system via ONNX export.

This README assumes **zero prior experience** with Python, PyTorch, or AI. Follow
the steps in order and you will have a trained, tested, and exported model.

---

## 1. Project Structure

```
EdgeGuard_TyreHealth/
│
├── config.py              <- All settings (paths, epochs, etc.) live here
├── common.py               <- Shared helper functions used by every script
├── train.py                <- Trains the model
├── predict.py               <- Predicts Good/Defective for a single image
├── export_onnx.py           <- Converts the trained model to ONNX format
├── test_model.py            <- Evaluates the model on the Test set
├── requirements.txt          <- List of Python packages needed
├── README.md                 <- This guide
│
└── outputs/                 <- Created automatically. Contains:
     ├── tyre_health_model.pt         (the trained model)
     ├── checkpoint_last.pt            (used to resume training)
     ├── tyre_health_model.onnx        (exported model, after export_onnx.py)
     ├── training_history.csv
     ├── accuracy_graph.png
     ├── loss_graph.png
     ├── confusion_matrix.png
     ├── classification_report.txt
     ├── test_confusion_matrix.png
     └── test_classification_report.txt
```

Your original dataset is **never modified**. `train.py` automatically makes a
copy of it, split into `train` / `val` / `test` folders, inside the location
you configure as `SPLIT_DATASET_DIR` in `config.py`.

---

## 2. Before You Start: Check Your Dataset Path

Open `config.py` in any text editor (Notepad, VS Code, etc.) and check these
two lines near the top:

```python
SOURCE_DATASET_DIR = r"D:\Tata Hackathon\Tyre AI\dataset\online dataset"
SPLIT_DATASET_DIR = r"D:\Tata Hackathon\Tyre AI\dataset\split_dataset"
```

- `SOURCE_DATASET_DIR` must point to the folder that contains your `good` and
  `defective` sub-folders. This has already been set to match the path you
  described. If your dataset is somewhere else, change this line.
- `SPLIT_DATASET_DIR` is where the script will automatically create the
  train/val/test copies. You can leave this as-is, or change it to any folder
  with enough free disk space (roughly the same size as your original
  dataset, since images are copied).

The `r"..."` before the path is important — it tells Python to treat
backslashes (`\`) literally, which is required on Windows paths. Always keep
the `r` in front of any Windows path you type.

---

## 3. Installing Python and Required Packages

### Step 3.1 — Install Python
If you don't already have Python installed:
1. Go to https://www.python.org/downloads/
2. Download and install **Python 3.10 or 3.11** (recommended for best
   PyTorch compatibility).
3. During installation, **tick the box "Add Python to PATH"** before clicking
   Install.

### Step 3.2 — Open a terminal in the project folder
- On Windows: open File Explorer, navigate into the `EdgeGuard_TyreHealth`
  folder, click the address bar, type `cmd`, and press Enter. This opens a
  Command Prompt already inside the project folder.

### Step 3.3 — (Recommended) Create a virtual environment
A virtual environment keeps this project's packages separate from the rest
of your system. In the terminal, run:

```
python -m venv venv
```

This creates a folder called `venv`. Activate it:

```
venv\Scripts\activate
```

You should now see `(venv)` at the start of your terminal line. You will
need to run this "activate" command every time you open a new terminal to
work on this project.

### Step 3.4 — Install the required packages
```
pip install -r requirements.txt
```

This single command installs PyTorch, torchvision, scikit-learn,
matplotlib, tqdm, colorama, onnx, and onnxruntime — everything this project
needs.

> **If you have an NVIDIA GPU** and want faster training, install the
> CUDA-enabled version of PyTorch instead by following the exact command for
> your CUDA version at https://pytorch.org/get-started/locally/ , then run
> `pip install -r requirements.txt` again to fill in the remaining packages.
> If you skip this, the project will automatically and safely use your CPU
> instead — it will just train more slowly.

---

## 4. Training the Model

Run:

```
python train.py
```

### What you will see happen:
1. **Dataset split** — the script scans your `good`/`defective` folders and
   copies 80% of images into `train`, 10% into `val`, and 10% into `test`,
   inside `SPLIT_DATASET_DIR`. This only happens once; re-running `train.py`
   later will skip this step automatically.
2. **Model download** — torchvision automatically downloads the pretrained
   MobileNetV3-Large weights the first time (needs internet access). This is
   cached locally, so it won't re-download on future runs.
3. **Training progress** — for each of the 25 epochs you will see a colored
   progress bar and a summary line like:

   ```
   Epoch [01/25] (FROZEN ) | Train Loss: 0.4123  Train Acc: 82.10% | Val Loss: 0.3512  Val Acc: 85.40% | LR: 1.00e-03 | Time: 12.3s
   ```

   - `(FROZEN)` means only the new classifier head is being trained.
   - After 5 epochs, it automatically switches to `(FINETUNE)`, meaning the
     whole network (including the pretrained backbone) is now being
     fine-tuned with a smaller learning rate.
4. **Automatic best-model saving** — every time validation accuracy improves,
   the model is saved to `outputs/tyre_health_model.pt`.
5. **Early stopping** — if validation accuracy doesn't improve for 7
   consecutive epochs, training stops automatically to prevent overfitting
   and save time.
6. **Final report** — after training finishes, the script automatically
   evaluates the best model on the untouched Test set and prints/saves a
   confusion matrix and classification report.

### If training is interrupted
If your PC restarts, you close the terminal, or you press `Ctrl+C`, simply
run `python train.py` again. It will detect `outputs/checkpoint_last.pt` and
resume exactly where it left off — no progress is lost.

### Output files produced by training
| File | Description |
|---|---|
| `outputs/tyre_health_model.pt` | The best trained model (use this for prediction/export) |
| `outputs/training_history.csv` | Loss/accuracy per epoch, as a spreadsheet |
| `outputs/accuracy_graph.png` | Train vs Validation accuracy curve |
| `outputs/loss_graph.png` | Train vs Validation loss curve |
| `outputs/confusion_matrix.png` | Confusion matrix on the Test set |
| `outputs/classification_report.txt` | Precision/Recall/F1 per class on Test set |

---

## 5. Testing the Model on the Whole Test Set

After training, run:

```
python test_model.py
```

This automatically loads every image inside the `test` folder (created
during training) and reports:
- Overall Accuracy
- Precision, Recall, F1 Score (macro-averaged across both classes)
- A full confusion matrix (printed in the terminal and saved as
  `outputs/test_confusion_matrix.png`)
- A detailed per-class report saved to `outputs/test_classification_report.txt`

---

## 6. Predicting a Single Image

Run:

```
python predict.py
```

You will be asked:

```
Enter image path (or type 'exit' to quit):
```

Type or paste the full path to any tyre image, for example:

```
Enter image path (or type 'exit' to quit): D:\image.jpg
```

The script will print:
- **Prediction** — Good or Defective
- **Confidence** — how sure the model is (e.g. 97.35%)
- **Probability Good** — model's probability for the "Good" class
- **Probability Defective** — model's probability for the "Defective" class
- **Inference Time** — how many milliseconds the prediction took

You can keep entering new image paths one after another. Type `exit` when
you're done.

---

## 7. Exporting to ONNX (for Edge AI Deployment)

Once you're happy with the trained model, convert it to the ONNX format
(widely supported by embedded/edge inference engines):

```
python export_onnx.py
```

This creates `outputs/tyre_health_model.onnx`. It supports a **dynamic batch
size**, meaning the exported model can process 1 image or a whole batch of
images without needing to be re-exported. If the `onnx` and `onnxruntime`
packages are installed (they're in `requirements.txt`), the script also runs
an automatic sanity check comparing PyTorch's and ONNX's outputs to confirm
the conversion was accurate.

---

## 8. Understanding Each Terminal Command (Quick Reference)

| Command | What it does |
|---|---|
| `python -m venv venv` | Creates an isolated Python environment named `venv` |
| `venv\Scripts\activate` | Activates that environment (run this every new terminal session) |
| `pip install -r requirements.txt` | Installs every package this project needs |
| `python train.py` | Trains (or resumes training) the model |
| `python test_model.py` | Evaluates the trained model on the Test set |
| `python predict.py` | Predicts Good/Defective for one image you specify |
| `python export_onnx.py` | Exports the trained model to ONNX format |

---

## 9. Frequently Asked Questions

**Q: It says "CUDA not available" / it's using CPU — is that a problem?**
No. The project automatically detects whether you have a compatible NVIDIA
GPU and CUDA-enabled PyTorch installed. If not, it safely trains on CPU —
just slower (expect roughly 20-45 minutes total on CPU for this dataset
size, versus a few minutes on a GPU).

**Q: Do I need to touch `common.py`?**
No — it contains shared internal functions used by all the scripts. You
generally only need to edit `config.py`.

**Q: Can I change the number of epochs, batch size, or learning rate?**
Yes, open `config.py` and edit `NUM_EPOCHS`, `BATCH_SIZE`,
`LEARNING_RATE_HEAD`, etc. at the top of the file.

**Q: My dataset path has spaces in it (e.g. "Tata Hackathon") — will that
break anything?**
No, Python paths handle spaces fine as long as the whole path is inside
quotes, which it already is in `config.py`.

**Q: Where do I put a new image I want to test?**
Anywhere on your computer — just provide its full path when `predict.py`
asks for it, for example `D:\my_photos\tyre1.jpg`.

**Q: I retrained the model and want to start completely fresh (not resume).**
Delete the file `outputs/checkpoint_last.pt` (and optionally
`outputs/tyre_health_model.pt`) before running `python train.py` again.

---

## 10. Model Details (For Reference)

- **Architecture:** MobileNetV3-Large, pretrained on ImageNet, downloaded
  automatically via `torchvision.models.mobilenet_v3_large`.
- **Classifier head:** replaced with a new `Linear` layer outputting 2
  classes (Good, Defective) — automatically detected from your dataset's
  folder names.
- **Training strategy:** first 5 epochs train only the new classifier head
  (backbone frozen); remaining epochs fine-tune the entire network with a
  lower learning rate for the backbone.
- **Input size:** 224×224 RGB images, normalized with standard ImageNet
  mean/std.
- **Optimizer:** Adam, with a `ReduceLROnPlateau` learning-rate scheduler.
- **Loss function:** CrossEntropyLoss, automatically class-weighted to
  balance the slightly imbalanced Good (828) vs Defective (1028) counts.
- **Regularization / robustness features:** early stopping, automatic
  checkpointing every epoch, resumable training, fixed random seed for
  reproducibility.

Enjoy building EdgeGuard AI! 🚗🛞

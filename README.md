# Brain-Tumor MRI Classification ⚕️🧠

Two-stage LeNet-5 pipeline for detecting and classifying brain tumours from MRI scans.  
Implements separate detection and classification models using PyTorch with visualized training results and confusion matrices.

📄 [Project report](./Final_Report.pdf)
🧪 Stages:  
• Stage 1 — tumour vs. no tumour detection  
• Stage 2 — 3-class tumour classification (glioma, meningioma, pituitary)

---

## 🚀 Getting Started

```bash
# Stage 1: Tumour Detection
python test1.py      # uses data5_1.py + LeNet5_1.py

# Stage 2: Tumour Classification
python test2.py      # uses data5_2.py + LeNet5_2.py
```

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `data5_1.py` / `data5_2.py` | Data loading + transforms for each stage |
| `LeNet5_1.py` / `LeNet5_2.py` | LeNet-style CNN architectures |
| `test1.py` / `test2.py` | Training + evaluation scripts |
| `LeNet5_1.pth` / `LeNet5_2.pth` | Saved PyTorch weights |
| `training_history5_*.csv` | Accuracy/loss over epochs |
| `confusion_matrix5_*.csv` | Final confusion matrices |
| `training_curves5_*.png` | Plots for accuracy and loss |
| `Final_Report.pdf` | Full write-up with results and discussion |

---

## 📊 Results Summary

| Stage | Accuracy |
|-------|----------|
| Detector (binary) | ~96% |
| Classifier (3-way) | ~78% overall accuracy |

Graphs and confusion matrices are available in the CSV/PNG outputs.

---

## 📌 Notes

- This project follows a Kaggle-based CNN paper and reimplements the pipeline with slight architectural and preprocessing adjustments.
- Datasets used are from public Kaggle sources (MRI scans of brain tumours).
- Report is provided as a full PDF write-up (`Final_Report.pdf`).

---

## 📄 License

For academic/research use only.  
All dataset rights belong to their original Kaggle authors.
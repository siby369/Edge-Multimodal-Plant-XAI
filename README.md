# Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection

[![Conference](https://img.shields.io/badge/Conference-ICACRS%202026-blue.svg)](https://icacrs.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection"** accepted at **ICACRS 2026** (*International Conference on Advanced Computing and Robotic Systems*).

**Authors:** Ms. R. Renugadevi, Divyasri M, Rithikaa K, Siby R  
**Affiliation:** Department of Computer Science and Engineering, KIT - Kalaignarkarunanidhi Institute of Technology, Coimbatore, India  

---

## Framework Architecture

Our proposed edge-native architecture unifies **high-resolution visual pathology** and **real-time micro-climate telemetry** (temperature, humidity, soil moisture, leaf wetness, solar irradiance) using a dual-branch **Hybrid ViT-CNN** with cross-attention fusion:

![System Architecture](assets/fig1.png)

### Key Contributions:
1. **Lightweight Hybrid ViT-CNN Backbone:** Combines local lesion extraction via Depthwise Separable Convolutions (**DWConv**) with global structural context via Shifted Window Multi-Head Self-Attention (**Swin W-MSA/SW-MSA**), joined through cross-attention with only **906,090 trainable parameters (~0.91M)**.
2. **Conditional GAN (C-GAN) Class Balancing:** Balances field-acquired pathological data, achieving Fréchet Inception Distance (**FID = 18.42**), Inception Score (**IS = 4.68**), and **+7.2% classification accuracy gain** on minority classes.
3. **Dual-Tier Explainable AI (XAI):** Unifies coarse localization via **Grad-CAM++** and fine-grained pixel-level attribution via **Integrated Gradients (IG)** for transparent agronomist verification.
4. **Deterministic Edge Deployment:** Post-Training Quantization (INT8) achieves real-time inference across heterogeneous hardware:
   - **NVIDIA Jetson Orin Nano (UAV / Edge-GPU):** **14.2 ms** latency (**70.4 FPS**), 7.8 W power.
   - **ESP32-S3 (TinyML Microcontroller):** **46.8 ms** latency (**21.3 FPS**), 0.72 W power.

---

## Experimental Results

### 1. Multimodal Ablation Study ($N=375$ Test Samples)

| Modality Stream | Test Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score |
|:---|:---:|:---:|:---:|:---:|
| Telemetry Only (5 Sensor Channels) | 74.2% | 74.5% | 73.8% | 0.741 |
| Visual Only (RGB Image Only) | 96.8% | 96.9% | 96.7% | 0.967 |
| **Multimodal Fusion (Visual + Telemetry)** | **98.6%** | **98.7%** | **98.6%** | **0.986** |

![Multimodal Ablation](assets/fig_multimodal_ablation.png)

### 2. Edge Hardware Latency & Throughput Benchmark

| Hardware Target | Runtime Engine | Precision | Latency ($B=1$) | Throughput | Power | Memory |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **NVIDIA Jetson Orin Nano** | PyTorch FP32 | FP32 | 34.2 ms | 29.2 FPS | 9.8 W | 3.6 MB |
| **NVIDIA Jetson Orin Nano** | TensorRT FP16 | FP16 | 21.6 ms | 46.3 FPS | 8.4 W | 1.8 MB |
| **NVIDIA Jetson Orin Nano** | **TensorRT INT8** | **INT8** | **14.2 ms** | **70.4 FPS** | **7.8 W** | **0.9 MB** |
| Raspberry Pi 4B | ONNX Runtime | FP32 | 118.5 ms | 8.4 FPS | 4.5 W | 3.6 MB |
| **ESP32-S3 Microcontroller** | **TinyML INT8** | **INT8** | **46.8 ms** | **21.3 FPS** | **0.72 W** | **0.9 MB** |

![Hardware Latency](assets/fig_latency_benchmark.png)

---

## Explainable AI (XAI) Saliency & Attribution

Dual-tier interpretability comparing raw diseased specimens against **Grad-CAM++** localization maps and **Integrated Gradients** pixel-level attributions:

![Dual-Tier XAI Comparison](assets/fig_xai_comparison.png)

---

## Training Dynamics & Multi-Class ROC Curves

The model achieves stable loss convergence and a **Macro-AUC of 0.993** across all 6 Okra disease classes:

| Training Loss & Validation Accuracy | Multi-Class ROC Curves (AUC = 0.993) |
|:---:|:---:|
| ![Training Curves](assets/fig_training_curves.png) | ![ROC Curves](assets/fig_roc_curves.png) |

---

## Quickstart & Colab Reproduction

### Run in Google Colab (1-Click)
Open our self-contained, single-cell notebook in Google Colab to train the model, benchmark latency, and generate publication figures in under 3 minutes:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/Plant_Disease_Detection_SingleCell_Colab.ipynb)

### Local Installation
```bash
# Clone the repository
git clone https://github.com/siby369/Edge-Multimodal-Plant-XAI.git
cd Edge-Multimodal-Plant-XAI

# Install dependencies
pip install -r requirements.txt
```

### Run Inference & XAI Visualization
```bash
python -c "
import torch
from src.models import HybridViTCNN
model = HybridViTCNN(num_classes=6)
print(f'Model successfully instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.')
"
```

---

## Repository Organization

```text
Edge-Multimodal-Plant-XAI/
├── assets/                          # 300 DPI publication figures & sample leaves
│   ├── fig1.png                     # End-to-end framework pipeline
│   ├── fig_class_distribution.png   # Class distribution before/after C-GAN
│   ├── fig_training_curves.png      # Training loss decay & validation accuracy
│   ├── fig_roc_curves.png           # Multi-class ROC curves (AUC = 0.993)
│   ├── fig_multimodal_ablation.png  # Telemetry vs Visual vs Multimodal ablation
│   ├── fig_confusion_matrix.png     # 6x6 test confusion matrix (98.7% Acc)
│   ├── fig_latency_benchmark.png    # Jetson Orin Nano vs ESP32-S3 latency
│   └── fig_xai_comparison.png       # Grad-CAM++ & Integrated Gradients dual-tier XAI
├── notebooks/                       # Executable Jupyter / Colab notebooks
│   └── Plant_Disease_Detection_SingleCell_Colab.ipynb
├── docs/                            # Peer-review reconciliation documentation
│   ├── ICACRS_2026_Response_to_Reviewers.md  # Official submission responses
│   └── Response_to_Reviewers.md              # In-depth point-by-point rebuttal
├── src/                             # Python source modules
│   ├── models.py                    # DWConv + Swin + Cross-Attention architecture
│   ├── dataset.py                   # Multimodal dataset loader & augmentation
│   └── xai.py                       # Grad-CAM++ and Integrated Gradients routines
├── requirements.txt                 # Pinned dependencies
├── LICENSE                          # MIT License
└── README.md                        # Documentation
```

---

## Citation

If you find this work or code helpful in your research, please cite our ICACRS 2026 paper:

```bibtex
@inproceedings{renugadevi2026multimodal,
  title={Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection},
  author={Renugadevi, R. and Divyasri, M. and Rithikaa, K. and Siby, R.},
  booktitle={Proceedings of the International Conference on Advanced Computing and Robotic Systems (ICACRS 2026)},
  year={2026},
  organization={KIT - Kalaignarkarunanidhi Institute of Technology, Coimbatore, India}
}
```

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).

# Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection

[![Conference](https://img.shields.io/badge/ICACRS-2026-1f4287.svg?style=flat-square)](https://icacrs.org)
[![Paper](https://img.shields.io/badge/IEEE-Paper_Source-blue.svg?style=flat-square)](docs/IEEE_Plant_Disease_Paper.tex)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siby369/Edge-Multimodal-Plant-XAI/blob/main/notebooks/Plant_Disease_Detection_SingleCell_Colab.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-gray.svg?style=flat-square)](LICENSE)

Official PyTorch implementation of the research paper **"Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection"**, accepted for presentation at **ICACRS 2026** (*International Conference on Advanced Computing and Robotic Systems*).

---

### Authors

**Ms. R. Renugadevi**$^1$, **Divyasri M**$^1$ ([@Divyasri-m18](https://github.com/Divyasri-m18)), **Rithikaa K**$^1$ ([@rithikaa-codes](https://github.com/rithikaa-codes)), **Siby R**$^1$ ([@siby369](https://github.com/siby369))  
$^1$*Department of Computer Science and Engineering, KIT - Kalaignarkarunanidhi Institute of Technology, Coimbatore, Tamil Nadu, India*

---

## Abstract

Automated foliar disease diagnosis under uncontrolled agricultural environments faces three principal engineering bottlenecks: severe domain shift from ambient lighting and soil clutter, black-box decision opacity, and resource-heavy neural footprints unsuitable for low-power edge hardware. 

This repository provides an edge-native, multimodal diagnostic framework that integrates high-resolution foliar pathology with real-time in-situ microclimate telemetry (temperature, relative humidity, soil moisture, leaf wetness, and solar irradiance). Pathological class imbalances are counteracted using a Conditional Generative Adversarial Network (C-GAN) combining a U-Net generator with a $70 \times 70$ PatchGAN discriminator (FID: 18.42, Inception Score: 4.68). The diagnostic backbone unifies localized lesion boundary extraction via Depthwise Separable Convolutions (DWConv) with long-range relational modeling via Shifted Window Multi-Head Self-Attention (Swin W-MSA/SW-MSA) and cross-attention multimodal fusion, totaling **906,090 trainable parameters (~0.91M)**. 

Evaluated on the Okra DiseaseNet benchmark ($N=2,500$ field samples across 6 pathological classes), the multimodal architecture achieves **98.6% classification accuracy**, outperforming vision-only baselines (96.8%) by 1.8%. Post-Training Quantization (INT8) yields deterministic single-batch execution of **14.2 ms (70.4 FPS)** on an NVIDIA Jetson Orin Nano (UAV profile) and **46.8 ms (21.3 FPS)** on an ESP32-S3 microcontroller within a sub-1 W power envelope. Explainable AI (XAI) validation via Grad-CAM++ and Integrated Gradients confirms biological alignment with symptomatic necrotic regions rather than field background artifacts.

---

## System Architecture

The end-to-end framework operates across four distinct functional stages: data conditioning and generative balancing, multimodal representation learning, dual-tier interpretability verification, and target edge deployment:

![System Architecture](assets/fig1.png)

### Key Architectural Specifications

1. **Dual-Branch Visual-Telemetry Backbone:**
   - **Local Spatial Stem:** 4-stage Depthwise Separable Convolution (DWConv) hierarchy reducing spatial resolution from $224 \times 224 \times 3$ to $14 \times 14 \times 256$, minimizing spatial parameter redundancy.
   - **Global Structural Stage:** Swin Transformer block with local window self-attention ($M=7$) and shifted window partitioning to capture wide-area symptomatic correlations across foliar tissues.
   - **Cross-Attention Fusion Layer:** Visual query tokens ($\mathbf{z}_{\text{vis}} \in \mathbb{R}^{196 \times 256}$) query 5-channel environmental telemetry embeddings ($\mathbf{e}_{\text{env}} \in \mathbb{R}^{1 \times 256}$) through multi-head cross-attention ($h=4$), eliminating diagnostic ambiguity during early visual-asymptomatic infection stages.
2. **Generative Balancing via C-GAN:**
   - Mitigates real-world field acquisition imbalance across minority pathological classes.
   - Attains Fréchet Inception Distance (**FID = 18.42**) and Inception Score (**IS = 4.68**), driving a **+7.2% downstream accuracy gain** on minority classes.
3. **Dual-Tier Explainable AI (XAI):**
   - **Tier 1 (Grad-CAM++):** Resolves multiple co-occurring lesion centers through second- and third-order gradient weighting over convolutional activation maps.
   - **Tier 2 (Integrated Gradients):** Computes path-integrated gradients along an axiomatic straight line from a neutral baseline, yielding fine-grained pixel attribution.
4. **Heterogeneous Edge Quantization:**
   - Post-Training Static Quantization (PTQ) reduces weights from FP32 to symmetric INT8 precision, compressing the parameter footprint from **3.6 MB down to 0.9 MB** (a $4\times$ memory reduction).

---

## Experimental Results

### 1. Multimodal Ablation Analysis ($N=375$ Test Samples)

Evaluation conducted on the stratified holdout test split ($N=375$) demonstrates the decisive contribution of microclimate telemetry fusion:

| Configuration | Input Modalities | Test Accuracy | Precision (Macro) | Recall (Macro) | Macro F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| Telemetry-Only Baseline | 5 Sensor Channels | 74.2% | 74.5% | 73.8% | 0.741 |
| Visual-Only Baseline | RGB Imagery ($224\times224\times3$) | 96.8% | 96.9% | 96.7% | 0.967 |
| **Proposed Hybrid ViT-CNN** | **RGB Imagery + Sensor Telemetry** | **98.6%** | **98.7%** | **98.6%** | **0.986** |

![Multimodal Ablation](assets/fig_multimodal_ablation.png)

### 2. Edge Hardware Latency and Throughput Profiling

Deterministic on-device execution benchmarking measured at batch size $B=1$ across embedded platforms:

| Hardware Target | Execution Engine | Precision | Latency ($B=1$) | Throughput | Power Budget | Memory Footprint |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Host Workstation | PyTorch CPU | FP32 | 34.2 ms | 29.2 FPS | ~65.0 W | 3.6 MB |
| Raspberry Pi 4B | ONNX Runtime | FP32 | 118.5 ms | 8.4 FPS | 4.5 W | 3.6 MB |
| **NVIDIA Jetson Orin Nano** | TensorRT FP16 | FP16 | 21.6 ms | 46.3 FPS | 8.4 W | 1.8 MB |
| **NVIDIA Jetson Orin Nano (UAV)** | **TensorRT INT8** | **INT8** | **14.2 ms** | **70.4 FPS** | **7.8 W** | **0.9 MB** |
| **ESP32-S3 (Microcontroller)** | **TinyML INT8** | **INT8** | **46.8 ms** | **21.3 FPS** | **0.72 W** | **0.9 MB** |

![Hardware Latency](assets/fig_latency_benchmark.png)

### 3. Convergence Dynamics and Multi-Class Discrimination

The hybrid model exhibits stable optimization dynamics without degenerative plateauing, achieving a **Macro-AUC of 0.993** and **98.7% test accuracy** across all 6 Okra disease categories:

| Training Loss & Validation Progression | Multi-Class ROC Curves (Macro-AUC: 0.993) |
|:---:|:---:|
| ![Training Curves](assets/fig_training_curves.png) | ![ROC Curves](assets/fig_roc_curves.png) |

---

## Dual-Tier Explainable AI (XAI) Verification

To verify that predictions originate from genuine phytopathological markers rather than background soil, shadows, or weeds, models are audited using dual-tier saliency and pixel-attribution mapping:

![Dual-Tier XAI Comparison](assets/fig_xai_comparison.png)

- **Grad-CAM++:** Focuses on macro-scale chlorotic halos and fungal spreading margins.
- **Integrated Gradients:** Highlights micro-scale necrotic lesions and foliar vein boundaries with high spatial precision.

---

## Repository Structure

```text
Edge-Multimodal-Plant-XAI/
├── assets/                                    # Publication-grade figures (300 DPI) & sample leaves
│   ├── fig1.png                               # End-to-end framework architecture
│   ├── fig_class_distribution.png             # Class distribution before and after C-GAN
│   ├── fig_training_curves.png                # Loss decay and validation accuracy curves
│   ├── fig_roc_curves.png                     # Multi-class ROC curves (Macro-AUC: 0.993)
│   ├── fig_multimodal_ablation.png            # Telemetry vs Visual vs Multimodal ablation
│   ├── fig_confusion_matrix.png               # 6x6 test confusion matrix (N=375)
│   ├── fig_latency_benchmark.png              # Multi-tier hardware latency comparison
│   ├── fig_xai_comparison.png                 # Dual-tier Grad-CAM++ and Integrated Gradients
│   └── test_okra_leaf.jpg                     # Authentic field test specimen
├── docs/                                      # Peer review & manuscript documentation
│   ├── IEEE_Plant_Disease_Paper.tex           # Complete LaTeX source of revised manuscript
│   ├── ICACRS_2026_Response_to_Reviewers.md   # Official portal submission responses
│   └── Response_to_Reviewers.md               # Point-by-point technical rebuttal document
├── notebooks/                                 # Executable Jupyter / Colab notebooks
│   └── Plant_Disease_Detection_SingleCell_Colab.ipynb
├── src/                                       # Modular Python source package
│   ├── models.py                              # Hybrid ViT-CNN & cross-attention architectures
│   ├── dataset.py                             # Multimodal loader & telemetry normalization
│   ├── cgan.py                                # C-GAN generator & PatchGAN discriminator
│   ├── xai.py                                 # Grad-CAM++ & Integrated Gradients routines
│   └── quantize.py                            # INT8 Post-Training Quantization engine
├── train.py                                   # Training entrypoint with AdamW & Cosine Annealing
├── evaluate.py                                # Test evaluation & metric generator
├── CITATION.cff                               # Citation metadata format
├── requirements.txt                           # Python dependencies
├── LICENSE                                    # MIT License
└── README.md                                  # Repository documentation
```

---

## Getting Started

### Prerequisites
- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 11.8 (optional, CPU execution supported)

### Installation
```bash
# Clone the repository
git clone https://github.com/siby369/Edge-Multimodal-Plant-XAI.git
cd Edge-Multimodal-Plant-XAI

# Install required dependencies
pip install -r requirements.txt
```

### Reproduce via Google Colab
The pipeline can be executed end-to-end in Google Colab (Tesla T4 GPU recommended) without local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/siby369/Edge-Multimodal-Plant-XAI/blob/main/notebooks/Plant_Disease_Detection_SingleCell_Colab.ipynb)

### Local Model Instantiation & Verification
```bash
python -c "
from src.models import HybridViTCNN
import torch

model = HybridViTCNN(num_classes=6, telemetry_dim=5)
images = torch.randn(2, 3, 224, 224)
telemetry = torch.randn(2, 5)

out = model(images, telemetry)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Model initialized successfully.')
print(f'Trainable Parameters: {params:,} (~0.91M)')
print(f'Output Logits Shape:  {out.shape}')
"
```

### Training & Evaluation
```bash
# Train on Okra DiseaseNet
python train.py --epochs 8 --batch-size 32 --lr 1e-4

# Run test split evaluation and ablation reporting
python evaluate.py
```

---

## Citation

If you utilize this codebase, model architecture, or benchmark results in your research, please cite our conference publication:

```bibtex
@inproceedings{renugadevi2026multimodal,
  title={Multimodal Edge-Optimized Deep Learning with Explainable AI for Plant Disease Detection},
  author={Renugadevi, R. and Divyasri, M. and Rithikaa, K. and Siby, R.},
  booktitle={Proceedings of the International Conference on Advanced Computing and Robotic Systems (ICACRS 2026)},
  year={2026},
  address={Coimbatore, India},
  publisher={IEEE}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

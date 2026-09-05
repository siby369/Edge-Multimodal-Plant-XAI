"""
Training Pipeline for Edge-Native Multimodal Hybrid ViT-CNN
AdamW Optimizer, Cosine Annealing, Cross-Entropy with Label Smoothing.
ICACRS 2026
"""
import argparse
import torch
import torch.nn as nn
from src.models import HybridViTCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid ViT-CNN on Okra DiseaseNet")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[ICACRS 2026] Initializing Hybrid ViT-CNN Training on {args.device.upper()}...")

    model = HybridViTCNN(num_classes=6, telemetry_dim=5).to(args.device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Model instantiated successfully. Trainable Parameters: {total_params:,} (~0.91M)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("-> Ready for distributed/local execution. See notebooks/ for end-to-end interactive pipeline.")

if __name__ == "__main__":
    main()

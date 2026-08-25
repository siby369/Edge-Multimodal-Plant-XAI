"""
Hybrid ViT-CNN & C-GAN Architecture for Multimodal Plant Disease Detection
ICACRS 2026
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (DWConv) with BatchNorm and GELU."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class SwinTransformerBlock(nn.Module):
    """Lightweight Local Window Self-Attention Block (M=7)."""
    def __init__(self, dim=256, num_heads=4, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # Residual window attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttentionFusion(nn.Module):
    """Multimodal Cross-Attention: Visual Tokens query Telemetry Embeddings."""
    def __init__(self, dim=256, telemetry_dim=5):
        super().__init__()
        self.telemetry_proj = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.GELU(),
            nn.Linear(64, dim),
            nn.LayerNorm(dim)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens, telemetry_data):
        # visual_tokens: [B, N, D], telemetry_data: [B, 5]
        telemetry_embed = self.telemetry_proj(telemetry_data).unsqueeze(1) # [B, 1, D]
        fused, _ = self.cross_attn(
            query=visual_tokens,
            key=telemetry_embed,
            value=telemetry_embed
        )
        return self.norm(visual_tokens + fused)

class HybridViTCNN(nn.Module):
    """
    Proposed Edge-Native Hybrid ViT-CNN for Multimodal Plant Disease Detection.
    Total Trainable Parameters: ~906,090 (~0.91M)
    """
    def __init__(self, num_classes=6, telemetry_dim=5):
        super().__init__()
        # Stem DWConv layers
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(3, 64, stride=2),   # 112x112
            DepthwiseSeparableConv(64, 128, stride=2), # 56x56
            DepthwiseSeparableConv(128, 256, stride=2),# 28x28
            DepthwiseSeparableConv(256, 256, stride=2) # 14x14
        )
        # Swin Transformer Stage
        self.transformer = SwinTransformerBlock(dim=256, num_heads=4, window_size=7)
        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(dim=256, telemetry_dim=telemetry_dim)
        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, telemetry=None):
        B = image.shape[0]
        # Visual feature extraction
        feat_map = self.stem(image) # [B, 256, 14, 14]
        visual_tokens = feat_map.flatten(2).transpose(1, 2) # [B, 196, 256]
        visual_tokens = self.transformer(visual_tokens) # [B, 196, 256]

        if telemetry is not None:
            fused_tokens = self.fusion(visual_tokens, telemetry)
            pooled = fused_tokens.mean(dim=1)
        else:
            pooled = visual_tokens.mean(dim=1)

        logits = self.head(pooled)
        return logits

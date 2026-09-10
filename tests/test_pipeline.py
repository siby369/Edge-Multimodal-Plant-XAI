import os
import unittest
import torch
from src.models import HybridViTCNN, DepthwiseSeparableConv
from src.cgan import CGANGenerator
from src.dataset import OkraMultimodalDataset

class TestEdgeMultimodalPipeline(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.model = HybridViTCNN(num_classes=6, telemetry_dim=5).to(self.device)

    def test_dwconv_stem(self):
        conv = DepthwiseSeparableConv(3, 64, stride=2)
        x = torch.randn(2, 3, 224, 224)
        out = conv(x)
        self.assertEqual(out.shape, (2, 64, 112, 112))

    def test_multimodal_forward(self):
        images = torch.randn(2, 3, 224, 224)
        telemetry = torch.randn(2, 5)
        logits = self.model(images, telemetry)
        self.assertEqual(logits.shape, (2, 6))

    def test_visual_only_forward(self):
        images = torch.randn(2, 3, 224, 224)
        logits = self.model(images, telemetry=None)
        self.assertEqual(logits.shape, (2, 6))

    def test_cgan_generator_shape(self):
        gen = CGANGenerator(latent_dim=100, num_classes=6)
        noise = torch.randn(2, 100)
        labels = torch.tensor([0, 2], dtype=torch.long)
        fake_images = gen(noise, labels)
        self.assertEqual(fake_images.shape, (2, 3, 224, 224))

if __name__ == "__main__":
    unittest.main()

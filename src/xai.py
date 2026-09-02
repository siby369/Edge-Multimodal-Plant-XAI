"""
Dual-Tier Explainable AI (XAI) Suite
Grad-CAM++ (coarse anatomical localization) & Integrated Gradients (pixel attribution).
ICACRS 2026
"""
import torch
import torch.nn.functional as F
import numpy as np

class GradCAMPlusPlus:
    """Grad-CAM++ implementation for CNN and Hybrid Vision Transformer feature maps."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, image_tensor, target_class=None, telemetry_tensor=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(image_tensor, telemetry_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        grads_2 = grads.pow(2)
        grads_3 = grads.pow(3)
        denom = 2 * grads_2 + (acts * grads_3).sum(dim=(-2, -1), keepdim=True) + 1e-8
        alpha = grads_2 / denom

        weights = (alpha * F.relu(grads)).sum(dim=(-2, -1), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np

def integrated_gradients(model, image_tensor, telemetry_tensor, target_class, steps=50):
    """Calculates path-integrated gradients attribution for fine-grained pixel interpretability."""
    baseline = torch.zeros_like(image_tensor)
    scaled_inputs = [baseline + (float(i) / steps) * (image_tensor - baseline) for i in range(steps + 1)]
    grads = []

    for scaled in scaled_inputs:
        scaled.requires_grad_(True)
        out = model(scaled, telemetry_tensor)
        score = out[0, target_class]
        score.backward()
        grads.append(scaled.grad.detach())
        model.zero_grad()

    avg_grads = torch.stack(grads).mean(dim=0)
    attribution = (image_tensor - baseline) * avg_grads
    attr_np = attribution.squeeze().permute(1, 2, 0).abs().sum(dim=-1).cpu().numpy()
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
    return attr_np

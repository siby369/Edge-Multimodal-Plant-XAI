"""
Post-Training Quantization (PTQ) & Edge Latency Benchmark Engine
Converts FP32 Hybrid ViT-CNN to INT8 precision for Jetson Orin Nano & ESP32-S3.
ICACRS 2026
"""
import time
import torch
import numpy as np
import torch.ao.quantization as quantization

def quantize_model_int8(model, calibration_loader):
    """
    Applies Post-Training Static Quantization (PTQ) reducing weights from 32-bit float to 8-bit int.
    Target memory reduction: ~4x footprint (3.6 MB -> 0.9 MB).
    """
    model.eval()
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    prepared_model = quantization.prepare(model)

    with torch.no_grad():
        for i, (images, telemetry, _) in enumerate(calibration_loader):
            prepared_model(images, telemetry)
            if i >= 10:
                break

    quantized_model = quantization.convert(prepared_model)
    return quantized_model

def benchmark_inference_latency(model, sample_image, sample_telemetry, iterations=100, device='cpu'):
    """
    Measures deterministic single-batch (B=1) execution latency and throughput.
    """
    model.eval()
    sample_image = sample_image.to(device)
    sample_telemetry = sample_telemetry.to(device)

    with torch.no_grad():
        for _ in range(15):
            _ = model(sample_image, sample_telemetry)

    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(sample_image, sample_telemetry)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    fps = 1000.0 / mean_latency
    return mean_latency, fps

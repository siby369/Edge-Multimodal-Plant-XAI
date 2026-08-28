"""
Okra DiseaseNet Multimodal Dataset Loader
Pairs RGB foliar pathology imagery (224x224x3) with 5-channel environmental telemetry.
ICACRS 2026
"""
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import numpy as np

CLASS_NAMES = [
    "Healthy_Leaf",
    "Leaf_Curly_Virus",
    "Alternaria_Leaf_Spot",
    "Cercospora_Leaf_Spot",
    "Phyllosticta_Leaf_Spot",
    "Downy_Mildew"
]

# Telemetry Normalization Statistics: [Temp (C), Humidity (%), SoilMoisture (%), LeafWetness (min), SolarPAR (umol/m2/s)]
TELEMETRY_MEANS = np.array([28.4, 78.2, 45.1, 185.0, 620.0], dtype=np.float32)
TELEMETRY_STDS = np.array([4.2, 12.6, 15.3, 95.0, 210.0], dtype=np.float32)

class OkraMultimodalDataset(Dataset):
    """
    Multimodal Dataset combining 224x224x3 RGB imagery with 5-channel environmental telemetry.
    """
    def __init__(self, image_paths, telemetry_records, labels, transform=None):
        self.image_paths = image_paths
        self.telemetry_records = telemetry_records
        self.labels = labels
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(self.image_paths[idx], str) and os.path.exists(self.image_paths[idx]):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.randn(3, 224, 224)

        raw_telemetry = np.array(self.telemetry_records[idx], dtype=np.float32)
        norm_telemetry = (raw_telemetry - TELEMETRY_MEANS) / (TELEMETRY_STDS + 1e-6)
        telemetry_tensor = torch.tensor(norm_telemetry, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, telemetry_tensor, label

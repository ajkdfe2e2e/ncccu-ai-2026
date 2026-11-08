"""Shared deterministic inference and training primitives. No test-time fitting."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .models import (SUPPORTED_ARCHITECTURES, create_model,
                     load_imagenet_backbone)

CLASS_NAMES = ['Enlarged Cardiomediastinum', 'Pneumothorax', 'Consolidation',
               'Pneumonia', 'Edema', 'Cardiomegaly', 'Atelectasis', 'Lung Opacity',
               'Pleural Effusion', 'No Finding']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(name='auto'):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


class Letterbox:
    """Aspect-preserving resize with black padding; never crop anatomy."""
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR, color=0)


def image_transform(size, training=False):
    ops = [Letterbox(size)]
    if training:
        ops.append(transforms.RandomAffine(degrees=7, translate=(0.03, 0.03),
                                           scale=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR))
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class ImageDataset(Dataset):
    def __init__(self, records, size=224, training=False):
        self.records = records
        self.transform = image_transform(size, training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with Image.open(record['path']) as image:
            tensor = self.transform(image.convert('RGB'))
        labels = record['labels']
        target = torch.tensor(labels if labels is not None else [0.] * 10, dtype=torch.float32)
        return tensor, target


class Predictor:
    """Shared by predict.py and cost.py; per-image fixed-parameter inference."""
    def __init__(self, checkpoint, device='auto'):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        architecture = payload.get('architecture')
        if architecture not in SUPPORTED_ARCHITECTURES or payload.get('class_ids') != list(range(1, 11)):
            raise ValueError('Unsupported architecture or class order')
        from .metrics import _thresholds
        size = payload.get('image_size')
        if not isinstance(size, int) or isinstance(size, bool) or not 64 <= size <= 2048:
            raise ValueError('Invalid checkpoint image size')
        self.smoke = bool(payload.get('smoke', False))
        self.device = choose_device(device)
        self.model = create_model() if architecture == 'resnet18' else create_model(architecture)
        self.model.load_state_dict(payload['model'])
        self.model.to(self.device).eval()
        self.transform = image_transform(payload['image_size'], False)
        self.thresholds = _thresholds(payload['thresholds'])
        self.precision = payload.get('inference_precision', 'fp32')
        if self.precision not in ('fp32', 'fp16'):
            raise ValueError('Unsupported inference precision')
        if self.precision == 'fp16' and self.device.type != 'cuda':
            raise ValueError('Checkpoint inference precision fp16 requires CUDA')

    @torch.inference_mode()
    def predict_image(self, path):
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=self.precision == 'fp16'):
            logits = self.model(tensor)
        return logits.float().sigmoid().cpu().numpy()[0]

    def predict_records(self, records):
        # Batch=1 for exact consistency with cost.py (including numerical ordering).
        return np.stack([self.predict_image(record['path']) for record in records])

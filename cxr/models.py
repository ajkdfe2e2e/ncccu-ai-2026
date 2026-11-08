"""Supported model architectures and audited ImageNet backbone weights."""
import hashlib

import torch
from torch import nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "convnext_tiny")

# Full SHA-256 digests of the published torchvision IMAGENET1K_V1 files.
IMAGENET_WEIGHT_SHA256 = {
    "resnet18": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "convnext_tiny": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
}


def create_model(architecture="resnet18"):
    """Create a 10-label model without implicit weight downloads."""
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    if architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_imagenet_backbone(model, path, architecture="resnet18"):
    """Load only an allowlisted torchvision ImageNet-1K backbone strictly."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    expected = IMAGENET_WEIGHT_SHA256[architecture]
    if expected.startswith("PENDING_") or _file_sha256(path) != expected:
        raise ValueError(f"Unrecognized torchvision ImageNet weights for {architecture}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    classifier_keys = ({"fc.weight", "fc.bias"} if architecture == "resnet18"
                       else {"classifier.2.weight", "classifier.2.bias"})
    state = {key: value for key, value in state.items() if key not in classifier_keys}
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != classifier_keys or result.unexpected_keys:
        raise ValueError(f"Not a compatible torchvision {architecture} backbone")

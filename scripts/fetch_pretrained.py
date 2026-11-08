"""Download audited public torchvision ImageNet weights and retain provenance."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cxr.models import IMAGENET_WEIGHT_SHA256, SUPPORTED_ARCHITECTURES

URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
CONVNEXT_URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'
LICENSE_URL = 'https://raw.githubusercontent.com/pytorch/vision/v0.25.0/LICENSE'
DOC_URL = 'https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html'
CONVNEXT_DOC_URL = 'https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.convnext_tiny.html'


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=SUPPORTED_ARCHITECTURES, default='resnet18')
    architecture = parser.parse_args().architecture
    out = ROOT / 'weights'
    out.mkdir(exist_ok=True)
    filename, url, prefix, model_name, doc_url = (
        ('resnet18-f37072fd.pth', URL, 'f37072fd', 'ResNet18', DOC_URL)
        if architecture == 'resnet18' else
        ('convnext_tiny-983f1562.pth', CONVNEXT_URL, '983f1562', 'ConvNeXt-Tiny', CONVNEXT_DOC_URL)
    )
    target = out / filename
    if not target.exists():
        fd, temp_name = tempfile.mkstemp(prefix=filename + '.', suffix='.download', dir=out)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            urllib.request.urlretrieve(url, temp_path)
            digest = _sha256(temp_path)
            if not digest.startswith(prefix):
                raise ValueError('Downloaded weights fail published filename hash prefix; do not use')
            if digest != IMAGENET_WEIGHT_SHA256[architecture]:
                raise ValueError(f'Downloaded SHA-256 is not allowlisted for {architecture}: {digest}')
            temp_path.replace(target)
        finally:
            temp_path.unlink(missing_ok=True)
    digest = _sha256(target)
    if not digest.startswith(prefix):
        raise ValueError('Weights fail published filename hash prefix; do not use')
    if digest != IMAGENET_WEIGHT_SHA256[architecture]:
        raise ValueError(f'Existing weights SHA-256 is not allowlisted for {architecture}: {digest}')
    with urllib.request.urlopen(LICENSE_URL, timeout=60) as response:
        license_text = response.read().decode('utf-8')
    (out / 'torchvision-LICENSE.txt').write_text(license_text, encoding='utf-8')
    report = dict(model=f'torchvision {model_name} IMAGENET1K_V1', weights_url=url,
                  upstream_training_data='ImageNet-1K', upstream_documentation=doc_url,
                  license_url=LICENSE_URL, sha256=digest, bytes=target.stat().st_size,
                  medical_pretraining=False,
                  review_note='Torchvision software license retained; verify all weight/data license conditions for organizer submission. No claim of organizer approval.')
    provenance_name = 'provenance.json' if architecture == 'resnet18' else f'{architecture}-provenance.json'
    (out / provenance_name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

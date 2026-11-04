"""Download public torchvision ResNet18 ImageNet weights and retain provenance.
Does not accept arbitrary URLs or medical weights. License review remains required.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
LICENSE_URL = 'https://raw.githubusercontent.com/pytorch/vision/v0.25.0/LICENSE'
DOC_URL = 'https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html'


def main():
    out = ROOT / 'weights'
    out.mkdir(exist_ok=True)
    target = out / 'resnet18-f37072fd.pth'
    if not target.exists():
        urllib.request.urlretrieve(URL, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'):
        raise ValueError('Downloaded weights fail published filename hash prefix; do not use')
    with urllib.request.urlopen(LICENSE_URL, timeout=60) as response:
        license_text = response.read().decode('utf-8')
    (out / 'torchvision-LICENSE.txt').write_text(license_text, encoding='utf-8')
    report = dict(model='torchvision ResNet18 IMAGENET1K_V1', weights_url=URL,
                  upstream_training_data='ImageNet-1K', upstream_documentation=DOC_URL,
                  license_url=LICENSE_URL, sha256=digest, bytes=target.stat().st_size,
                  medical_pretraining=False,
                  review_note='Torchvision software license retained; verify all weight/data license conditions for organizer submission. No claim of organizer approval.')
    (out / 'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

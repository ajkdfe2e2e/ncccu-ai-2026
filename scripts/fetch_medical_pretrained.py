"""Fetch the requested CheXpert-only research weights and retain their provenance."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
FILENAME = 'chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
URL = f'https://github.com/mlmed/torchxrayvision/releases/download/v1/{FILENAME}'
ASSET_API = 'https://api.github.com/repos/mlmed/torchxrayvision/releases/assets/18622079'
LICENSE_URL = 'https://raw.githubusercontent.com/mlmed/torchxrayvision/master/LICENSE'
WEIGHTS_NAME = 'densenet121-res224-chex'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def request(url, accept=None):
    headers = {'User-Agent': 'Chest-X-ray-local-research'}
    if accept:
        headers['Accept'] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90)


def main():
    output = ROOT / 'weights' / 'medical'
    output.mkdir(parents=True, exist_ok=True)
    with request(ASSET_API, 'application/vnd.github+json') as response:
        asset = json.load(response)
    if asset['name'] != FILENAME or asset['browser_download_url'] != URL:
        raise ValueError('The official release asset does not match the requested CheXpert-only weights')
    (output / 'upstream-asset.json').write_text(json.dumps(asset, indent=2) + '\n', encoding='utf-8')
    original = output / FILENAME
    provenance_path = output / 'provenance.json'
    if not original.exists():
        fd, temporary = tempfile.mkstemp(prefix=FILENAME + '.', suffix='.download', dir=output)
        try:
            print(f'Downloading official asset: {URL}', flush=True)
            with os.fdopen(fd, 'wb') as destination:
                for attempt_url in (URL, ASSET_API + '?download=1', URL + '?download=1'):
                    destination.seek(0)
                    destination.truncate()
                    with request(attempt_url, 'application/octet-stream') as source:
                        while chunk := source.read(1024 * 1024):
                            destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                    received = destination.tell()
                    print(f'Received {received}/{asset["size"]} bytes', flush=True)
                    if received == asset['size']:
                        break
                else:
                    raise ValueError(f'Official download was incomplete: {received}/{asset["size"]} bytes')
            os.replace(temporary, original)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if original.stat().st_size != asset['size']:
        raise ValueError('Existing weights have the wrong size')
    source_sha256 = digest(original)
    upstream_digest = asset.get('digest')
    if upstream_digest and upstream_digest != f'sha256:{source_sha256}':
        raise ValueError('The downloaded SHA-256 differs from the upstream release digest')
    if provenance_path.exists():
        recorded = json.loads(provenance_path.read_text(encoding='utf-8'))
        if recorded['original']['sha256'] != source_sha256:
            raise ValueError('Existing weights differ from the locally recorded provenance')
    print(f'Official asset downloaded: {original.stat().st_size} bytes; sha256={source_sha256}', flush=True)

    import torch
    import torchxrayvision as xrv
    from torch import nn
    metadata = xrv.models.model_urls[WEIGHTS_NAME]
    if metadata['weights_url'] != URL:
        raise ValueError('Installed TorchXRayVision registry points to different weights')
    # The upstream asset is a legacy pickled Module. This one-time conversion follows
    # TorchXRayVision's own loader and only reads the official asset checked above.
    # All subsequent training and inference load the converted tensor-only checkpoint.
    legacy = torch.load(original, map_location='cpu', weights_only=False)
    if not isinstance(legacy, nn.Module):
        raise ValueError('Unexpected official checkpoint format')
    for module in legacy.modules():
        if not hasattr(module, '_non_persistent_buffers_set'):
            module._non_persistent_buffers_set = set()
    state = {name: tensor.detach().cpu() for name, tensor in legacy.state_dict().items()}
    if state['classifier.weight'].shape != (18, 1024) or state['features.conv0.weight'].shape != (64, 1, 7, 7):
        raise ValueError('Unexpected CheXpert DenseNet classifier or grayscale input shape')
    converted = output / f'{WEIGHTS_NAME}-state.pt'
    payload = {'model': state, 'pathologies': metadata['labels'], 'weights_name': WEIGHTS_NAME,
               'original_sha256': source_sha256, 'medical_pretraining': True}
    fd, temporary = tempfile.mkstemp(prefix=converted.name + '.', suffix='.tmp', dir=output)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        # Ensure future consumers can use the restricted tensor-only loader.
        torch.load(temporary, map_location='cpu', weights_only=True)
        os.replace(temporary, converted)
    finally:
        Path(temporary).unlink(missing_ok=True)
    with request(LICENSE_URL) as response:
        (output / 'torchxrayvision-LICENSE.txt').write_bytes(response.read())
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'weights_name': WEIGHTS_NAME, 'upstream_training_data': 'CheXpert only (upstream model registry)',
        'weights_url': URL, 'asset_api': ASSET_API, 'upstream_digest': upstream_digest,
        'digest_note': 'SHA-256 recorded locally; upstream asset has no published digest.' if not upstream_digest else 'Upstream release digest verified.',
        'torchxrayvision_version': version('torchxrayvision'), 'pathologies': metadata['labels'],
        'original': {'path': str(original), 'bytes': original.stat().st_size, 'sha256': source_sha256},
        'tensor_checkpoint': {'path': str(converted), 'bytes': converted.stat().st_size, 'sha256': digest(converted)},
        'conversion': 'Official legacy Module -> CPU tensor state_dict; no training or weight modification.',
        'license_url': LICENSE_URL, 'medical_pretraining': True,
        'usage': 'Separate user-requested research experiment; excluded from the prior competition-compliant track.',
        'data_overlap_status': 'Unknown: local patient/study/image identifiers are anonymized; source IDs are unavailable.',
        'license_note': 'Software license retained. This does not establish downstream weight/data license permission or competition eligibility.',
    }
    provenance_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

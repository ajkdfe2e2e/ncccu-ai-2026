"""Fail early if pinned training stack or CUDA execution is unavailable."""
import json
import torch
import torchvision


def main():
    if torch.__version__.split('+')[0] != '2.10.0' or torchvision.__version__.split('+')[0] != '0.25.0':
        raise RuntimeError('Expected torch 2.10.0 and torchvision 0.25.0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    x = torch.randn(8, 8, device='cuda', requires_grad=True)
    x.square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError('CUDA backward produced non-finite gradients')
    print(json.dumps(dict(status='CUDA forward/backward PASS', torch=str(torch.__version__),
                          torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
                          device=torch.cuda.get_device_name(0),
                          total_memory=torch.cuda.get_device_properties(0).total_memory)), flush=True)


if __name__ == '__main__':
    main()

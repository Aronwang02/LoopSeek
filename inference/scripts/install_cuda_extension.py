#!/usr/bin/env python3
"""Install a matching prebuilt wheel, or build the pinned CUDA extension.

Torch 2.14 FlashAttention uses mjun0812's pinned v0.10.0 release. A failed
third-party wheel install stops instead of silently starting a long compilation.
JEV_CUDA_FORCE_BUILD=1 still explicitly requests source compilation.

Older flash-attn / causal-conv1d sdists hardcode C++17. PyTorch >= 2.13
requires C++20. Only their build flags are patched; package versions and
kernel sources remain pinned. Downloaded sdists are verified against PyPI.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from urllib.error import URLError
from urllib.parse import quote, unquote
from pathlib import Path


PACKAGES = {'flash-attn': ('flash-attention', 'flash_attn'),
            'causal-conv1d': ('causal-conv1d', 'causal_conv1d')}


def third_party_flash_wheel(version, torch):
    major_minor = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
    if major_minor != '2.14':
        return None
    py = f'cp{sys.version_info.major}{sys.version_info.minor}'
    cuda = (torch.version.cuda or '').replace('.', '')
    # Actual assets from mjun0812/flash-attention-prebuild-wheels v0.10.0.
    hashes = {
        '126': '87e2f7356a1e5dd955a1c7b1280c28643a22bfec3b7128d633e701a029ea30b6',
        '130': '623308fbb31944b6c312b0cc0c500d73d41e5c41e9d8933c27d290e0ca45caac',
        '132': '552a504fa49569e8f21d2bc90e91bdd53419dd4c9b6bd34399fae86f6d0e99dd',
    }
    if (version != '2.8.3' or py != 'cp312' or cuda not in hashes or sys.platform != 'linux'
            or platform.machine() != 'x86_64' or not torch._C._GLIBCXX_USE_CXX11_ABI):
        raise SystemExit('No pinned third-party FlashAttention wheel for this Python/CUDA/platform/ABI; '
                         'use Python 3.12, Linux x86_64, torch 2.14 with CUDA 12.6/13.0/13.2 and CXX11 ABI, '
                         'or set FLASH_ATTN_WHEEL / JEV_CUDA_FORCE_BUILD=1 explicitly')
    filename = f'flash_attn-{version}+cu{cuda}torch{major_minor}-{py}-{py}-linux_x86_64.whl'
    url = 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.0/' + quote(filename)
    return url, hashes[cuda]


def official_wheel_url(package, version, torch):
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or not torch.version.cuda:
        return None
    repo, module = PACKAGES[package]
    major, minor = torch.__version__.split('+')[0].split('.')[:2]
    cuda = torch.version.cuda.split('.')[0]
    abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    py = f'cp{sys.version_info.major}{sys.version_info.minor}'
    filename = f'{module}-{version}+cu{cuda}torch{major}.{minor}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl'
    return f'https://github.com/Dao-AILab/{repo}/releases/download/v{version}/{quote(filename)}'


def install_prebuilt(package, version, torch, run):
    third_party = third_party_flash_wheel(version, torch) if package == 'flash-attn' else None
    url = third_party[0] if third_party else official_wheel_url(package, version, torch)
    method = 'third_party_wheel' if third_party else 'official_wheel'
    if url is None:
        return False
    wheel_dir = run / 'prebuilt'
    wheel_dir.mkdir(exist_ok=True)
    wheel = wheel_dir / unquote(url.rsplit('/', 1)[1])
    partial = wheel.with_suffix('.part')
    try:
        if not wheel.exists():
            print(f'downloading {method}: {url}', flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output)
            partial.replace(wheel)
        if third_party and hashlib.sha256(wheel.read_bytes()).hexdigest() != third_party[1]:
            wheel.unlink(missing_ok=True)
            raise OSError('pinned FlashAttention wheel SHA256 mismatch')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)],
                       check=True)
        module = PACKAGES[package][1]
        subprocess.run([sys.executable, '-c',
                        f'import {module}; assert {module}.__version__.split("+")[0] == {version!r}'], check=True)
    except (URLError, OSError, subprocess.CalledProcessError) as exc:
        partial.unlink(missing_ok=True)
        if third_party:
            raise SystemExit(f'Pinned third-party FlashAttention wheel installation failed ({exc}); '
                             'retry the download or set FLASH_ATTN_WHEEL. No automatic source compilation.') from exc
        print(f'official wheel unavailable or unusable ({exc}); compiling {package} {version}', flush=True)
        return False
    info = {'method': method, 'package': package, 'version': version,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'wheel_url': url}
    if third_party:
        info['wheel_sha256'] = third_party[1]
    (run / 'install_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)
    return True


def build_source(package, version, torch, run):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=30) as r:
        metadata = json.load(r)
    source = next(x for x in metadata['urls'] if x['packagetype'] == 'sdist')
    archive = run / source['filename']
    if not archive.exists():
        with urllib.request.urlopen(source['url'], timeout=60) as r:
            archive.write_bytes(r.read())
    actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_hash != source['digests']['sha256']:
        raise SystemExit('PyPI source checksum mismatch')
    source_dir = run / 'source'
    if not source_dir.exists():
        source_dir.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(source_dir, filter='data')
    setup = next(source_dir.glob('*/setup.py'))
    major, minor = map(int, torch.__version__.split('+')[0].split('.')[:2])
    text = setup.read_text()
    count = text.count('-std=c++17') if (major, minor) >= (2, 13) else 0
    if count:
        setup.write_text(text.replace('-std=c++17', '-std=c++20'))
    info = {'method': 'source', 'package': package, 'version': version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'sdist_sha256': actual_hash,
            'cxx_standard': 'c++20' if (major, minor) >= (2, 13) else 'c++17',
            'build_flag_replacements': count}
    (run / 'build_info.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)
    env = dict(os.environ)
    env['FLASH_ATTENTION_FORCE_BUILD'] = 'TRUE'
    env['CAUSAL_CONV1D_FORCE_BUILD'] = 'TRUE'
    wheels = run / 'wheels'
    wheels.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(setup.parent), '--no-build-isolation',
                    '--no-deps', '--wheel-dir', str(wheels)], env=env, check=True)
    wheel = next(wheels.glob('*.whl'))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)], check=True)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', choices=tuple(PACKAGES))
    parser.add_argument('version')
    args = parser.parse_args()
    root = Path(os.environ.get('JEV_CUDA_BUILD_ROOT', str(Path(tempfile.gettempdir()) / 'jev-cuda-builds')))
    run = root / f'{args.package}-{args.version}-torch{torch.__version__}'
    run.mkdir(parents=True, exist_ok=True)
    if os.environ.get('JEV_CUDA_FORCE_BUILD', '0') != '1':
        if install_prebuilt(args.package, args.version, torch, run):
            return
    build_source(args.package, args.version, torch, run)


if __name__ == '__main__':
    main()

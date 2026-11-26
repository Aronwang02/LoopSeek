"""CUDA wheel selection and source fallback, without network or GPU access."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


spec = importlib.util.spec_from_file_location(
    'cuda_installer', Path(__file__).resolve().parents[1] / 'scripts/install_cuda_extension.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
TORCH = SimpleNamespace(__version__='2.9.1+cu126', version=SimpleNamespace(cuda='12.6'),
                        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True))


class CudaExtensionInstall(unittest.TestCase):
    def test_official_wheel_matches_cuda_torch_python_and_abi(self):
        with patch.object(installer.sys, 'platform', 'linux'), \
                patch.object(installer.platform, 'machine', return_value='x86_64'), \
                patch.object(installer.sys, 'version_info', SimpleNamespace(major=3, minor=12)):
            for package, module in [('flash-attn', 'flash_attn'), ('causal-conv1d', 'causal_conv1d')]:
                url = installer.official_wheel_url(package, '1.0', TORCH)
                self.assertIn(f'{module}-1.0%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl', url)
            self.assertIsNone(installer.official_wheel_url(
                'flash-attn', '1.0', SimpleNamespace(version=SimpleNamespace(cuda=None))))

    def test_missing_wheel_requests_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(installer.urllib.request, 'urlopen',
                             side_effect=HTTPError('wheel', 404, 'missing', None, None)), \
                patch.object(installer.subprocess, 'run') as install:
            self.assertFalse(installer.install_prebuilt('flash-attn', '2.8.3', TORCH, Path(tmp)))
            install.assert_not_called()

    def test_binary_import_failure_requests_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(installer.urllib.request, 'urlopen', return_value=io.BytesIO(b'wheel')), \
                patch.object(installer.subprocess, 'run',
                             side_effect=[None, subprocess.CalledProcessError(1, ['import'])]) as install:
            self.assertFalse(installer.install_prebuilt('flash-attn', '2.8.3', TORCH, Path(tmp)))
            self.assertIn('--no-deps', install.call_args_list[0].args[0])
            self.assertIn('--force-reinstall', install.call_args_list[0].args[0])
            self.assertFalse((Path(tmp) / 'install_info.json').exists())

    def test_success_records_prebuilt_install(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(installer.urllib.request, 'urlopen', return_value=io.BytesIO(b'wheel')), \
                patch.object(installer.subprocess, 'run'):
            self.assertTrue(installer.install_prebuilt('causal-conv1d', '1.7.0', TORCH, Path(tmp)))
            self.assertEqual(json.loads((Path(tmp) / 'install_info.json').read_text())['method'], 'official_wheel')


if __name__ == '__main__':
    unittest.main()

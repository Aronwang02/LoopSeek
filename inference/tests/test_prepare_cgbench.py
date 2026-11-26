"""CG-Bench download cache (scripts/prepare_cgbench.py) against a fake Hub: no network, token or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location(
    'prepare_cgbench', Path(__file__).resolve().parents[1] / 'scripts/prepare_cgbench.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class FakeHub:
    """Serves annotation files and video_chunk_NN.zip volumes from a local directory, counting downloads."""

    def __init__(self, src):
        self.src, self.downloads = src, []
        self.module = types.SimpleNamespace(hf_hub_download=self.download, HfApi=lambda: self)

    def download(self, repo, filename, repo_type, revision, local_dir):
        assert (repo, repo_type, revision) == (prep.REPO, 'dataset', prep.REVISION)
        self.downloads.append(filename)
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((self.src / filename).read_bytes())
        return str(out)

    def list_repo_files(self, repo, repo_type, revision):
        return sorted(p.name for p in self.src.iterdir()) + ['clue_video_chunk_01.zip', 'subtitles.zip']


class PrepareCgbench(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cgbench_prep_')
        base = Path(self.tmp.name)
        self.src, self.root, self.scratch = base / 'hub', base / 'blob/cgbench', base / 'ssd'
        self.src.mkdir()
        full = [{'qid': q, 'video_uid': f'v{q}'} for q in range(4)]
        (self.src / 'cgbench.json').write_text(json.dumps(full))
        (self.src / 'cgbench_mini.json').write_text(json.dumps(full[:1]))
        for n, uids in ((1, ['v0', 'v1']), (2, ['v2', 'v3'])):
            with zipfile.ZipFile(self.src / f'video_chunk_{n:02d}.zip', 'w') as zf:
                for u in uids:
                    zf.writestr(f'cg_videos_720p/sub/{u}.mp4', f'video {u}'.encode())
                zf.writestr('cg_videos_720p/readme.txt', b'not a video')
        self.hub = FakeHub(self.src)
        self.env = patch.dict(os.environ, {'HF_TOKEN': 'test-token', 'DATA_ROOT': str(self.root)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_main(self, *extra):
        argv = ['prepare_cgbench.py', '--root', str(self.root), '--scratch', str(self.scratch), '--workers', '2', *extra]
        with patch.dict(sys.modules, {'huggingface_hub': self.hub.module}), patch.object(sys, 'argv', argv), \
                patch.object(prep.time, 'sleep'):
            prep.main()

    def test_first_run_downloads_then_later_runs_hit_the_cache(self):
        self.run_main()
        videos = sorted(p.name for p in (self.root / 'videos').iterdir())
        self.assertEqual(videos, ['v0.mp4', 'v1.mp4', 'v2.mp4', 'v3.mp4'])           # nested paths flattened
        self.assertEqual((self.root / 'videos/v2.mp4').read_bytes(), b'video v2')
        record = json.loads((self.root / 'PREPARED.json').read_text())
        self.assertEqual((record['revision'], record['videos'], record['chunks']), (prep.REVISION, 4, 2))
        self.assertNotIn('clue_video_chunk_01.zip', self.hub.downloads)
        self.assertFalse(list((self.scratch / 'cgbench-download/chunks').glob('*.zip')))  # volumes deleted
        self.assertFalse((self.root / '.prepare/lock.json').exists())
        n = len(self.hub.downloads)
        self.run_main()
        self.assertEqual(len(self.hub.downloads), n)                                 # cache hit: no Hub access
        self.run_main('--check-only')

    def test_interrupted_run_resumes_after_the_finished_volume(self):
        real = prep._extract_chunk

        def fail_second(name, *a):
            if name == 'video_chunk_02.zip':
                raise OSError('connection reset')
            return real(name, *a)
        with patch.object(prep, '_extract_chunk', side_effect=fail_second), self.assertRaises(OSError):
            self.run_main()
        self.assertTrue((self.root / '.prepare/chunks/video_chunk_01.zip.json').exists())
        self.assertFalse((self.root / 'PREPARED.json').exists())
        self.assertFalse((self.root / '.prepare/lock.json').exists())                  # released for the next job
        self.hub.downloads.clear()
        self.run_main()
        self.assertNotIn('video_chunk_01.zip', self.hub.downloads)
        self.assertIn('video_chunk_02.zip', self.hub.downloads)
        self.assertTrue((self.root / 'PREPARED.json').exists())

    def test_missing_annotated_video_is_not_marked_prepared(self):
        full = json.loads((self.src / 'cgbench.json').read_text()) + [{'qid': 9, 'video_uid': 'v_absent'}]
        (self.src / 'cgbench.json').write_text(json.dumps(full))
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertIn('v_absent', str(cm.exception))
        self.assertFalse((self.root / 'PREPARED.json').exists())
        with self.assertRaises(SystemExit):
            self.run_main('--check-only')

    def test_without_token_fails_with_instructions(self):
        os.environ.pop('HF_TOKEN')
        with self.assertRaises(SystemExit) as cm:
            self.run_main()
        self.assertIn('HF_TOKEN', str(cm.exception))
        self.assertIn('huggingface.co/datasets/CG-Bench/CG-Bench', str(cm.exception))

    def test_waits_for_another_job_and_then_uses_its_result(self):
        (self.root / '.prepare').mkdir(parents=True)
        (self.root / '.prepare/lock.json').write_text(json.dumps({'owner': {'host': 'other'}, 'heartbeat': time.time()}))
        checks = iter([False, False, True])                     # the other job finishes while we wait
        with patch.object(prep, 'cache_hit', side_effect=lambda root: next(checks)):
            self.run_main()
        self.assertEqual(self.hub.downloads, [])

    def test_stale_lock_is_taken_over(self):
        (self.root / '.prepare').mkdir(parents=True)
        (self.root / '.prepare/lock.json').write_text(json.dumps({'owner': {'host': 'dead'}, 'heartbeat': 0}))
        self.run_main()
        self.assertTrue((self.root / 'PREPARED.json').exists())


if __name__ == '__main__':
    unittest.main()

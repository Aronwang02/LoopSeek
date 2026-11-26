"""Parallel Video-MME chunk download (scripts/prepare_videomme.download) against a fake Hub: no network or GPU."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_cgbench
from scripts import prepare_videomme as prep


class FakeHub:
    """Serves videos_chunked_NN.zip from a local directory; records which were fetched and peak concurrency."""

    def __init__(self, src):
        self.src, self.fetched, self.active, self.peak = src, [], 0, 0
        self.guard = threading.Lock()
        self.module = types.SimpleNamespace(hf_hub_download=self.download)

    def download(self, repo, filename, repo_type, revision, local_dir):
        assert (repo, repo_type, revision) == (prep.REPO, 'dataset', prep.REVISION)
        with self.guard:
            self.fetched.append(filename)
            self.active += 1
            self.peak = max(self.peak, self.active)
        threading.Event().wait(0.05)                 # overlap with the other workers
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((self.src / filename).read_bytes())
        with self.guard:
            self.active -= 1
        return str(out)


class ParallelDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='videomme_dl_')
        base = Path(self.tmp.name)
        self.src, self.root, self.scratch = base / 'hub', base / 'shared/videomme', base / 'ssd'
        self.src.mkdir()
        self.root.mkdir(parents=True)
        (self.root / 'test.parquet').write_bytes(b'annotations present')   # skips the annotation download
        self.rows = [{'videoID': f'v{i:02d}'} for i in range(1, 21)]
        for i in range(1, 21):
            with zipfile.ZipFile(self.src / f'videos_chunked_{i:02d}.zip', 'w') as zf:
                zf.writestr(f'data/v{i:02d}.mp4', f'video {i}'.encode())
                zf.writestr(f'data/other{i:02d}.mp4', b'a video no question needs')
        # chunks 1 and 2 were finished by an earlier (serial) run
        videos, done = self.root / 'videos', self.root / '.prepare/chunks'
        videos.mkdir()
        done.mkdir(parents=True)
        for i in (1, 2):
            (videos / f'v{i:02d}.mp4').write_bytes(f'video {i}'.encode())
            prep.write_json(done / f'videos_chunked_{i:02d}.zip.json',
                            {'revision': prep.REVISION, 'subsets': list(prep.SUBSETS),
                             'videos': {f'v{i:02d}.mp4': len(f'video {i}')}})
        self.hub = FakeHub(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_remaining_chunks_download_in_parallel_and_complete_the_cache(self):
        with patch.dict(sys.modules, {'huggingface_hub': self.hub.module}), \
                patch.object(prep, 'source_rows', return_value=self.rows), \
                patch.object(prepare_cgbench.time, 'sleep'):
            prep.download(self.root, self.scratch, workers=4)
        self.assertEqual(sorted(self.hub.fetched), [f'videos_chunked_{i:02d}.zip' for i in range(3, 21)])
        self.assertGreater(self.hub.peak, 1)
        self.assertLessEqual(self.hub.peak, 4)
        names = sorted(p.name for p in (self.root / 'videos').iterdir())
        self.assertEqual(names, [f'v{i:02d}.mp4' for i in range(1, 21)])   # no unneeded or temp files
        self.assertEqual((self.root / 'videos/v17.mp4').read_bytes(), b'video 17')
        self.assertEqual(json.loads((self.root / 'PREPARED.json').read_text())['revision'], prep.REVISION)
        self.assertFalse(list((self.scratch / 'videomme-download/chunks').glob('*.zip')))
        self.assertFalse((self.root / '.prepare/lock.json').exists())


    def test_a_chunk_extracted_for_other_subsets_is_extracted_again(self):
        """A marker of an earlier run that needed other subsets (no "subsets" or other ones) does not skip its chunk."""
        prep.write_json(self.root / '.prepare/chunks/videos_chunked_01.zip.json',
                        {'revision': prep.REVISION, 'videos': {'v01.mp4': len('video 1')}})
        with patch.dict(sys.modules, {'huggingface_hub': self.hub.module}), \
                patch.object(prep, 'source_rows', return_value=self.rows), \
                patch.object(prepare_cgbench.time, 'sleep'):
            prep.download(self.root, self.scratch, workers=4)
        self.assertEqual(sorted(self.hub.fetched), [f'videos_chunked_{i:02d}.zip' for i in [1] + list(range(3, 21))])


if __name__ == '__main__':
    unittest.main()

"""CPU orchestration checks of run.sh and scripts/train_pipeline.sh: the expensive Python entry points are replaced by a
stub that records its arguments; the shell stages, done markers, resume and merges run as they are.

python -m pytest tests/test_experiment_scripts.py -q
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]

FAKE_PY = r'''
import json, os, pathlib, sys, types
a = sys.argv[1:]
launcher = None
if a[:2] == ['-m', 'torch.distributed.run']:          # SFT_GPUS: record the torchrun arguments, then the script
    k = next(i for i, x in enumerate(a) if x.endswith('.py'))
    launcher, a = a[2:k], a[k:]
if a[0] == '-c':
    sys.argv = ['-c'] + a[2:]
    exec(a[1])
    raise SystemExit
if a[0] == '-':
    sys.argv = ['-'] + a[1:]
    sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: len(os.environ['GPUS'].split())))
    exec(sys.stdin.read())
    raise SystemExit
script = pathlib.Path(a[0]).name
with open('calls.jsonl', 'a') as f:
    f.write(json.dumps({'script': script, 'args': a[1:], 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'launcher': launcher, 'local_features': os.environ.get('JEV_LOCAL_FEATURES')}) + '\n')
if os.environ.get('FAKE_FAIL_SCRIPT') == script:
    raise SystemExit(7)
if script in ('mix_long_manifests.py', 'merge_shards.py'):
    os.execv(sys.executable, [sys.executable] + a)
def arg(k, default=None):
    return a[a.index(k) + 1] if k in a else default
def write(path, value):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value)
name = arg('--name')
d = pathlib.Path('runs') / name if name else None
if script == 'build_cgbench.py':
    rows = [{'sample_id': f'{s}{i}', 'split': s, 'key_spans': [[1, 2]]}
            for s, n in (('train', 16), ('dev', 8), ('test', 8)) for i in range(n)]
    write(arg('--out'), ''.join(json.dumps(r) + '\n' for r in rows))
elif script == 'run_agent_v3.py':
    shard = int(arg('--shard', '0/1').split('/')[0])
    r = {'sample_id': f'q{shard}', 'group_key': 'v', 'correct': True, 'parsed': 'A'}
    write(d / 'results.jsonl', json.dumps(r) + '\n')
    write(d / 'trajectories.jsonl', json.dumps({'sample_id': r['sample_id'], 'status': 'ok'}) + '\n')
    write(d / 'config.yaml', pathlib.Path(arg('--config')).read_text())
elif script == 'gen_role_sft_from_trajectories.py':
    write(arg('--out'), json.dumps({'split': arg('--split')}) + '\n')
elif script == 'train_sft.py':
    write(d / 'best_snapshot.txt', str(d / 'snapshots/step_0200/latest') + '\n')
elif script == 'make_policy_set.py':
    ps = {k: arg('--' + k) for k in ('controller', 'localize', 'frame', 'verifier') if '--' + k in a}
    write(arg('--out'), json.dumps(ps))
elif script == 'repair_features.py':
    pass  # its frame references are exercised in test_repair_features.py
elif script == 'filter_cgbench_vts.py':
    if '--cache-restore' in a:                       # VTS_CACHE hit for the splits in FAKE_VTS_HITS, exit 3 = miss
        if arg('--split') not in os.environ.get('FAKE_VTS_HITS', '').split(','):
            raise SystemExit(3)
    elif ('lvhaystack_all' in arg('--manifest') or 'videoitg_all' in arg('--manifest')) and '--shard' not in a:
        rows = [json.loads(l) for l in pathlib.Path(arg('--manifest')).read_text().splitlines()]
        write(arg('--out'), ''.join(json.dumps(r) + '\n' for r in rows if r['split'] == arg('--split')))
elif script == 'build_lvhaystack.py':
    rows = [{'sample_id': f'lvh:{s}:{i}', 'split': s, 'key_spans': [[10, 12], [30, 32]], 'source': 'lvhaystack'}
            for s, n in (('train', 16), ('dev', 8)) for i in range(n)]
    write(pathlib.Path(arg('--out-dir')) / 'lvhaystack_all.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
elif script == 'build_videoitg.py':
    rows = [{'sample_id': f'videoitg:{s}{i}', 'split': s, 'key_spans': [[10, 11]], 'source': 'videoitg'}
            for s, n in (('train', 16), ('dev', 8)) for i in range(n)]
    write(pathlib.Path(arg('--out-dir')) / 'videoitg_all.jsonl', ''.join(json.dumps(r) + '\n' for r in rows))
elif script == 'sample_videoitg.py':
    k = a.index('--manifest') + 1
    mans = []
    while k < len(a) and not a[k].startswith('--'):
        mans.append(a[k]); k += 1
    rows = [json.loads(l) for m in mans for l in pathlib.Path(m).read_text().splitlines()]
    if '--rest-of' in a:                             # the train rows not in the given selection
        taken = {json.loads(l)['sample_id'] for l in pathlib.Path(arg('--rest-of')).read_text().splitlines()}
        rows = [r for r in rows if r['split'] == 'train' and r['sample_id'] not in taken]
    elif '--bins' in a and arg('--train') == '0':    # a dev draw only
        rows = [r for r in rows if r['split'] == 'dev'][:int(arg('--dev'))]
    write(arg('--out'), ''.join(json.dumps(r) + '\n' for r in rows))
else:
    raise SystemExit(f'unmocked command: {a}')
'''


class TrainingScripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='jev_train_test_')
        self.root = Path(self.tmp.name)
        (self.root / 'scripts').mkdir()
        for name in ('train_pipeline.sh', 'mix_long_manifests.py', 'merge_shards.py', 'dagger_rounds.py'):
            shutil.copy(ROOT / 'scripts' / name, self.root / 'scripts' / name)
        shutil.copy(ROOT / 'run.sh', self.root / 'run.sh')
        shutil.copytree(ROOT / 'configs', self.root / 'configs')
        md = self.root / 'data/manifests/longvid'
        md.mkdir(parents=True)
        self.manifest = [{'sample_id': f'q{i}', 'group_key': 'v', 'key_spans': [[1, 2]]} for i in range(2)]
        for split in ('train', 'dev', 'test'):
            (md / f'cgbench_{split}_vts.jsonl').write_text(''.join(json.dumps(dict(r, split=split, source='cgbench')) + '\n'
                                                               for r in self.manifest))
        fake = self.root / 'python_stub'
        fake.write_text(f'#!{sys.executable}\n' + textwrap.dedent(FAKE_PY))
        fake.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'JEV_', 'SFT_', 'WANDB_', 'VTS_'))}
        self.env.update(PY=str(fake), PYTHONPATH=str(ROOT), RUN_ENV_LOADED='1', TAG='t', GPUS='2,3', PREPARED_DATA='1',
                        DRY='0', DAGGER_ROUNDS='1', WANDB_MODE='disabled', RUNSH_LOGGED='1', LVH_TRAIN='0',
                        COLLECT_QUEUE='0')

    def tearDown(self):
        self.tmp.cleanup()

    def run_sh(self, cmd='sft', **env):
        return subprocess.run(['bash', 'run.sh', cmd], cwd=self.root, env={**self.env, **env}, text=True,
                              capture_output=True, timeout=600)

    def calls(self):
        f = self.root / 'calls.jsonl'
        return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []

    def names(self, script):
        return [c['args'][c['args'].index('--name') + 1] for c in self.calls() if c['script'] == script and '--name' in c['args']]

    def test_sft_runs_every_round_in_order_and_resumes(self):
        p = self.run_sh()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        collect = [n for n in self.names('run_agent_v3.py') if not n.endswith(('_p1',))]
        self.assertEqual(collect, ['t_s1A_train_p0', 't_s1B_train_p0', 't_s1A_dev_p0', 't_s1B_dev_p0', 't_s3_train_p0',
                                   't_s3_dev_p0', 't_s5_train_p0', 't_s5_dev_p0', 't_dagger1_train_p0', 't_dagger1_dev_p0'])
        self.assertEqual(self.names('train_sft.py'), ['t_grounding1', 't_verifier1', 't_grounding2', 't_verifier2',
                                                      't_controller0', 't_controller1'])
        sets = [c['args'][c['args'].index('--out') + 1] for c in self.calls() if c['script'] == 'make_policy_set.py']
        self.assertEqual(sets, ['runs/t_ps1.json', 'runs/t_ps2.json', 'runs/t_ps3_r0.json', 'runs/t_ps3_r1.json'])
        done = sorted(x.name for x in (self.root / 'runs/.done').iterdir())
        self.assertEqual(done, sorted(f't.{s}' for s in ('s1_collect', 's1_data', 's1_train', 's2_train', 's3_collect',
                                                          's4_train', 's5_collect', 's5_train', 's5_dagger1')))
        n = len(self.calls())
        p = self.run_sh()                                      # a second call finds every step done
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(len(self.calls()), n)

    def test_collection_runs_one_shard_per_gpu_and_merges(self):
        p = self.run_sh()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        s1 = [c for c in self.calls() if c['script'] == 'run_agent_v3.py' and 't_s1A_train' in ' '.join(c['args'])]
        self.assertEqual(sorted((c['gpus'], c['args'][c['args'].index('--shard') + 1]) for c in s1), [('2', '0/2'), ('3', '1/2')])
        self.assertTrue((self.root / 'runs/t_s1A_train/results.jsonl').exists())
        rows = [json.loads(l) for l in (self.root / 'runs/t_s1A_train/results.jsonl').read_text().splitlines()]
        self.assertEqual(sorted(r['sample_id'] for r in rows), ['q0', 'q1'])

    def test_collection_records_verifier_inputs_and_oracle_only_in_round_one(self):
        p = self.run_sh()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        by = {}
        for c in self.calls():
            if c['script'] == 'run_agent_v3.py':
                by[c['args'][c['args'].index('--name') + 1].rsplit('_p', 1)[0]] = ' '.join(c['args'])
        self.assertIn('budget.verifier_frame_oracle_pad=3.0', by['t_s1A_train'])
        self.assertIn('budget.record_verifier_inputs=true', by['t_s3_train'])
        self.assertNotIn('oracle', by['t_s3_train'])
        self.assertIn('--label-controller --dagger-beta 0.3', by['t_dagger1_train'])
        self.assertIn('--checkpoint runs/t_ps3_r0.json', by['t_dagger1_train'])

    def test_sft_gpus_run_every_sft_on_all_gpus_with_torchrun_and_fit_checks_first(self):
        p = self.run_sh(SFT_GPUS='all', SFT_FIT_N='64', SFT_FIT_STEPS='100')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        sft = [c for c in self.calls() if c['script'] == 'train_sft.py']
        self.assertTrue(sft and all(c['launcher'] and '--nproc_per_node=2' in c['launcher'] for c in sft))
        self.assertEqual(self.names('train_sft.py')[:2], ['t_grounding1_fit64', 't_grounding1'])
        fits = [n for n in self.names('train_sft.py') if '_fit' in n]
        self.assertEqual(fits, ['t_grounding1_fit64', 't_verifier1_fit64', 't_controller0_fit64'])

    def test_short_video_profile_collects_and_trains_on_both_profiles(self):
        sm = self.root / 'data/manifests/videoitg'
        sm.mkdir(parents=True)
        for split in ('train', 'dev'):
            (sm / f'videoitg_{split}_vts.jsonl').write_text(json.dumps({'sample_id': f'v{split}', 'split': split,
                                                                        'key_spans': [[1, 2]]}) + '\n')
        p = self.run_sh(SHORT_VIDEO='1', SHORT_SOURCE='videoitg', SHORT_MAN=str(sm))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        names = set(n.rsplit('_p', 1)[0] for n in self.names('run_agent_v3.py'))
        self.assertTrue({'t_s1A_train', 't_sv_s1A_train', 't_s5_dev', 't_sv_s5_dev', 't_sv_dagger1_train'} <= names, names)
        sv = [c for c in self.calls() if c['script'] == 'run_agent_v3.py' and 't_sv_' in ' '.join(c['args'])]
        self.assertTrue(all('configs/short.yaml' in c['args'] for c in sv))

    def test_data_step_filters_train_tags_dev_and_test_and_builds_videoitg(self):
        env = {**self.env, 'SHORT_VIDEO': '1', 'SHORT_SOURCE': 'videoitg', 'VIDEOITG_JSON': str(self.root / 'itg.json'),
               'VIDEOITG_VIDEOS': str(self.root), 'CGB_JSON': str(self.root), 'CGB_MINI': str(self.root),
               'CGB_VIDEOS': str(self.root), 'GPUS': '2 3', 'VIDEOITG_MULTI_SEGMENT': '1'}
        p = subprocess.run(['bash', 'scripts/train_pipeline.sh', 'data'], cwd=self.root, env=env, text=True,
                           capture_output=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        filt = [c for c in self.calls() if c['script'] == 'filter_cgbench_vts.py']
        splits = [(Path(c['args'][c['args'].index('--manifest') + 1]).name, c['args'][c['args'].index('--split') + 1],
                   '--tag-only' in c['args']) for c in filt if '--shard' not in c['args']]
        self.assertEqual(splits, [('cgbench_all.jsonl', 'test', True), ('cgbench_all.jsonl', 'train', False),
                                  ('cgbench_all.jsonl', 'dev', True), ('videoitg_all.jsonl', 'train', False),
                                  ('videoitg_all.jsonl', 'dev', True)])
        shards = [c for c in filt if '--shard' in c['args']]
        self.assertTrue(shards and {c['gpus'] for c in shards} == {'2', '3'})
        vitg = [c for c in filt if 'videoitg_all' in ' '.join(c['args'])]
        self.assertTrue(all('--multi-segment' in c['args'] for c in vitg))
        # the paper's VideoITG pool: a binned dev draw, every train question, then their union is what the filter reads
        samples = [c['args'] for c in self.calls() if c['script'] == 'sample_videoitg.py']
        self.assertEqual(len(samples), 3)
        self.assertIn('--bins', samples[0])
        self.assertEqual(samples[0][samples[0].index('--train') + 1], '0')
        self.assertIn('--rest-of', samples[1])
        self.assertIn('--all', samples[2])
        pool = [json.loads(l) for l in (self.root / 'data/manifests/videoitg/videoitg_all.jsonl').read_text().splitlines()]
        self.assertEqual(sorted(r['split'] for r in pool).count('train'), 16)   # all train rows of the build

    def test_videoitg_sample_pool_draws_before_the_filter(self):
        env = {**self.env, 'SHORT_VIDEO': '1', 'SHORT_SOURCE': 'videoitg', 'VIDEOITG_JSON': str(self.root / 'itg.json'),
               'VIDEOITG_VIDEOS': str(self.root), 'CGB_JSON': str(self.root), 'CGB_MINI': str(self.root),
               'CGB_VIDEOS': str(self.root), 'GPUS': '2 3', 'VIDEOITG_POOL': 'sample', 'VIDEOITG_TRAIN_LIMIT': '5'}
        p = subprocess.run(['bash', 'scripts/train_pipeline.sh', 'data'], cwd=self.root, env=env, text=True,
                           capture_output=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        samples = [c['args'] for c in self.calls() if c['script'] == 'sample_videoitg.py']
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][samples[0].index('--train') + 1], '5')
        self.assertNotIn('--bins', samples[0])

    def test_failed_sft_stops_the_pipeline_and_is_not_marked_done(self):
        p = self.run_sh(FAKE_FAIL_SCRIPT='train_sft.py')
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse((self.root / 'runs/.done/t.s1_train').exists())
        self.assertTrue((self.root / 'runs/.done/t.s1_data').exists())
        self.assertNotIn('t_s3_train_p0', self.names('run_agent_v3.py'))

    def test_dry_run_prints_the_plan_only(self):
        p = self.run_sh(DRY='1')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('PLAN t.s1_collect: P s1_collect', p.stdout)
        self.assertIn('PLAN t.s5_dagger1: P s5_dagger 1', p.stdout)
        self.assertEqual(self.calls(), [])


if __name__ == '__main__':
    unittest.main()

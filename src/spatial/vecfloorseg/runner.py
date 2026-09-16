"""Subprocess orchestration for the VecFloorSeg adapter.

Runs in the **cadruler** env and never imports torch: it shells out to the
``vecfloorseg`` conda interpreter for both the input bridge and the model, which
is what keeps the two environments isolated.

Flow
----
1. ``geometry.build_spec`` (caller) -> JSON spec
2. ``vecfloorseg/python build_dataset.py --spec …`` -> pkl + txt + PNG + triangles.npz
3. replicate the sample into the other two splits (the loader always builds
   train/val/test, and ``my_set_dataset_info`` even calls ``dataset[0]``)
4. stage the checkpoint at ``<out_dir>/CUBI/0/ckpt/best1.ckpt``
5. ``vecfloorseg/python graphgym/main.py --cfg … --eval …``
6. return the run dir so the caller can read ``val_result.pkl``

⚠️ ``checkpoint.py::load_ckpt`` returns 0 **silently** when the file is missing,
so this module verifies the checkpoint itself instead of trusting the run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
BRIDGE = os.path.join(HERE, 'build_dataset.py')
DEFAULT_VF_ROOT = os.path.join(REPO_ROOT, 'third_party', 'VecFloorSeg')
SPLITS = ('train', 'val', 'test')


class VecFloorSegError(RuntimeError):
    """Raised when the bridge or the model run fails."""


@dataclass
class RunResult:
    ok: bool
    run_dir: str = ''
    dataset_dir: str = ''
    work_dir: str = ''
    triangles_npz: str = ''
    sample_id: str = ''
    n_regions: int = 0
    transform: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def tail(self, n: int = 12) -> str:
        return '\n'.join(self.log[-n:])


def _run(cmd, cwd, timeout, log):
    log.append('$ ' + ' '.join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        raise VecFloorSegError('timeout after %ss: %s' % (timeout, cmd[1:3]))
    dt = time.time() - t0
    if proc.stdout.strip():
        log.extend(proc.stdout.strip().splitlines()[-40:])
    if proc.returncode != 0:
        if proc.stderr.strip():
            log.extend(proc.stderr.strip().splitlines()[-40:])
        raise VecFloorSegError('command failed (rc=%d, %.1fs): %s'
                               % (proc.returncode, dt, ' '.join(cmd[1:4])))
    log.append('  rc=0 (%.1fs)' % dt)
    return proc.stdout


def _stage_checkpoint(checkpoint: str, out_dir: str, log: list[str]) -> str:
    if not checkpoint or not os.path.isfile(checkpoint):
        raise VecFloorSegError(
            'checkpoint not found: %r  (load_ckpt() would silently use RANDOM '
            'weights, so this is fatal)' % checkpoint)
    dst_dir = os.path.join(out_dir, 'CUBI', '0', 'ckpt')
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, 'best1.ckpt')
    if (not os.path.exists(dst)
            or os.path.getsize(dst) != os.path.getsize(checkpoint)):
        shutil.copy2(checkpoint, dst)
    log.append('staged ckpt -> %s (%.1f MB)'
               % (dst, os.path.getsize(dst) / 1048576.0))
    return dst


def run_adapter(spec: dict[str, Any], *, python_exe: str, checkpoint: str,
                out_dir: str, vf_root: str | None = None,
                timeout_s: int = 1800, device: str = 'cuda:0') -> RunResult:
    """Build the sample, run ``--eval`` on it and return the run directory."""
    vf_root = vf_root or os.environ.get('VECFLOORSEG_ROOT') or DEFAULT_VF_ROOT
    works = os.path.dirname(spec['work_dir'].rstrip('\\/')) or spec['work_dir']
    os.makedirs(works, exist_ok=True)
    os.makedirs(spec['dataset_dir'], exist_ok=True)
    os.makedirs(spec['work_dir'], exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log: list[str] = []
    res = RunResult(ok=False, work_dir=spec['work_dir'],
                    dataset_dir=spec['dataset_dir'],
                    sample_id=str(spec['sample_id']),
                    transform=spec.get('transform', {}), log=log)

    env = dict(os.environ)
    env.setdefault('VECFLOORSEG_ROOT', vf_root)
    env.setdefault('PYTHONPATH', vf_root)
    env['PYTHONIOENCODING'] = 'utf-8'

    # ---- 1) bridge (val split only; the others are copies) -----------------
    spec_val = dict(spec, split='val')
    spec_path = os.path.join(spec['work_dir'], 'spec_val.json')
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump(spec_val, f)

    old_env, os.environ = os.environ, env
    try:
        out = _run([python_exe, '-u', BRIDGE, '--spec', spec_path],
                   cwd=works, timeout=timeout_s, log=log)
    finally:
        os.environ = old_env
    try:
        info = json.loads(out.strip().splitlines()[-1])
        res.n_regions = int(info.get('n_regions', 0))
    except Exception:
        pass

    # ---- 2) replicate into train/test -------------------------------------
    for extra in ('train', 'test'):
        shutil.copy2(os.path.join(spec['dataset_dir'], 'merge_val_phase_V10.pkl'),
                     os.path.join(spec['dataset_dir'], 'merge_%s_phase_V10.pkl' % extra))
        with open(os.path.join(spec['dataset_dir'], '%s.txt' % extra), 'w',
                  encoding='utf-8') as f:
            f.write('/cad2graph/%s/\n' % res.sample_id)
        dst_dir = os.path.join(spec['dataset_dir'], 'img_dir', extra)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(os.path.join(spec['dataset_dir'], 'img_dir', 'val',
                                  '%s.png' % res.sample_id),
                     os.path.join(dst_dir, '%s.png' % res.sample_id))
    log.append('replicated the sample into train/val/test')

    # ---- 3) checkpoint + model run ---------------------------------------
    _stage_checkpoint(checkpoint, out_dir, log)
    res.run_dir = os.path.join(out_dir, 'CUBI', '0')
    res.triangles_npz = os.path.join(spec['work_dir'], 'triangles.npz')

    _run([python_exe, '-u', 'graphgym/main.py',
          '--cfg', 'graphgym/configs/CUBI.yaml', '--eval',
          'train.epoch_resume', '1', 'train.ckpt_prefix', 'best',
          'seed', '0', 'dataset.dir', spec['dataset_dir'], 'device', device,
          'out_dir', out_dir],
         cwd=vf_root, timeout=timeout_s, log=log)

    if not os.path.exists(os.path.join(res.run_dir, 'val_result.pkl')):
        raise VecFloorSegError('model run produced no val_result.pkl')
    res.ok = True
    return res


def available() -> bool:
    """Cheap capability probe used by the extractor to fail gracefully."""
    return os.path.isdir(DEFAULT_VF_ROOT)


if __name__ == '__main__':                                  # tiny CLI for debugging
    import argparse

    ap = argparse.ArgumentParser(description='VecFloorSeg adapter runner')
    ap.add_argument('spec')
    ap.add_argument('--python', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    with open(a.spec, 'r', encoding='utf-8') as f:
        sp = json.load(f)
    r = run_adapter(sp, python_exe=a.python, checkpoint=a.checkpoint,
                    out_dir=a.out_dir)
    print('ok=%s run_dir=%s n_regions=%d' % (r.ok, r.run_dir, r.n_regions))
    sys.exit(0 if r.ok else 1)

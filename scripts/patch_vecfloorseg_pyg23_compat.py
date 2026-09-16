#!/usr/bin/env python
"""PyG >= 2.3 compatibility shim for VecFloorSeg.

WHY
---
`third_party/VecFloorSeg/torch_geometric/graphgym/contrib/transform/graph_transform.py`
defines its OWN `to_undirected()` whose body ends with

    return coalesce(edge_index, edge_attr, num_nodes, reduce)

its docstring promises

    :rtype: LongTensor if `edge_attr` is None, else (LongTensor, Tensor|List[Tensor])

and `MyToUndirected1.__call__` relies on that promise:

    store[n] = to_undirected(store[n])        # n in edge_index_d2/d4/d8

However, since PyG 2.3 `coalesce()` returns a 2-tuple even when `edge_attr` is
None, so `edge_index_d2/d4/d8` silently become `(Tensor, None)`.  Downstream,
`message_passing.__lift__()` only accepts a Tensor or a SparseTensor and dies
with a bare `raise ValueError` (the fall-through branch), which is a very
misleading error.

This shim restores the documented contract, WITHOUT touching the submodule: it
patches the *installed* copy in site-packages, and must therefore be run again
every time the overlay is re-applied.

Idempotent: a second run reports "already patched".

Usage
    python scripts/patch_vecfloorseg_pyg23_compat.py            # patch + verify
    python scripts/patch_vecfloorseg_pyg23_compat.py --env-python <path/to/python.exe>
"""
import argparse
import os
import subprocess
import sys

MARKER = '# --- PyG>=2.3 compat shim'

OLD = '    return coalesce(edge_index, edge_attr, num_nodes, reduce)\n'

NEW = '''    _out = coalesce(edge_index, edge_attr, num_nodes, reduce)
    # --- PyG>=2.3 compat shim (CAD2Graph) ---------------------------------
    # PyG>=2.3 `coalesce()` returns a 2-tuple even when edge_attr is None,
    # while this function's contract (see :rtype: above) is to return a bare
    # LongTensor when edge_attr is None.  MyToUndirected1 depends on that:
    #     store[n] = to_undirected(store[n])      # edge_index_d2/d4/d8
    # Without this, those attributes become (Tensor, None) and
    # message_passing.__lift__() raises an opaque ValueError.
    if edge_attr is None:
        return _out[0] if isinstance(_out, tuple) else _out
    return _out
'''


def site_packages(python_exe):
    out = subprocess.run([python_exe, '-c',
                          'import site; print(site.getsitepackages()[-1])'],
                         capture_output=True, text=True)
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env-python', default=sys.executable,
                    help='python of the env holding the installed torch_geometric')
    args = ap.parse_args()

    sp = site_packages(args.env_python)
    target = os.path.join(sp, 'torch_geometric', 'graphgym', 'contrib',
                          'transform', 'graph_transform.py')
    print('python        : %s' % args.env_python)
    print('site-packages : %s' % sp)
    print('target        : %s' % target)

    if not os.path.isfile(target):
        print('!! file not found - is torch_geometric (with the overlay) installed?')
        return 2

    src = open(target, 'r', encoding='utf-8').read()

    if MARKER in src:
        print('already patched - nothing to do')
    else:
        n = src.count(OLD)
        if n == 0:
            print('!! anchor line not found; the vendor code may have changed:')
            print('   %r' % OLD.strip())
            return 3
        if n > 1:
            print('!! anchor line is not unique (%d occurrences); aborting' % n)
            return 4
        open(target, 'w', encoding='utf-8').write(src.replace(OLD, NEW))
        print('patched (1 occurrence replaced)')

    # ------------------------------------------------------------------ verify
    # (a) textual check first: needs no imports, so it can never be blocked by a
    #     not-yet-installed dependency (torch_geometric 2.6 imports fsspec at
    #     import time, which is why the runtime check below can legitimately fail
    #     on a partially provisioned env).
    patched = open(target, 'r', encoding='utf-8').read()
    if MARKER not in patched or 'if edge_attr is None:' not in patched:
        print('!! textual verification failed: the shim lines are not in the file')
        return 5
    print('textual check : OK (shim present in graph_transform.py)')

    # (b) runtime check: only meaningful once torch_geometric's own deps exist.
    code = (
        'import torch\n'
        'from torch_geometric.utils import coalesce\n'
        'ei = torch.tensor([[0, 1], [1, 0]])\n'
        'print("coalesce(ei, None) ->", type(coalesce(ei, None, None, "add")).__name__)\n'
        'from torch_geometric.graphgym.contrib.transform.graph_transform '
        'import to_undirected\n'
        'out = to_undirected(ei)\n'
        'print("local to_undirected(ei) ->", type(out).__name__, tuple(out.shape))\n'
        'assert not isinstance(out, tuple), "shim did not take effect"\n'
        'print("SHIM OK")\n'
    )
    r = subprocess.run([args.env_python, '-c', code], capture_output=True, text=True)
    print('--- runtime verify ---')
    print((r.stdout or '').strip())
    if r.returncode != 0:
        blob = (r.stdout or '') + (r.stderr or '')
        if 'ModuleNotFoundError' in blob:
            print('SKIPPED: torch_geometric cannot be imported yet (a dependency')
            print('         such as fsspec is probably missing).  The patch IS')
            print('         applied - re-run this script after the deps are')
            print('         installed to get the runtime confirmation.')
            return 0
        print(blob.strip())
        return 5
    return 0


if __name__ == '__main__':
    sys.exit(main())

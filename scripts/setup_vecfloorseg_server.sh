#!/usr/bin/env bash
#
# Set up the VecFloorSeg training environment on a Linux GPU server.
#
# Fidelity target = the author's EXACT stack, read from
# third_party/VecFloorSeg/requirements.txt:
#
#     python      = 3.8.15
#     pytorch     = 1.13.0  (py3.8_cuda11.7_cudnn8.5.0_0)
#     torchvision = 0.14.0  (py38_cu117)
#     torchaudio  = 0.13.0
#
# Verified GPU class: GTX 2080 Ti (Turing, sm_75). CUDA 11.7 supports sm_75
# natively, so we can stay on the author's stack instead of jumping to cu118.
#
# Usage
#   # recommended: install the CUDA wheels from a local directory (offline)
#   bash scripts/setup_vecfloorseg_server.sh --wheels-dir ~/vfs_wheels
#
#   # or let it fetch from the internet (needs a working proxy)
#   bash scripts/setup_vecfloorseg_server.sh --proxy http://10.0.0.1:7890
#
# Options
#   --env NAME         conda env name            (default: vecfloorseg)
#   --wheels-dir DIR   dir with the pre-downloaded CUDA wheels (torch,
#                      torchvision, torch_scatter, torch_sparse, torch_cluster,
#                      torch_spline_conv). Enables --no-index install.
#   --proxy URL        HTTP(S) proxy for conda/pip/downloads
#   --mirror URL       pip index-url for pure-python deps
#                      (default: Tsinghua)
#   --skip-backbones   do not download the 4 backbone .pth files
#   --with-preproc-deps  also install cv2 / skimage / triangle / svgpathtools
#                      (only needed to regenerate the dataset from CubiCasa-5k)
#   -h | --help        this text
#
set -euo pipefail

ENV_NAME="vecfloorseg"
WHEELS_DIR=""
PROXY=""
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
SKIP_BACKBONES=0
WITH_PREPROC=0

TORCH_VER="1.13.0"
TV_VER="0.14.0"
CUDA_TAG="cu117"
PYG_TAG="pt113cu117"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VF="$REPO_ROOT/third_party/VecFloorSeg"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)             ENV_NAME="$2"; shift 2 ;;
        --wheels-dir)      WHEELS_DIR="$2"; shift 2 ;;
        --proxy)           PROXY="$2"; shift 2 ;;
        --mirror)          MIRROR="$2"; shift 2 ;;
        --skip-backbones)  SKIP_BACKBONES=1; shift ;;
        --with-preproc-deps) WITH_PREPROC=1; shift ;;
        -h|--help)         sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

say()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

[[ -d "$VF" ]] || die "submodule not found: $VF  (git submodule update --init --recursive)"

if [[ -n "$PROXY" ]]; then
    export HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
    export http_proxy="$PROXY" https_proxy="$PROXY"
    say "proxy: $PROXY"
fi

# ---------------------------------------------------------------- 1) conda env
say "[1/8] conda env '$ENV_NAME' (python 3.8)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "already exists, reusing"
else
    conda create -y -n "$ENV_NAME" python=3.8
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
python -m pip install --quiet --upgrade pip setuptools wheel

# --------------------------------------------------- 2) torch + PyG extensions
say "[2/8] torch ${TORCH_VER}+${CUDA_TAG} + PyG extensions (${PYG_TAG})"
TORCH_PKGS=("torch==${TORCH_VER}+${CUDA_TAG}" "torchvision==${TV_VER}+${CUDA_TAG}")

if [[ -n "$WHEELS_DIR" ]]; then
    [[ -d "$WHEELS_DIR" ]] || die "--wheels-dir not found: $WHEELS_DIR"
    ls -1 "$WHEELS_DIR"/*.whl >/dev/null 2>&1 || die "no .whl files in $WHEELS_DIR"
    echo "CUDA wheels from : $WHEELS_DIR"
    echo "pure-python deps : $MIRROR"
    # torch / torchvision only exist with the +cu117 tag -> they resolve from the
    # local dir; their pure-python deps (filelock, sympy, typing-extensions,
    # jinja2, fsspec, networkx, pillow, requests...) resolve from the mirror.
    # (Do NOT use --no-index here: the deps are not in the local dir.)
    python -m pip install -i "$MIRROR" --find-links "$WHEELS_DIR" "${TORCH_PKGS[@]}"
    # The extensions only depend on torch, which is already in place.
    python -m pip install --no-index --find-links "$WHEELS_DIR" --no-deps \
        torch_scatter torch_sparse torch_cluster torch_spline_conv
else
    warn "no --wheels-dir: fetching from the internet (needs working access)"
    python -m pip install \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
        --extra-index-url "$MIRROR" \
        "${TORCH_PKGS[@]}"
    python -m pip install \
        torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
fi

# --------------------------------------------------------- 3) runtime deps
# MUST run BEFORE the overlay/shim: the shim verifies itself by importing
# torch_geometric, and PyG 2.6.1 imports fsspec at import time.
# Verified by grepping the import graph of graphgym / torch_geometric:
#   torch_geometric -> fsspec                (io/txt_array.py)
#   loader          -> PIL                    (CUBIMergeDualImgDataset_1.py)
#   train           -> tqdm                   (graphgym/train.py)
#   config          -> yacs                   (graphgym/config.py)
#   encoder/loader  -> ogb                    (only on some code paths, keep it)
#   NOTE torch_geometric/data/dataset.py uses the removed np.bool
#        => numpy MUST stay < 1.24
say "[3/8] runtime deps (numpy / pillow / tqdm / yacs / networkx / fsspec / ogb)"
python -m pip install -i "$MIRROR" "numpy==1.23.5" pillow tqdm yacs networkx fsspec
# --no-deps so ogb does not drag torch to a newer version
python -m pip install -i "$MIRROR" --no-deps "ogb==1.3.6" outdated

if [[ "$WITH_PREPROC" -eq 1 ]]; then
    say "[3b/8] optional dataset-preparation deps"
    # Only needed by DataPreparation/ , Replace_with_CubiCasa/ and Utils/.
    # Training/eval on the preprocessed CUBI_3 dataset does NOT touch them.
    # (triangle dropped cp38 wheels in recent releases -> pin an old one)
    python -m pip install -i "$MIRROR" \
        opencv-python-headless scikit-image svgpathtools matplotlib \
        "triangle==20220202"
fi

# ------------------------------------------------------------- 4) clean PyG
# Do NOT follow the repo README's "pyg==2.0.4": its vendored torch_geometric/
# is a cross-version partial overlay (307 files) that cannot be dropped onto any
# released PyG.  Verified working combination: clean PyG 2.6.1 + a TARGETED
# overlay of only torch_geometric/{graphgym,nn}.
say "[4/8] clean torch-geometric 2.6.1"
python -m pip install -i "$MIRROR" --force-reinstall --no-deps "torch-geometric==2.6.1"

# --------------------------------------------------- 5) targeted overlay
say "[5/8] targeted overlay: torch_geometric/{graphgym,nn}"
SP="$(python -c 'import site; print(site.getsitepackages()[-1])')"
echo "site-packages = $SP"
python - "$VF" "$SP" <<'PY'
import os
import shutil
import sys

src_root, site_packages = sys.argv[1], sys.argv[2]
for sub in ('graphgym', 'nn'):
    src = os.path.join(src_root, 'torch_geometric', sub)
    dst = os.path.join(site_packages, 'torch_geometric', sub)
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.pyc'):
                continue
            s = os.path.join(root, f)
            rel = os.path.relpath(s, src)
            d = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    print('  overlaid %-9s = %d files' % (sub, n))
PY

# ------------------------------------- 6) PyG>=2.3 compatibility shim
say "[6/8] PyG>=2.3 compatibility shim"
# MUST run after every overlay.  The vendor transform relies on its local
# `to_undirected()` returning a bare Tensor when edge_attr is None, but since
# PyG 2.3 `coalesce()` always returns a 2-tuple -> edge_index_d2/d4/d8 become
# tuples -> message_passing.__lift__() dies with a bare `raise ValueError`.
# Run this script again after any re-overlay.
python "$REPO_ROOT/scripts/patch_vecfloorseg_pyg23_compat.py" \
    --env-python "$(command -v python)" \
    || die "compat shim failed"

# --------------------------------------------------------- 7) backbone weights
# ImgEncoderDict instantiates ALL FOUR backbones at import time, so all four
# files must be present even though CUBI.yaml only uses resnet50 (~880 MB).
if [[ "$SKIP_BACKBONES" -eq 0 ]]; then
    say "[7/8] backbone weights (~880 MB)"
    mkdir -p "$VF/models"
    declare -A BB=(
        [resnet34-torch.pth]="https://download.pytorch.org/models/resnet34-b627a593.pth"
        [resnet50-torch.pth]="https://download.pytorch.org/models/resnet50-0676ba61.pth"
        [resnet101-torch.pth]="https://download.pytorch.org/models/resnet101-63fe2227.pth"
        [vgg16_bn-torch.pth]="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
    )
    for name in "${!BB[@]}"; do
        if [[ -s "$VF/models/$name" ]]; then
            echo "  $name already present"
            continue
        fi
        echo "  downloading $name"
        curl -fL --retry 3 -o "$VF/models/$name" "${BB[$name]}"
    done
    ls -lh "$VF/models"
else
    say "[7/8] backbones skipped"
fi

# ------------------------------------------------------------- 7) verify
say "[8/8] verify"
python - <<PY
import sys
import torch
print('python        :', sys.version.split()[0])
print('torch         :', torch.__version__)
print('cuda build    :', torch.version.cuda)
print('cudnn         :', torch.backends.cudnn.version())
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu           :', torch.cuda.get_device_name(0))
    print('capability    :', torch.cuda.get_device_capability(0))
    print('arch list     :', torch.cuda.get_arch_list())
    x = torch.randn(2048, 2048, device='cuda')
    print('matmul        : OK', float((x @ x).sum()))
import torchvision
print('torchvision   :', torchvision.__version__)
for m in ('torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv'):
    mod = __import__(m)
    print('%-14s: %s' % (m, getattr(mod, '__version__', '?')))
import torch_geometric
print('torch_geom    :', torch_geometric.__version__)
import torch_geometric.graphgym
print('graphgym      :', torch_geometric.graphgym.__file__)
from torch_geometric.nn import DeepGCNLayer, MyDeepGCNLayer, GENConv
print('custom nn     : OK')
PY

cat <<'EOT'

--------------------------------------------------------------------
Setup done.  Next: point the config at the dataset (no need to edit the
submodule - override on the command line).

  # 1. smoke test  (~1 epoch, validates data loading + VRAM)
  #    NOTE: max_epoch lives under `optim`, NOT `train`
  #    NOTE: train.iter_per_epoch is a NO-OP for sampler=full_batch
  #          -> 1 epoch == len(train_loader) == 524 iterations
  cd third_party/VecFloorSeg
  python graphgym/main.py --cfg graphgym/configs/CUBI.yaml seed 0 \
      dataset.dir /path/to/CUBI_3 optim.max_epoch 1

  # 2. training -- start with 50 epochs, then judge from the mIoU curve
  #    (measured: 1 epoch ~= 11 min on a 4070 Ti, so ~15-28 min on a 2080 Ti;
  #     200 epochs would be ~2-4 days, 50 epochs ~= half a day)
  #    num_workers 4 pays off on Linux: data loading is ~55% of a train step
  #    (0.589 s of 1.080 s/iter).  Do NOT raise it on Windows - spawn makes
  #    every worker re-import custom_graphgym and reload the 4 backbones (~880MB).
  python graphgym/main.py --cfg graphgym/configs/CUBI.yaml seed 0 \
      dataset.dir /path/to/CUBI_3 num_workers 4 optim.max_epoch 50

  # 3. inference / eval
  python graphgym/main.py --cfg graphgym/configs/CUBI.yaml --eval \
      train.epoch_resume 1 train.ckpt_prefix best val.extra_infos True \
      seed 0 dataset.dir /path/to/CUBI_3

  !! Do NOT write "--seed 0" (the repo README's Train command does, and it is
     BROKEN for this code version): cmd_args.py declares `opts` with
     nargs=argparse.REMAINDER, which only starts capturing at the first
     non-option token, so "--seed 0" dies with
         main.py: error: unrecognized arguments: --seed
     Overrides are `key value` pairs, and the FIRST one must not start with
     "--"  (that is why the README's Eval command happens to be correct).

Expected artifacts
  results/CUBI/config.yaml            resolved config dump
  results/CUBI/0/ckpt/best1.ckpt      THE CHECKPOINT
  results/CUBI/0/ckpt/last*.ckpt      periodic snapshots (ckpt_period)
  <run_dir>/{val,test}_result.pkl     predictions
                                      (filename, pred, gt, aux_pred, aux_gt, valid_edge)
  <run_dir>/{val,test}_result_extra.pkl  attention weights
                                      (only when val.extra_infos True)

WARNING: torch_geometric/graphgym/checkpoint.py::load_ckpt() returns 0 SILENTLY
when the checkpoint file is missing, so a "successful" eval run can be using
random weights.  Always confirm the .ckpt file exists.
--------------------------------------------------------------------
EOT

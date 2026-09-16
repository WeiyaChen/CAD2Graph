"""SAGE-E 训练 / 评测脚本（阶段 A：复现论文数字）。

设计要点
--------
* **不需要 DGL**。作者用 DGL 只是为了做消息传递，这里用 ``index_add_`` 等价实现，
  语义与 ``dgl.update_all(message_func, fn.sum('m', 'h_neigh'))`` 一致：
      agg[v] = Σ_{u→v} ReLU(W_msg [h_u ‖ e_uv])
      h_v    = ReLU(W_apply [h_v ‖ agg[v]])
  注意作者的 notebook 里 ``add_self_loop`` 是**注释掉的**，所以这里也不加自环。
* 论文配置：``bs=1, lr=0.005, weight_decay=5e-4, epochs=200, dropout=0.2``，
  CPU 上约 6 分钟。论文报告 test accuracy **0.7970** / macro-F1 **0.7801**。
* 数据划分严格照抄 notebook：
      trainvalid, test  = split(224, test_size=0.2, seed=42)   # 179 / 45
      train, valid      = split(179, test_size=0.1, seed=42)   # 161 / 18
  sklearn 的 ``train_test_split`` 内部就是 ``RandomState(42).permutation(n)``，
  所以用 numpy 就能逐位复现，不必引入 sklearn 依赖。

用法::

    # 复现论文（从头训练）
    python scripts/train_sagee.py --dataset data/roomgraph.npz --epochs 200

    # 直接评测作者给的两个权重
    python scripts/train_sagee.py --dataset data/roomgraph.npz \
        --eval-only --init-from third_party/SAGE-E/code/best_default.pt

    # 训练完导出成 numpy 可读的 .npz，供推理侧使用
    python scripts/train_sagee.py --dataset data/roomgraph.npz \
        --export-weights models/sagee_roomgraph.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def _load_standalone(name: str, relpath: str):
    """按文件路径直接加载 ``src/spatial/sagee/`` 下的独立模块。

    训练环境固定是 **Python 3.8**（VecFloorSeg 的 torch 1.13 把它锁死了），而
    ``src/spatial/__init__.py`` 会 eager import 所有轮廓算法，其中 ``rgp.py`` 用了
    PEP 585 的运行期下标（``tuple[float, float]``），需要 3.9+。
    ``sagee/{dataset,model}.py`` **只依赖 numpy**，所以这里绕开包机制直接加载，
    避免为了跑一个 15k 参数的小模型去动训练环境的 Python 版本。
    """
    import importlib.util

    module_path = os.path.join(ROOT, relpath)
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sagee_dataset = _load_standalone('sagee_dataset', os.path.join('src', 'spatial', 'sagee', 'dataset.py'))

NpzGraphDataset = _sagee_dataset.NpzGraphDataset
confusion = _sagee_dataset.confusion
macro_f1 = _sagee_dataset.macro_f1
split_indices = _sagee_dataset.split_indices
weighted_f1 = _sagee_dataset.weighted_f1
GraphSet = NpzGraphDataset        # 别名，保持下文的可读性

# --------------------------------------------------------------------------- #
# 模型：逐行对照作者的 code/SAGEE.py，只是把 DGL 的聚合换成 index_add_
# --------------------------------------------------------------------------- #


class SAGEELayer(nn.Module):
    """与作者同名同结构，便于直接 torch.load 作者的 .pt。"""

    def __init__(self, ndim_in: int, edims: int, ndim_out: int, activation=F.relu):
        super().__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def forward(self, nfeat, edge_index, efeat):
        src, dst = edge_index[0], edge_index[1]
        msg = self.activation(self.W_msg(torch.cat([nfeat[src], efeat], dim=1)))
        # 聚合结果的宽度是 ndim_out（DGL 的 fn.sum 输出与消息同宽），不是输入宽度
        agg = msg.new_zeros((nfeat.shape[0], msg.shape[1]))
        agg.index_add_(0, dst, msg)                 # == fn.sum('m', 'h_neigh')
        return self.activation(self.W_apply(torch.cat([nfeat, agg], dim=1)))


class SAGEE(nn.Module):
    """4 层 SAGE-E（论文用 4 层取得最好效果）。"""

    def __init__(self, ndim_in: int, ndim_out: int, edim: int, activation=F.relu,
                 dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            SAGEELayer(ndim_in, edim, 50, activation),
            SAGEELayer(50, edim, 50, activation),
            SAGEELayer(50, edim, 25, activation),
            SAGEELayer(25, edim, ndim_out, activation),
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nfeat, edge_index, efeat):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeat = self.dropout(nfeat)
            nfeat = layer(nfeat, edge_index, efeat)
        return nfeat


def _register_stub_module() -> None:
    """让 ``torch.load`` 能解开作者 pickle 出来的 ``SAGEE`` 对象。

    作者是 ``torch.save(model)``（整个对象）而不是 ``state_dict()``，pickle 里记录的
    类路径是 ``SAGEE.SAGEE``。把本模块的类注册成同名模块即可，无需 dgl。
    """
    mod = types.ModuleType('SAGEE')
    mod.SAGEE = SAGEE
    mod.SAGEELayer = SAGEELayer
    sys.modules.setdefault('SAGEE', mod)


# --------------------------------------------------------------------------- #
# 数据与指标：实现放在 src/spatial/sagee/dataset.py，与 numpy 回归测试共用
# --------------------------------------------------------------------------- #


def _to_torch(batch, device):
    nf, ei, ef, lb = batch
    return (
        torch.as_tensor(nf, dtype=torch.float32).to(device),
        torch.as_tensor(ei, dtype=torch.long).to(device),
        torch.as_tensor(ef, dtype=torch.float32).to(device),
        torch.as_tensor(lb, dtype=torch.long).to(device),
    )


def evaluate(model, ds: GraphSet, n_class: int, device='cpu', norm=None):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            nf, ei, ef, lb = _to_torch(apply_norm(ds.get(i), norm), device)
            logits = model(nf, ei, ef)
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_gt.append(lb.cpu().numpy())
    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    cm = confusion(pred, gt, n_class)
    acc = float((pred == gt).mean())
    return acc, macro_f1(cm), weighted_f1(cm), cm


def compute_norm(train_ds) -> dict:
    """由**训练集**统计逐列 z-score 参数。

    是否标准化由 ``--normalize`` 控制，**默认关闭** —— 原论文没有这一步，
    关着才与论文配方一致。

    ⚠️ 实测（5 折按图纸切分，CORE 14 类）**打开标准化反而更差**：
    准确率 0.6004 -> 0.5551、加权 F1 0.5720 -> 0.5273，且折间标准差从
    ±0.0366 涨到 ±0.0921。所以保留这个开关只是备查，“量纲差异大不标准化
    训不动”这个当初的假设是**错的**。

    ⚠️ 统计量必须**随权重一起落盘**并在推理时复用，否则输入分布对不上。
    """
    node = train_ds.gather('node_feat')
    edge = train_ds.gather('edge_feat')

    def _stats(x: np.ndarray):
        if x.shape[0] == 0:
            return np.zeros(x.shape[1]), np.ones(x.shape[1])
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0            # 常量列不缩放，避免除零
        return mean, std

    n_mean, n_std = _stats(node)
    e_mean, e_std = _stats(edge)
    return {
        'node_mean': n_mean.tolist(), 'node_std': n_std.tolist(),
        'edge_mean': e_mean.tolist(), 'edge_std': e_std.tolist(),
    }


def apply_norm(batch, norm):
    nf, ei, ef, lb = batch
    if norm is None:
        return batch
    nf = (nf - np.asarray(norm['node_mean'])) / np.asarray(norm['node_std'])
    if ef.shape[0]:
        ef = (ef - np.asarray(norm['edge_mean'])) / np.asarray(norm['edge_std'])
    return nf, ei, ef, lb


def export_weights(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    arrays = {}
    for k, v in model.state_dict().items():
        arrays[k.replace('.', '__')] = v.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)


def drop_stale_norm(weights_path: str) -> None:
    """删掉上次遗留的 ``<weights>.norm.json``（本次训练没开 ``--normalize`` 时）。

    ⚠️ 这不是洁癖：``SageeNumpy`` 会自动加载同名的 ``.norm.json``。如果那份参数
    是上一次开了 ``--normalize`` 的训练留下的，推理时就会被**静默**套到一个从未
    这样训练过的模型上 —— 输入分布直接错位，而且不报任何错。宁可删掉。
    """
    side = os.path.splitext(weights_path)[0] + '.norm.json'
    if os.path.isfile(side):
        os.remove(side)
        print('清除遗留标准化: %s（本次未启用 --normalize）' % side)


def write_recipe_meta(weights_path: str, *, args, n_class: int, dim_in: int,
                      dim_edge: int) -> None:
    """把训练配方写进 ``<weights>.meta.json``，供推理侧/自检核对。

    关键意义：让人一眼看出这份权重**是否**带着 z-score 和类别权重，否则只凭文件名
    无法区分「论文忠实配方」与「工程上更强但不等价的配方」。
    """
    meta = {
        'recipe': 'faithful' if not (args.normalize or args.class_weights) else 'tuned',
        'normalize': bool(args.normalize),
        'class_weights': bool(args.class_weights),
        'epochs': args.epochs, 'lr': args.lr, 'weight_decay': args.weight_decay,
        'dropout': args.dropout, 'seed': args.seed,
        'n_class': n_class, 'dim_node': dim_in, 'dim_edge': dim_edge,
    }
    with open(os.path.splitext(weights_path)[0] + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def make_folds(n_graph: int, n_fold: int, seed: int) -> list[list[int]]:
    """把图均分成 ``n_fold`` 份（每张图纸一张图，所以这是**按图纸**切分）。"""
    perm = np.random.RandomState(seed).permutation(n_graph)
    return [sorted(perm[i::n_fold].tolist()) for i in range(n_fold)]


def _load_graph_keys(dataset: str) -> list[str]:
    """从导出脚本写的 ``<dataset>.graphs.json`` 读回「图下标 → 图纸键」。"""
    sidecar = os.path.splitext(dataset)[0] + '.graphs.json'
    if not os.path.isfile(sidecar):
        return []
    with open(sidecar, encoding='utf-8') as f:
        data = json.load(f)
    return [g['key'] for g in data.get('graphs', [])]


def train_fold(train_ds, valid_ds, test_ds, *, args, norm=None, label=''):
    """训练一轮并评测，返回 ``(acc, mf1, wf1, cm, best_val, model)``。

    ``valid_ds=None`` 时不做验证集选模型，直接采用最后一轮的权重 —— 小数据集上
    验证集本身只有几张图，用"最佳验证"选模型等于用噪声选模型。
    """
    n_class = train_ds.n_class
    model = SAGEE(train_ds.dim_node, n_class, train_ds.dim_edge,
                  F.relu, args.dropout).to(args.device)

    weight = None
    if args.class_weights:
        counts = np.bincount(train_ds.gather('node_label').astype(np.int64),
                             minlength=n_class).astype(np.float64)
        weight = torch.as_tensor(
            np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0),
            dtype=torch.float32).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = -1.0, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, accs = [], []
        for i in range(len(train_ds)):
            nf, ei, ef, lb = _to_torch(apply_norm(train_ds.get(i), norm), args.device)
            logits = model(nf, ei, ef)
            loss = F.cross_entropy(logits, lb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            accs.append(float((logits.argmax(1) == lb).float().mean()))

        vacc = float('nan')
        if valid_ds is not None and len(valid_ds):
            with torch.no_grad():
                vacc, _, _, _ = evaluate(model, valid_ds, n_class, args.device, norm)
            if vacc > best_val:
                best_val = vacc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.verbose and (epoch % 20 == 0 or epoch == 1):
            print('  %sEpoch %03d train | Acc %.4f | Loss %.4f%s'
                  % (label, epoch, float(np.mean(accs)), float(np.mean(losses)),
                     '' if valid_ds is None else ' || Val | Acc %.4f' % vacc))

    if args.select == 'val' and best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.time() - t0
    if args.verbose:
        print('  %s训练耗时 %.1f s (%d epochs)%s'
              % (label, elapsed, args.epochs,
                 '' if best_state is None else '，采用最佳验证权重(val %.4f)' % best_val))

    acc, mf1, wf1, cm = evaluate(model, test_ds, n_class, args.device, norm)
    return acc, mf1, wf1, cm, best_val, model


def main() -> int:
    ap = argparse.ArgumentParser(description='SAGE-E train / eval (no DGL)')
    ap.add_argument('--dataset', default='data/roomgraph.npz')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--init-from', default=None, help='作者的 .pt（整对象）或本脚本导出的 .npz')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--export-weights', default=None, help='训练后导出 .npz 供 numpy 推理')
    ap.add_argument('--normalize', action='store_true',
                    help='逐列 z-score。⚠️ 原论文没有这一步（RoomGraph 的特征本来就在'
                         '小范围内）—— 打开就不再是论文忠实配方，默认关闭。')
    ap.add_argument('--norm-out', default=None, help='标准化参数写出路径（与权重一同使用）')
    ap.add_argument('--norm-from', default=None, help='载入已有的标准化参数（评测时必须与训练时一致）')
    ap.add_argument('--class-weights', action='store_true',
                    help='按类频次反比加权交叉熵。⚠️ 原论文用纯交叉熵 —— 打开就不是'
                         '论文忠实配方，默认关闭。')
    ap.add_argument('--labels-from', default=None,
                    help='标签空间 json；会复制到 <weights 同名>.labels.json，'
                         '使推理侧的一整套产物（npz/norm.json/labels.json）齐全')
    ap.add_argument('--emit-folds', default=None,
                    help='把每折的模型导到该目录（fold_k.npz/.norm.json/.labels.json '
                         '+ folds.json），供「留出法在线评估」用')
    ap.add_argument('--cv', type=int, default=1,
                    help='>1 时按图纸做 K-fold 交叉验证（小数据集上唯一可信的口径）')
    ap.add_argument('--select', choices=['val', 'final'], default='val',
                    help='val=用验证集选最佳轮次（默认，与 RoomGraph 复现一致）；'
                         'final=直接取最后一轮（小数据集上更诚实）')
    ap.add_argument('--quiet', dest='verbose', action='store_false', default=True,
                    help='CV 模式下默认关闭逐 epoch 输出')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_all = NpzGraphDataset(args.dataset)
    train_idx, valid_idx, test_idx = split_indices(ds_all.n_graph_total, seed=args.seed)
    train_ds = GraphSet(args.dataset, train_idx)
    valid_ds = GraphSet(args.dataset, valid_idx)
    test_ds = GraphSet(args.dataset, test_idx)

    n_class = ds_all.n_class
    dim_in = ds_all.dim_node
    dim_e = ds_all.dim_edge

    print('数据集      : %s' % args.dataset)
    print('图总数      : %d' % ds_all.n_graph_total)
    if args.cv > 1 and not args.eval_only:
        print('评估口径    : %d-fold 交叉验证（每张图纸一张图，即按图纸切分）' % args.cv)
    else:
        print('图  train/valid/test : %d / %d / %d'
              % (len(train_ds), len(valid_ds), len(test_ds)))
        print('节点 train/valid/test : %d / %d / %d'
              % (train_ds.n_nodes(), valid_ds.n_nodes(), test_ds.n_nodes()))
    print('特征        : node=%d  edge=%d   classes=%d' % (dim_in, dim_e, n_class))

    # ---------------------------------------------------------------- K-fold
    if args.cv > 1 and not args.eval_only:
        folds = make_folds(ds_all.n_graph_total, args.cv, args.seed)
        # 图下标 -> 图纸键（graphs.json 是导出脚本写的，含每张图的 key）
        graph_keys = _load_graph_keys(args.dataset)
        cm_total = np.zeros((n_class, n_class), dtype=np.int64)
        rows = []
        fold_manifest = []
        for k, fold_test in enumerate(folds):
            held = set(fold_test)
            rest = [i for i in range(ds_all.n_graph_total) if i not in held]
            tr = NpzGraphDataset(args.dataset, rest)
            te = NpzGraphDataset(args.dataset, fold_test)
            # 标准化参数按 fold 重新统计，避免验证信息泄漏
            fold_norm = compute_norm(tr) if args.normalize else None
            a, f1, w1, cm, _bv, fold_model = train_fold(
                tr, None, te, args=args, norm=fold_norm,
                label='fold %d/%d ' % (k + 1, args.cv))
            cm_total += cm
            test_keys = [graph_keys[i] for i in fold_test] if graph_keys else []
            rows.append({'fold': k + 1, 'n_test_graphs': len(fold_test),
                         'n_test_nodes': te.n_nodes(), 'accuracy': a,
                         'macro_f1': f1, 'weighted_f1': w1,
                         'test_keys': test_keys})
            print('  fold %d/%d: acc %.4f  macro-F1 %.4f  weighted-F1 %.4f  (%d 图 / %d 节点)'
                  % (k + 1, args.cv, a, f1, w1, len(fold_test), te.n_nodes()))

            if args.emit_folds:
                os.makedirs(args.emit_folds, exist_ok=True)
                wpath = os.path.join(args.emit_folds, 'fold_%d.npz' % (k + 1))
                export_weights(fold_model, wpath)
                if fold_norm is not None:
                    with open(os.path.join(args.emit_folds, 'fold_%d.norm.json' % (k + 1)),
                              'w', encoding='utf-8') as f:
                        json.dump(fold_norm, f, indent=2)
                else:
                    drop_stale_norm(wpath)
                write_recipe_meta(wpath, args=args, n_class=n_class,
                                  dim_in=dim_in, dim_edge=dim_e)
                if args.labels_from:
                    import shutil
                    shutil.copyfile(args.labels_from,
                                    os.path.join(args.emit_folds, 'fold_%d.labels.json' % (k + 1)))
                fold_manifest.append({'fold': k + 1, 'weights': os.path.abspath(wpath),
                                      'test_keys': test_keys})

        if args.emit_folds and fold_manifest:
            with open(os.path.join(args.emit_folds, 'folds.json'), 'w', encoding='utf-8') as f:
                json.dump({'dataset': os.path.abspath(args.dataset),
                           'n_graph_total': ds_all.n_graph_total,
                           'folds': fold_manifest}, f, indent=2, ensure_ascii=False)
            print('折清单导出  : %s' % os.path.join(args.emit_folds, 'folds.json'))

        accs = np.array([r['accuracy'] for r in rows])
        f1s = np.array([r['macro_f1'] for r in rows])
        w1s = np.array([r['weighted_f1'] for r in rows])
        pooled_acc = float(np.diag(cm_total).sum() / max(1, cm_total.sum()))
        pooled_f1 = macro_f1(cm_total)

        print('\n===== %d-fold 交叉验证汇总 =====' % args.cv)
        print('Accuracy    : %.4f ± %.4f   (合并后 %.4f)' % (accs.mean(), accs.std(), pooled_acc))
        print('Macro F1    : %.4f ± %.4f   (合并后 %.4f)' % (f1s.mean(), f1s.std(), pooled_f1))
        print('Weighted F1 : %.4f ± %.4f   (合并后 %.4f)'
              % (w1s.mean(), w1s.std(), weighted_f1(cm_total)))
        print('合并混淆矩阵 (行=真实, 列=预测):')
        print(cm_total)

        if args.export_weights or args.norm_out:
            # 部署用的模型：在**全部**图纸上再训一次（不再留测试集）
            full = NpzGraphDataset(args.dataset)
            full_norm = compute_norm(full) if args.normalize else None
            print('\n在全量 %d 张图上训练部署模型 ...' % full.n_graph_total)
            _a, _f, _w, _cm, _bv, final_model = train_fold(
                full, None, full, args=args, norm=full_norm, label='final ')
            if args.export_weights:
                export_weights(final_model, args.export_weights)
                print('权重导出    : %s' % args.export_weights)
                if args.labels_from:
                    import shutil
                    dst = os.path.splitext(args.export_weights)[0] + '.labels.json'
                    shutil.copyfile(args.labels_from, dst)
                    print('类别表导出  : %s' % dst)
            if full_norm is not None and args.norm_out:
                with open(args.norm_out, 'w', encoding='utf-8') as f:
                    json.dump(full_norm, f, indent=2)
                print('标准化导出  : %s' % args.norm_out)
            elif args.export_weights:
                drop_stale_norm(args.export_weights)
            if args.export_weights:
                write_recipe_meta(args.export_weights, args=args, n_class=n_class,
                                  dim_in=dim_in, dim_edge=dim_e)
                print('训练配方    : %s（normalize=%s, class_weights=%s）'
                      % ('faithful' if not (args.normalize or args.class_weights) else 'tuned',
                         args.normalize, args.class_weights))

        if args.out_json:
            with open(args.out_json, 'w', encoding='utf-8') as f:
                json.dump({'cv': args.cv, 'folds': rows,
                           'accuracy_mean': float(accs.mean()),
                           'accuracy_std': float(accs.std()),
                           'macro_f1_mean': float(f1s.mean()),
                           'macro_f1_std': float(f1s.std()),
                           'weighted_f1_mean': float(w1s.mean()),
                           'pooled_accuracy': pooled_acc, 'pooled_macro_f1': pooled_f1,
                           'confusion_matrix': cm_total.tolist()}, f, indent=2)
        return 0

    # 标准化参数：优先载入（评测必须与训练一致），否则按需统计
    norm = None
    if args.norm_from:
        with open(args.norm_from, encoding='utf-8') as f:
            norm = json.load(f)
        print('标准化      : 载入 %s' % args.norm_from)
    elif args.normalize:
        norm = compute_norm(train_ds)
        print('标准化      : 由训练集统计逐列 z-score')

    model = SAGEE(dim_in, n_class, dim_e, F.relu, args.dropout).to(args.device)

    if args.init_from:
        if args.init_from.endswith('.npz'):
            sd = {k.replace('__', '.'): torch.as_tensor(v)
                  for k, v in np.load(args.init_from).items()}
            model.load_state_dict(sd)
            print('载入权重     : %s (npz)' % args.init_from)
        else:
            _register_stub_module()
            loaded = torch.load(args.init_from, map_location=args.device)
            model = loaded if isinstance(loaded, nn.Module) else model
            model.to(args.device)
            print('载入权重     : %s (torch object)' % args.init_from)
        print('参数量       : %d' % sum(p.numel() for p in model.parameters()))

    if args.eval_only:
        acc, mf1, wf1, cm = evaluate(model, test_ds, n_class, args.device, norm)
        print('\n===== TEST =====')
        print('Accuracy    : %.4f    (论文 0.7970)' % acc)
        print('Macro F1    : %.4f    (论文 0.7801)' % mf1)
        print('Weighted F1 : %.4f' % wf1)
        print('Confusion matrix (行=真实, 列=预测):')
        print(cm)
        if args.export_weights:
            export_weights(model, args.export_weights)
            print('权重导出    : %s' % args.export_weights)
        if args.out_json:
            with open(args.out_json, 'w', encoding='utf-8') as f:
                json.dump({'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1,
                           'confusion_matrix': cm.tolist()}, f, indent=2)
        return 0

    # 类别加权（长尾数据集上能明显改善 macro-F1）
    acc, mf1, wf1, cm, best_val, model = train_fold(
        train_ds, valid_ds, test_ds, args=args, norm=norm)

    print('\n===== TEST =====')
    print('Accuracy    : %.4f' % acc)
    print('Macro F1    : %.4f' % mf1)
    print('Weighted F1 : %.4f' % wf1)
    if not np.isnan(best_val):
        print('最佳验证    : %.4f' % best_val)
    print('Confusion matrix (行=真实, 列=预测):')
    print(cm)

    if args.export_weights:
        export_weights(model, args.export_weights)
        print('权重导出    : %s' % args.export_weights)
    if norm is not None and args.norm_out:
        with open(args.norm_out, 'w', encoding='utf-8') as f:
            json.dump(norm, f, indent=2)
        print('标准化导出  : %s' % args.norm_out)
    elif args.export_weights:
        drop_stale_norm(args.export_weights)
    if args.export_weights:
        write_recipe_meta(args.export_weights, args=args, n_class=n_class,
                          dim_in=dim_in, dim_edge=dim_e)
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump({'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1,
                       'best_val_accuracy': best_val,
                       'confusion_matrix': cm.tolist()}, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

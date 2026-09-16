"""VecFloorSeg input bridge — CAD2Graph geometry -> the author's dataset format.

Runs in the **vecfloorseg** conda env (needs numpy / triangle / matplotlib and the
author's own modules from ``third_party/VecFloorSeg``).  It is invoked as a
subprocess by ``runner.py`` and must never be imported from the cadruler env.

Why this exists
---------------
The author's preprocessing chain is ``model.svg -> SVGParserCUBI ->
delaunayTriangulation -> graph building -> pkl``.  The SVG only serves to build a
**PSLG** (``{vertex: [id, neighbours...]}``), and CAD2Graph already *has* that
geometry, so we skip SVG parsing entirely and reuse every algorithmic step:

    isLineIntersection -> extendCornerWall -> tr.triangulate
    -> _genTriangleGraph -> buildDualRelationship -> graphCrune

Two deliberate deviations, both verified safe:

1. **No annotation raster voting.**  ``triangleCorrespondingLabel`` would need
   skimage + a palette PNG.  ``graphCrune``'s region merging is 100 % topological
   (labels are only read *after* grouping, to compute the per-region label and the
   PARTITION flag), so for inference we synthesise ``triangles_label = 255``
   (IGNORE) and compute ``triangles_area`` with the shoelace formula.  Every
   geometry-bearing output is bit-identical to a run with perfect labels.
2. **We keep the triangle table.**  The merged pkl has no ``triangles`` key, so
   room polygons cannot be recovered from the prediction alone — we dump
   ``triangles.npz`` for the output bridge.

Input:  a JSON spec (pixel coordinates, see ``runner.py`` / ``geometry.py``)
Output: <dataset_dir>/merge_<split>_phase_V10.pkl
        <dataset_dir>/<split>.txt
        <dataset_dir>/img_dir/<split>/<sample_id>.png
        <work_dir>/triangles.npz
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# the author's modules live at <VecFloorSeg>/ ; make them importable
# --------------------------------------------------------------------------- #
VFS_ROOT = os.environ.get('VECFLOORSEG_ROOT') or os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'VecFloorSeg')
)
if VFS_ROOT not in sys.path:
    sys.path.insert(0, VFS_ROOT)

from DataPreparation.SvgProcessing_CubiCasa import (  # noqa: E402
    _genTriangleGraph,
    buildDualRelationship,
)
from Utils.extendWall import extendCornerWall  # noqa: E402
from Utils.graphicsUtilsRe import graphCrune, isLineIntersection  # noqa: E402

import triangle as tr  # noqa: E402

# the door/window slots are SWAPPED by the author's parser (see the design doc §3.1):
#   primitiveDoors  <- <g id="Window">   (drawn black, typed DOORTYPE  = 2)
#   primitiveWindows<- <g id="Door">     (drawn blue,  typed WINDOWTYPE = 3)
# We therefore feed windows into the ``doors`` slot below to stay consistent with
# the trained model's feature semantics.
SCALE_COEFF = 50          # author's `s`: coords are multiplied by this internally
IMG_SIZE = 512            # released images are 512x512 RGBA

# Synthetic label for every triangle.  The author's ``IGNORE_LABEL = 255`` is
# NOT usable here: ``graphgym/contrib/loss/data_imbalance_loss.py`` builds a
# one-hot via ``scatter_`` over ``dim_out`` (= 12) classes, so a 255 target trips
# a CUDA device-side assert ("index out of bounds" in ScatterGatherKernel).
# Region merging, the merged centroids and every edge mapping are provably
# independent of the labels (see docs/vecfloorseg_adapter_design.md §2.3), so any
# valid class id is fine and the main head is unaffected.
SYNTHETIC_LABEL = 0


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def build_vertices_dict(walls_px, door_quads, window_quads, canvas_w, canvas_h):
    """Build the author's ``verticesDict``: ``{(x, y): [self_idx, nbr_idx, ...]}``.

    ``walls_px`` are 2-point segments (wall centrelines); openings are 4-corner
    quads.  Every polygon/segment contributes consecutive-corner adjacency, which
    is exactly what ``SVGParserCUBI.getWallShape`` produces from the CubiCasa
    ``<polygon>`` children.
    """
    vertices: dict[tuple[int, int], list[int]] = {}

    def add_edge(p, q):
        """Register an edge, assigning ids in first-seen order (the author's
        ``getWallShape`` returns ``{coord: [self_id, neighbour_id, ...]}``)."""
        kp = (int(round(p[0])), int(round(p[1])))
        kq = (int(round(q[0])), int(round(q[1])))
        if kp == kq:
            return
        if kp not in vertices:
            vertices[kp] = [len(vertices)]
        if kq not in vertices:
            vertices[kq] = [len(vertices)]
        ip, iq = vertices[kp][0], vertices[kq][0]
        if iq not in vertices[kp]:
            vertices[kp].append(iq)
        if ip not in vertices[kq]:
            vertices[kq].append(ip)

    for x1, y1, x2, y2 in walls_px:
        add_edge((x1, y1), (x2, y2))

    # NOTE: the quad corners participate in the PSLG so openings act as barriers.
    for quads in (door_quads, window_quads):
        for quad in quads:
            for i in range(len(quad)):
                add_edge(quad[i], quad[(i + 1) % len(quad)])

    # 4 canvas border corners + the border rectangle (author: pointsBdary)
    max_x, max_y = canvas_w - 1, canvas_h - 1
    corners = [(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)]
    for i, pt in enumerate(corners):
        add_edge(pt, corners[(i + 1) % 4])
    return vertices


def quad_vertex_ids(vertices_by_coord, quads):
    """Map quads to their vertex ids (as the author's pDoors/pWindows lists)."""
    out = []
    for quad in quads:
        ids = []
        for p in quad:
            key = (int(round(p[0])), int(round(p[1])))
            if key in vertices_by_coord:
                ids.append(vertices_by_coord[key][0])
        if len(ids) == 4:
            out.append(ids)
    return out


def triangle_areas(vertices, triangles):
    """Shoelace area per triangle (replaces ``triangleCorrespondingLabel``)."""
    pts = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles)]
    x, y = pts[:, :, 0], pts[:, :, 1]
    area = 0.5 * np.abs(
        x[:, 0] * (y[:, 1] - y[:, 2])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 0] - y[:, 1])
    )
    return area.reshape(-1, 1)


def vertex_coord_map(vertices_dict):
    """``{vertex_id: (x, y)}`` — the author's dict is keyed by coordinate."""
    return {v[0]: k for k, v in vertices_dict.items()}


# --------------------------------------------------------------------------- #
# the adapted ``worker()``
# --------------------------------------------------------------------------- #
def build_sample(spec):
    canvas_w, canvas_h = int(spec['canvas']['w']), int(spec['canvas']['h'])
    walls_px = spec['walls_px']
    door_quads = spec['doors_px']       # fed into the "pDoors" slot  (see header)
    window_quads = spec['windows_px']   # fed into the "pWindows" slot

    vertices_dict = build_vertices_dict(walls_px, window_quads, door_quads,
                                        canvas_w, canvas_h)
    coord_of = vertex_coord_map(vertices_dict)
    p_doors = quad_vertex_ids(vertices_dict, door_quads)
    p_windows = quad_vertex_ids(vertices_dict, window_quads)
    excluded = [i for q in (p_doors + p_windows) for i in q]

    # ---- delaunayTriangulation (verbatim flow, image lookup removed) --------
    vertices, edge_index = isLineIntersection(vertices_dict, scaleCoeff=SCALE_COEFF)
    walls = dict(vertices=vertices, segments=edge_index)
    walls, extend_walls = extendCornerWall(
        walls, min(canvas_w - 1, canvas_h - 1), scaleCoeff=SCALE_COEFF,
        excludeVIdxes=excluded,
    )

    num_extend1_vs = len(walls['vertices'])
    vertices, edge_index = isLineIntersection(walls, scaleCoeff=1)
    walls = dict(vertices=vertices, segments=edge_index)
    for v_idx in range(num_extend1_vs, len(vertices)):
        for edge in edge_index:
            if edge[0] == v_idx or edge[1] == v_idx:
                extend_walls.append(edge)

    seg_walls = tr.triangulate(walls, 'p')

    # ---- labels/areas (author: triangleCorrespondingLabel) -----------------
    seg_walls['triangles_label'] = np.full(
        (len(seg_walls['triangles']), 1), float(SYNTHETIC_LABEL))
    seg_walls['triangles_area'] = triangle_areas(seg_walls['vertices'],
                                                 seg_walls['triangles'])

    # ---- triangle adjacency graph (author: _genTriangleGraph) --------------
    f_wall = [(min(*s), max(*s)) for s in seg_walls['segments']]
    t_edges, t_edge_attr, t_edge_type, nodes, node_attr, t_edge_dual, whole_edge_set = \
        _genTriangleGraph(seg_walls['vertices'], seg_walls['triangles'], f_wall,
                          extend_walls, p_doors, p_windows)

    fplan_veno = {
        'vertices': nodes, 'x': node_attr,
        'segments': t_edges, 'segment_attr': t_edge_attr,
        'segments_type': t_edge_type, 'scale_coeff': SCALE_COEFF,
        'segment_dual': t_edge_dual,
    }
    seg_walls = buildDualRelationship(segWalls=seg_walls, venoGraph=fplan_veno,
                                      edgeTriPairs=whole_edge_set)
    merge_seg_walls, merge_fplan_veno = graphCrune(seg_walls, fplan_veno, walls,
                                                   whole_edge_set)
    return {
        'entry': [merge_seg_walls, merge_fplan_veno],
        'walls': walls,
        'seg_walls': seg_walls,
        'p_doors': p_doors,
        'p_windows': p_windows,
        'vertices_dict': vertices_dict,
        'coord_of': coord_of,
    }


# --------------------------------------------------------------------------- #
# image rendering (mirrors ImageProcessing_CubiCasa.worker)
# --------------------------------------------------------------------------- #
def render_image(result, canvas_w, canvas_h, out_png, img_size=IMG_SIZE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from PIL import Image

    walls = result['walls']
    seg_walls = result['seg_walls']
    v_dict = result['vertices_dict']
    coord_of = result['coord_of']

    def seg_pts(vertex_array, segment_list):
        va = np.asarray(vertex_array, dtype=float)
        out = []
        for a, b in segment_list:
            out.append([va[int(a)], va[int(b)]])
        return out

    fig = plt.figure(figsize=(canvas_w / 100.0, canvas_h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)          # image-down, like the released renders
    ax.set_aspect('equal')
    ax.axis('off')

    # black: wall PSLG after extension
    ax.add_collection(LineCollection(seg_pts(walls['vertices'], walls['segments']),
                                     colors='black', linewidths=0.5))
    # black: the "pDoors" slot == WINDOWS (author's swap)
    for quad in result['p_doors']:
        pts = np.array([coord_of[i] for i in quad] + [coord_of[quad[0]]], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color='black', linewidth=0.5)
    # blue: the "pWindows" slot == DOORS
    for quad in result['p_windows']:
        pts = np.array([coord_of[i] for i in quad] + [coord_of[quad[0]]], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color='blue', linewidth=0.5)
    # red: the full constrained-triangulation segment list (present in the
    # released images, absent from the published script)
    ax.add_collection(LineCollection(
        seg_pts(seg_walls['vertices'], seg_walls['segments']),
        colors='red', linewidths=0.5))

    fig.canvas.draw()
    try:
        raw = fig.canvas.tostring_rgb()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), raw)
    except AttributeError:                       # matplotlib >= 3.10
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)

    # letterbox to a square (the released set is 512x512 with white padding)
    w, h = img.size
    scale = float(img_size) / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.NEAREST)
    canvas = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    canvas.paste(img, ((img_size - nw) // 2, (img_size - nh) // 2))
    canvas.save(out_png)
    return out_png


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True, help='JSON geometry spec')
    args = ap.parse_args()

    with open(args.spec, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    sample_id = str(spec['sample_id'])
    canvas_w, canvas_h = int(spec['canvas']['w']), int(spec['canvas']['h'])
    dataset_dir = spec['dataset_dir']
    work_dir = spec['work_dir']
    split = spec.get('split', 'val')

    result = build_sample(spec)

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    img_dir = os.path.join(dataset_dir, 'img_dir', split)
    os.makedirs(img_dir, exist_ok=True)

    pkl_path = os.path.join(dataset_dir, 'merge_%s_phase_V10.pkl' % split)
    with open(pkl_path, 'wb') as f:
        pickle.dump({sample_id: result['entry']}, f)

    txt_path = os.path.join(dataset_dir, '%s.txt' % split)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('/cad2graph/%s/\n' % sample_id)

    png_path = os.path.join(img_dir, '%s.png' % sample_id)
    render_image(result, canvas_w, canvas_h, png_path)

    # the triangle table the merged pkl does not keep (needed by postprocess)
    merge2tri = result['entry'][1]['merge2tri']
    np.savez_compressed(
        os.path.join(work_dir, 'triangles.npz'),
        vertices=np.asarray(result['seg_walls']['vertices'], dtype=np.float64),
        triangles=np.asarray(result['seg_walls']['triangles'], dtype=np.int64),
        triangles_type=np.asarray(result['walls']['segments'], dtype=np.int64),
        scale_coeff=np.asarray([SCALE_COEFF], dtype=np.int64),
        canvas=np.asarray([[canvas_w, canvas_h]], dtype=np.int64),
        merge2tri_ids=np.asarray(sorted(merge2tri.keys()), dtype=np.int64),
        merge2tri_flat=np.concatenate([np.asarray(merge2tri[k], dtype=np.int64)
                                       for k in sorted(merge2tri.keys())]),
        merge2tri_sizes=np.asarray([len(merge2tri[k])
                                    for k in sorted(merge2tri.keys())],
                                   dtype=np.int64),
    )
    print(json.dumps({
        'sample_id': sample_id,
        'pkl': pkl_path,
        'txt': txt_path,
        'png': png_path,
        'n_regions': int(result['entry'][1]['vertices'].shape[0]),
        'n_triangles': int(len(result['seg_walls']['triangles'])),
        'work_dir': work_dir,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()

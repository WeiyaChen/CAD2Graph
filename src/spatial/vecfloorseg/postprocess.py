"""Output bridge: VecFloorSeg per-region predictions -> room polygons in mm.

Runs in the **cadruler** env (needs only numpy + shapely).

Why the triangle table is needed: the author's merged pkl keeps ``merge2tri``
(region -> triangle ids) but *not* the triangle->vertex table, so a region's
geometry cannot be rebuilt from the prediction alone.  ``build_dataset.py``
therefore dumps ``triangles.npz`` alongside, and this module dissolves the
triangles of each predicted label into polygons.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import numpy as np

# ``ISpatialContourExtractor`` contract: closed exterior rings, millimetres.
# Class ids used by the author's CUBI label space:
#   0 background, 1 Outdoor, 2 Wall, 3 Kitchen, 4 Dining, 5 Bedroom, 6 Bath,
#   7 Entrance, 8 Railing, 9 Closet, 10 Garage, 11 Corridor
DEFAULT_ROOM_LABELS = frozenset({3, 4, 5, 6, 7, 9, 10, 11})
# 1 (Outdoor) is explicitly excluded: it is the exterior region.
EXCLUDED_LABELS = frozenset({0, 1, 2, 8})


class TriangleTable:
    """The triangulation the bridge kept, in **scaled** author coordinates."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        self.vertices = np.asarray(data['vertices'], dtype=np.float64)   # (N,2) *50
        self.triangles = np.asarray(data['triangles'], dtype=np.int64)   # (T,3)
        self.scale_coeff = float(np.asarray(data['scale_coeff']).reshape(-1)[0])
        self.canvas = np.asarray(data['canvas']).reshape(-1)             # (w,h)
        ids = np.asarray(data['merge2tri_ids'], dtype=np.int64)
        flat = np.asarray(data['merge2tri_flat'], dtype=np.int64)
        sizes = np.asarray(data['merge2tri_sizes'], dtype=np.int64)
        self.merge2tri: dict[int, list[int]] = {}
        pos = 0
        for rid, size in zip(ids.tolist(), sizes.tolist()):
            self.merge2tri[int(rid)] = flat[pos:pos + size].tolist()
            pos += size

    def triangle_points_px(self) -> np.ndarray:
        """(T,3,2) triangle corners in **pixel** space (scaled coords undone)."""
        return self.vertices[self.triangles] / self.scale_coeff

    def label_per_triangle(self, region_pred: Sequence[int]) -> np.ndarray:
        """Expand per-region predictions to per-triangle labels."""
        out = np.full((len(self.triangles),), -1, dtype=np.int64)
        for region_id, tri_ids in self.merge2tri.items():
            if region_id >= len(region_pred):
                continue
            out[np.asarray(tri_ids, dtype=np.int64)] = int(region_pred[region_id])
        return out


def _polygonize(rings: Iterable[np.ndarray]) -> Any:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = []
    for ring in rings:
        if len(ring) < 3:
            continue
        try:
            poly = Polygon(ring)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        polys.append(poly)
    if not polys:
        return None
    return unary_union(polys)


def regions_to_contours(table: TriangleTable, region_pred: Sequence[int],
                        transform: dict[str, Any],
                        keep_labels: Iterable[int] | None = None,
                        min_area_mm2: float = 0.0,
                        simplify_tol_mm: float = 0.0,
                        shape_filter: Any = None) -> tuple[list[Any], list[Any]]:
    """Dissolve same-label triangles into room polygons.

    Returns ``(contours, debug_triangles_px)`` where ``contours`` is a list of
    ``SpatialContour`` and ``debug_triangles_px`` the (T,3,2) array used.
    """
    from src.spatial.domain import SpatialContour
    from shapely.geometry import Polygon

    keep = set(DEFAULT_ROOM_LABELS if keep_labels is None else keep_labels)
    tri_pts_px = table.triangle_points_px()
    tri_labels = table.label_per_triangle(region_pred)

    mm_per_px = 1.0 / float(transform['scale'])
    to_mm = _make_to_mm(transform)

    contours: list[Any] = []
    counter = 0
    for label in sorted(keep):
        sel = np.where(tri_labels == label)[0]
        if sel.size == 0:
            continue
        ring_px = _polygonize(tri_pts_px[sel])
        if ring_px is None:
            continue
        parts = list(getattr(ring_px, 'geoms', [ring_px]))
        for part in parts:
            if simplify_tol_mm > 0:
                part = part.simplify(simplify_tol_mm * mm_per_px, preserve_topology=True)
            poly_mm = _to_mm_polygon(part, to_mm)
            if poly_mm is None or poly_mm.area < min_area_mm2:
                continue
            if shape_filter is not None:
                try:
                    if shape_filter.check(poly_mm) is not None:
                        continue
                except Exception:
                    pass
            counter += 1
            contours.append(SpatialContour(
                id='Space_%03d' % counter,
                label='Unknown',
                geometry=[(float(x), float(y)) for x, y in poly_mm.exterior.coords],
            ))
    return contours, tri_pts_px


def _make_to_mm(transform: dict[str, Any]):
    scale = float(transform['scale'])
    off_x = float(transform['off_x'])
    off_y = float(transform['off_y'])
    bbox = transform['bbox']
    height = float(transform['height'])
    y0 = float(bbox[1])

    def to_mm(px: float, py: float) -> tuple[float, float]:
        x = (px - off_x) / scale
        y = y0 + (height - off_y - py) / scale
        return x, y

    return to_mm


def _to_mm_polygon(poly_px: Any, to_mm) -> Any:
    from shapely.geometry import Polygon
    try:
        coords = [to_mm(x, y) for x, y in poly_px.exterior.coords]
        out = Polygon(coords)
        return out.buffer(0) if not out.is_valid else out
    except Exception:
        return None


def load_predictions(result_pkl: str) -> tuple[list[int], str]:
    """Read ``pred`` (per merged region) from ``{val,test}_result.pkl``.

    Entry layout (``model.has_aux=True``):
        (filename, pred, gt, aux_pred, aux_gt, valid_edge)
    """
    import pickle

    with open(result_pkl, 'rb') as f:
        data = pickle.load(f)
    if not data:
        raise ValueError('empty prediction file: %s' % result_pkl)
    entry = data[0]
    filename = entry[0]
    if isinstance(filename, (list, tuple)):
        filename = filename[0] if filename else ''
    pred = np.asarray(entry[1]).reshape(-1).astype(int).tolist()
    return pred, str(filename)


def find_result_pkl(run_dir: str, prefer: str = 'val') -> str:
    """Locate ``{val,test}_result.pkl`` under a GraphGym run dir."""
    order = [prefer] + [s for s in ('val', 'test') if s != prefer]
    for split in order:
        p = os.path.join(run_dir, '%s_result.pkl' % split)
        if os.path.exists(p):
            return p
    raise FileNotFoundError('no *_result.pkl under %s' % run_dir)

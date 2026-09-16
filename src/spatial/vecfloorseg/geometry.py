"""CAD2Graph-side geometry handling for the VecFloorSeg adapter.

Two responsibilities, both living in the **cadruler** env (no torch here):

* :class:`CanvasTransform` — the lossless mm (CAD, y-up, arbitrary origin) to
  pixel (image, y-down, origin top-left) mapping.  The same object inverts the
  prediction back to millimetres, which is what makes the output bridge exact.
* :func:`build_spec` — turn the ``walls`` / ``doors`` / ``windows`` the pipeline
  hands to an ``ISpatialContourExtractor`` into the JSON spec consumed by
  ``vecfloorseg/build_dataset.py`` (see that module's docstring for the format).

Wall thickness: the pipeline only gives us wall **centrelines** (``clean_lines``
returns two-point ``LineString``s), whereas VecFloorSeg was trained on wall
*bands* (CubiCasa wall polygons).  ``wall_thickness_mm`` expands each centreline
into a 4-corner quad so the triangulation barriers and the rendered image match
the training distribution more closely; ``0`` keeps bare centrelines (fastest,
but the resulting rooms are each inflated by ~half a wall).
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

CANVAS_PX = 512


class CanvasTransform:
    """mm <-> pixel mapping with y-flip and centring.

    ``bbox`` is ``(min_x, min_y, max_x, max_y)`` in drawing millimetres.
    """

    def __init__(self, bbox: Sequence[float], width: int = CANVAS_PX,
                 height: int = CANVAS_PX, margin_ratio: float = 0.08):
        x0, y0, x1, y1 = (float(v) for v in bbox)
        span = max(x1 - x0, y1 - y0)
        if span <= 0:
            span = 1.0
        self.bbox = (x0, y0, x1, y1)
        self.width = int(width)
        self.height = int(height)
        self.margin_ratio = float(margin_ratio)
        self.scale = (min(self.width, self.height) * (1.0 - 2.0 * margin_ratio)) / span
        self.off_x = (self.width - (x1 - x0) * self.scale) / 2.0 - x0 * self.scale
        self.off_y = (self.height - (y1 - y0) * self.scale) / 2.0

    # -- forward -----------------------------------------------------------
    def to_px(self, x: float, y: float) -> tuple[float, float]:
        px = x * self.scale + self.off_x
        py = self.height - self.off_y - (y - self.bbox[1]) * self.scale
        return px, py

    def point_sets_mm_to_px(self, geometry: Iterable[Any]) -> list[list[tuple[float, float]]]:
        """Convert shapely-ish geometries to a list of pixel point rings."""
        out: list[list[tuple[float, float]]] = []
        for geom in geometry:
            ring = _exterior_coords(geom)
            if len(ring) >= 3:
                out.append([self.to_px(x, y) for x, y in ring])
        return out

    def extend_rings_px(self, geometry: Iterable[Any]) -> list[list[tuple[float, float]]]:
        """Same as :meth:`point_sets_mm_to_px` but for open polylines (walls)."""
        out: list[list[tuple[float, float]]] = []
        for geom in geometry:
            pts = _line_coords(geom)
            if len(pts) >= 2:
                out.append([self.to_px(x, y) for x, y in pts])
        return out

    # -- inverse -----------------------------------------------------------
    def to_mm(self, px: float, py: float) -> tuple[float, float]:
        x = (px - self.off_x) / self.scale
        y = self.bbox[1] + (self.height - self.off_y - py) / self.scale
        return x, y

    def ring_px_to_mm(self, ring: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
        return [self.to_mm(px, py) for px, py in ring]

    # -- helpers -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            'bbox': list(self.bbox), 'scale': self.scale,
            'off_x': self.off_x, 'off_y': self.off_y,
            'width': self.width, 'height': self.height,
            'margin_ratio': self.margin_ratio,
        }


def _exterior_coords(geom: Any) -> list[tuple[float, float]]:
    if geom is None:
        return []
    ext = getattr(geom, 'exterior', None)
    if ext is not None:                      # shapely Polygon
        return [(float(x), float(y)) for x, y in ext.coords]
    coords = getattr(geom, 'coords', None)
    if coords is not None:
        return [(float(x), float(y)) for x, y in coords]
    return [(float(p[0]), float(p[1])) for p in geom]


def _line_coords(geom: Any) -> list[tuple[float, float]]:
    if geom is None:
        return []
    coords = getattr(geom, 'coords', None)
    if coords is not None:
        return [(float(x), float(y)) for x, y in coords]
    ext = getattr(geom, 'exterior', None)
    if ext is not None:
        return [(float(x), float(y)) for x, y in ext.coords]
    return [(float(p[0]), float(p[1])) for p in geom]


def walls_to_quads_px(walls_px: list[list[tuple[float, float]]],
                      transform: CanvasTransform,
                      thickness_mm: float) -> list[tuple[float, ...]]:
    """Expand wall centrelines (pixel space) into 4-corner bands.

    Uses a simple axis-independent offset along the segment normal; returns
    ``(x1, y1, x2, y2, x3, y3, x4, y4)`` tuples so the bridge can consume them
    uniformly with the openings.
    """
    if thickness_mm <= 0:
        return []
    half_px = (thickness_mm * transform.scale) / 2.0
    quads: list[tuple[float, ...]] = []
    for pts in walls_px:
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            dx, dy = bx - ax, by - ay
            length = (dx * dx + dy * dy) ** 0.5
            if length < 1e-9:
                continue
            nx, ny = -dy / length * half_px, dx / length * half_px
            quads.append((ax + nx, ay + ny, bx + nx, by + ny,
                          bx - nx, by - ny, ax - nx, ay - ny))
    return quads


def bbox_of(points: Iterable[Any]) -> tuple[float, float, float, float]:
    """Bounding box over shapely-ish geometries (mm)."""
    xs: list[float] = []
    ys: list[float] = []
    for geom in points:
        bounds = getattr(geom, 'bounds', None)
        if bounds is not None:
            xs.extend([bounds[0], bounds[2]])
            ys.extend([bounds[1], bounds[3]])
            continue
        for x, y in _line_coords(geom):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs), min(ys), max(xs), max(ys))


def build_spec(sample_id: str, walls: Sequence[Any], doors: Sequence[Any],
               windows: Sequence[Any], dataset_dir: str, work_dir: str,
               split: str = 'val', canvas_px: int = CANVAS_PX,
               margin_ratio: float = 0.08, wall_thickness_mm: float = 0.0,
               transform: CanvasTransform | None = None) -> dict[str, Any]:
    """Assemble the JSON spec for ``vecfloorseg/build_dataset.py``.

    Coordinates are emitted in **pixels** (image space), because the author's
    whole preprocessing pipeline works in annotation-pixel space.
    """
    if transform is None:
        transform = CanvasTransform(bbox_of(list(walls) + list(doors) + list(windows)),
                                    canvas_px, canvas_px, margin_ratio)

    wall_lines_px = transform.extend_rings_px(walls)
    walls_px = [seg for pts in wall_lines_px
                for seg in ((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                            for i in range(len(pts) - 1))]

    # wall bands (optional) are extra barriers/geometry
    for quad in walls_to_quads_px(wall_lines_px, transform, wall_thickness_mm):
        for i in range(4):
            x1, y1 = quad[2 * i], quad[2 * i + 1]
            x2, y2 = quad[2 * ((i + 1) % 4)], quad[2 * ((i + 1) % 4) + 1]
            walls_px.append((x1, y1, x2, y2))

    doors_px = transform.point_sets_mm_to_px(doors)
    windows_px = transform.point_sets_mm_to_px(windows)

    return {
        'sample_id': str(sample_id),
        'split': split,
        'canvas': {'w': transform.width, 'h': transform.height},
        'walls_px': [list(map(float, w)) for w in walls_px],
        'doors_px': [[list(map(float, p)) for p in ring] for ring in doors_px],
        'windows_px': [[list(map(float, p)) for p in ring] for ring in windows_px],
        'dataset_dir': dataset_dir,
        'work_dir': work_dir,
        'transform': transform.to_dict(),
    }

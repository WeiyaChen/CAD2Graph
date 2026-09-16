"""VecFloorSeg adapter — CAD2Graph side of the input bridge.

Sub-modules
-----------
``geometry``        mm <-> pixel transform + JSON spec builder (runs in cadruler)
``build_dataset``   geometry -> pkl/PNG/triangle-table (runs in vecfloorseg)
``runner``          temp dataset + ``main.py --eval`` subprocess orchestration
``postprocess``     per-region predictions -> room polygons in mm
"""

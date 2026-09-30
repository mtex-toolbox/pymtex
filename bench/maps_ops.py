# the maps layer against MATLAB on one synthetic map: grain segmentation, grain shape, boundary
# smoothing, the intragranular misorientations and the denoising of the orientations; and the
# registration of the EBSDTrueEbsd page on trueEbsdWCCoSmall
#
#   .venv/bin/python bench/maps_ops.py export              the map
#   .venv/bin/python bench/maps_ops.py run [op ...]        this port
#   .venv/bin/python bench/maps_ops.py matlab [op ...]     bench/matlab/bench_maps_ops.m through the bridge
#   .venv/bin/python bench/maps_ops.py julia [op ...]      bench/maps_ops.jl, the Julia port on the same input
#   .venv/bin/python bench/maps_ops.py compare
#
# The map of bench/maps.py: N x N pixels of unit spacing, the grains the Voronoi cells of random
# seeds, each with its orientation about z and a gradient of 0.002 rad per pixel along x inside,
# written as rows (x, y, angle). The large map is the 2.5 million pixels of steel 1C_1 as
# bench/calcgrains.py exports it. The denoising runs on two measured maps, each port reading them
# itself, as the EBSDDenoising page prepares them: twins (magnesium, square grid) and ferrite (hex
# grid). The ports number grains and pixels their own way, so every result row starts with a key,
# the grain's centroid or the pixel's position, and compare matches by it. The registration runs the
# page's job, pixelSizeMatch once outside the timing; its shifts are rows (hop, stage, tile centre,
# shift, peak), stage -1 the fitError pass, in the tile order both ports share, compared row by row.
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "maps")
N, NG = 300, 200
ALGORITHM = [
  {"name": "grain segmentation",
   "entry": {"matlab": ["@EBSD.calcGrains"], "python": ["mtex.maps.calcgrains.calcGrains"],
             "julia": ["calcGrains!"]},
   "cases": {"calcgrains": "calcGrains of the 300 x 300 map at 5 degree: the area of every grain",
             "calcgrains_1C_1": "calcGrains of the 2.5 million pixel steel map 1C_1 at 10 degree, one run"},
   # MATLAB's boundary vertices sit up to 1e-6 off the pixel lattice; on 1C_1 the closing joins
   # differently (docs/research/benchmark-calcgrains.md): 22 grains more here, 1077 of 118930
   # without a match, the others to 1.3e-4 in their area
   "tolerance": {"calcgrains": 1e-5, "calcgrains_1C_1": 1e-3}},
  {"name": "grain shape",
   "entry": {"matlab": ["@grain2d.perimeter", "@grain2d.aspectRatio"],
             "python": ["mtex.maps.polygon.polygonSet.perimeter", "mtex.maps.shape.aspectRatio"],
             "julia": ["perimeter", "aspectRatio"]},
   "cases": {"perimeter": "the perimeter of every grain", "aspect": "aspectRatio of every grain"},
   # the aspect ratio is the one of the ellipse of the second area moments here (docs/plans/maps.md 9)
   "tolerance": {"perimeter": 1e-5, "aspect": 0.2}},
  {"name": "boundary smoothing",
   "entry": {"matlab": ["@grain2d.smoothBoundary"], "python": ["mtex.maps.smoothing.smoothBoundary"],
             "julia": ["smoothBoundary"]},
   "cases": {"smooth_boundary": "smoothBoundary(5): the perimeter of every grain after"},
   # the boundary filters take a smoothing length here (docs/plans/maps.md 9)
   "tolerance": 1e-3},
  {"name": "intragranular misorientation",
   "entry": {"matlab": ["@EBSD.KAM", "@EBSD.calcGROD"],
             "python": ["mtex.maps.intragranular.KAM", "mtex.maps.intragranular.calcGROD"],
             "julia": ["KAM", "calcGROD"]},
   "cases": {"kam": "KAM, first order, every pixel", "grod": "the angle of calcGROD, every pixel"},
   # MATLAB takes small angles through acos
   "tolerance": 1e-6},
  {"name": "orientation denoising",
   "entry": {"matlab": ["@EBSD.smooth", "@l1TVFilter.smooth"],
             "python": ["mtex.analysis.denoising.smooth", "mtex.analysis.variational.solveField"],
             "julia": ["smooth"]},
   "cases": {"smooth_mean_twins": "smooth(meanFilter) of the twins map, every indexed pixel",
             "smooth_spline_twins": "smooth(splineFilter) of the twins map",
             "smooth_l1tv_twins": "smooth(l1TVFilter) of the twins map",
             "smooth_mean_ferrite": "smooth(meanFilter) of the ferrite map, hexagonal grid",
             "smooth_spline_ferrite": "smooth(splineFilter) of the ferrite map",
             "smooth_l1tv_ferrite": "smooth(l1TVFilter) of the ferrite map",
             "smooth_hq_twins": "smooth(halfQuadraticFilter) of the twins map",
             "smooth_hq_ferrite": "smooth(halfQuadraticFilter) of the ferrite map",
             "smooth_fill_twins": "smooth(splineFilter, 'fill') of the twins map, the holes filled",
             "smooth_fill_ferrite": "smooth(splineFilter, 'fill') of the ferrite map, the holes filled"},
   # the misorientation angle in radians
   "tolerance": 1e-6},
  {"name": "map registration",
   "entry": {"matlab": ["@trueEbsd.calcDistortion", "@trueEbsd.undistort"],
             "python": ["mtex.maps.trueebsd.trueEbsd.calcDistortion", "mtex.maps.trueebsd.trueEbsd.undistort"],
             "julia": []},
   "cases": {"trueebsd_distortion": "calcDistortion('fitErr') of the EBSDTrueEbsd page: every tile's shift and peak",
             "trueebsd_undistort": "undistort: the phase and rotation of every pixel of the undistorted map"},
   # the shifts agree to the digit and the peaks, sums of up to 7e6, to 1e-14 (docs/plans/maps.md M11);
   # the undistorted map is picked by nearest pixel, the same pixels in both
   "tolerance": 1e-12},
]
DENOISE = [f"smooth_{f}_{m}" for m in ("twins", "ferrite") for f in ("mean", "spline", "l1tv", "hq", "fill")]
COLS = {"calcgrains": 3, "calcgrains_1C_1": 3, "perimeter": 3, "aspect": 3, "smooth_boundary": 3, "kam": 3,
        "grod": 3, "trueebsd_distortion": 7, "trueebsd_undistort": 7} | {op: 6 for op in DENOISE}
SYM = {"twins": "6/mmm", "ferrite": "m-3m"}
# the ops timed by a single run
ONCE = ("calcgrains_1C_1", "trueebsd_distortion", "smooth_l1tv_twins", "smooth_l1tv_ferrite", "smooth_hq_twins", "smooth_hq_ferrite")
LARGE = os.path.join(HERE, "out", "calcgrains", "1C_1.bin")


def export():
  from scipy.spatial import cKDTree
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
  P = np.column_stack([i.reshape(-1), j.reshape(-1)]).astype(float)
  grain = cKDTree(rng.uniform(0, N, (NG, 2))).query(P)[1]
  ang = rng.uniform(0, 2 * np.pi, NG)[grain] + 0.002 * P[:, 0]
  np.column_stack([P, ang]).astype("<f8").tofile(os.path.join(OUT, "map.bin"))
  if not os.path.exists(LARGE):
    subprocess.run([sys.executable, os.path.join(HERE, "calcgrains.py"), "export", "1C_1"], check=True)


def run(ops):
  from mtex import (EBSD, calcGrains, crystalFrame, degree, halfQuadraticFilter, l1TVFilter, meanFilter, mtexdata,
                    phaseList, rotation, splineFilter, zvector)
  sys.path.insert(0, HERE)
  import calcgrains
  A = np.fromfile(os.path.join(OUT, "map.bin"), "<f8").reshape(-1, 3)
  pos = np.column_stack([A[:, :2], np.zeros(len(A))])
  e = EBSD(pos, rotation.byAxisAngle(zvector, A[:, 2]), np.ones(len(A), dtype=np.int8), phaseList(crystalFrame("m-3m", mineral="test")))
  xy = np.asarray(e.pos.data, float).reshape(-1, 3)[:, :2]
  s = {}

  def grains():
    s["g"] = calcGrains(e, angle=5 * degree)
    s["c"] = np.asarray(s["g"].centroid.data, float).reshape(-1, 3)[:, :2]
    return s["g"]

  byGrain = lambda v: np.column_stack([s["c"], np.asarray(v, float).reshape(-1)])
  byPixel = lambda v: np.column_stack([xy, np.asarray(v, float).reshape(-1, np.asarray(v).size // len(xy))])
  # a smoothed map comes back on its grid, in the grid's order
  quat = lambda m: np.column_stack([np.asarray(m.pos.data, float).reshape(-1, 3)[:, :2],
                                    np.asarray(m.rotations.data, float).reshape(-1, 4)])
  # the segmentation alone is timed, as in the Julia half; the grains' measures are the result
  large = lambda: calcGrains(s["large"], angle=10 * degree)
  byLarge = lambda g: np.column_stack([np.asarray(g.centroid.data, float).reshape(-1, 3)[:, :2], np.asarray(g.area, float)])

  # the measured maps as the EBSDDenoising page prepares them
  def measured(name):
    m = mtexdata(name)
    calcGrains(m, angle=10 * degree, minPixel=5)
    return m["indexed"]

  s["large"] = calcgrains.loadMap("1C_1") if "calcgrains_1C_1" in ops else None
  if any(op.startswith("trueebsd") for op in ops):
    s["page"] = trueEbsdPage()
  maps = {m: measured(m) for m in SYM if any(op.endswith(m) for op in ops if op in DENOISE)}
  filters = {"mean": meanFilter, "spline": splineFilter, "l1tv": l1TVFilter, "hq": halfQuadraticFilter, "fill": splineFilter}
  # the input of the denoising, to measure how far each port moves it
  for m, e0 in maps.items():
    np.column_stack([np.asarray(e0.pos.data, float).reshape(-1, 3)[:, :2],
                     np.asarray(e0.rotations.data, float).reshape(-1, 4)]).astype("<f8").tofile(os.path.join(OUT, f"input_{m}.bin"))
  denoise = lambda op: lambda: maps[op.split("_")[2]].smooth(filters[op.split("_")[1]](), fill=op.split("_")[1] == "fill")["indexed"]
  table = {
    "calcgrains_1C_1": (large, byLarge),
    "calcgrains": (grains, lambda g: byGrain(g.area)),
    "perimeter": (lambda: s["g"].perimeter, byGrain),
    "aspect": (lambda: s["g"].aspectRatio(), byGrain),
    "smooth_boundary": (lambda: s["g"].smoothBoundary(5), lambda g: byGrain(g.perimeter)),
    "kam": (lambda: e.KAM(), byPixel),
    "grod": (lambda: e.calcGROD(s["g"]).angle(), byPixel),
  } | {op: (denoise(op), quat) for op in DENOISE} | {
    "trueebsd_distortion": (lambda: distortion(s), shiftRows),
    "trueebsd_undistort": (lambda: s["job"].undistort() or s["job"].undistortedList[0].ebsd, undistortedRows)}
  crosscheck.run(OUT, ALGORITHM, table, ops, reps={op: 1 for op in ONCE}, always=("calcgrains", "trueebsd_distortion"))


# the images, hops and matched grid of the EBSDTrueEbsd page
def trueEbsdPage():
  from mtex import (imboxfilt, mapImage, mapImageList, mtexdata, rescale, spatialTransformDrift, spatialTransformId,
                    spatialTransformShift, spatialTransformTilt, trueEbsd)
  ebsd = mtexdata("trueEbsdWCCoSmall")
  img = ebsd.meta["trueEbsdImgs"]
  imgList = mapImageList([mapImage(ebsd.bc, ebsd, name="bcImg")] +
                         [mapImage(getattr(img, n), dxy=img.pixSzImg, name=n) for n in ("fsdB3", "fsdT3", "fsdT1", "fsdT10")])
  imgList[1:] = rescale(imboxfilt(imgList[1:], 3))
  hops = lambda: [spatialTransformShift() + spatialTransformDrift(), spatialTransformId(), spatialTransformShift(), spatialTransformTilt()]
  job = trueEbsd(imgList, hops())
  job.pixelSizeMatch()
  return imgList, hops, job.resizedList


# a fresh job on the matched grid, since calcDistortion alters the one it runs on
def distortion(s):
  from mtex import trueEbsd
  imgList, hops, resized = s["page"]
  job = trueEbsd(imgList, hops())
  job.resizedList = resized
  job.setOptions(numROI=16)
  job.calcDistortion("fitErr", verbose=False)
  s["job"] = job
  return job


def shiftRows(job):
  def rows(n, m, ps):
    xyz = lambda v: np.asarray(v.data, float).reshape(-1, 3)
    return np.column_stack([np.full(ps.peak.size, n), np.full(ps.peak.size, m), xyz(ps.pos)[:, :2], xyz(ps.u)[:, :2], ps.peak])
  R = [rows(n, m, ps) for n, hop in enumerate(job.shifts) for m, ps in enumerate(hop) if not ps.isempty]
  R += [rows(n, -1, ps) for n, ps in enumerate(job.fitError) if not ps.isempty]
  return np.vstack(R)


def undistortedRows(e):
  return np.column_stack([np.asarray(e.pos.data, float).reshape(-1, 3)[:, :2], e.phaseId.reshape(-1),
                          np.asarray(e.rotations.data, float).reshape(-1, 4)])


# MATLAB one op per call, each limited to 20 times Python's time per run plus loading, since a
# call that overruns keeps MATLAB busy after its client is gone: the first timeout ends the batch
def matlab(ops):
  for op in ops:
    f = os.path.join(OUT, f"python_{op}.txt")
    runs = 2 if op in ONCE else 7
    limit = min(290, 20 * float(open(f).read()) * runs + 40) if os.path.exists(f) else 290
    print(f"{op}: limit {limit:.0f} s", flush=True)
    crosscheck.matlab("bench_maps_ops", repr(OUT), crosscheck.cellOf([op]), "1e7", repr(LARGE), timeout=limit)


# rows matched by their keys (the first two columns, a centroid or a position) to the nearest within
# 1e-3; orientations compared by their misorientation angle in m-3m, the other values relative to the largest
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  # the tiles in one order in both ports, each column relative to its largest value
  if op == "trueebsd_distortion":
    if a.shape != b.shape or (np.isnan(a) != np.isnan(b)).any() or (a[:, :2] != b[:, :2]).any():
      return float("nan")
    return float(np.nanmax(np.nanmax(np.abs(a - b), axis=0)[2:] / np.nanmax(np.abs(b), axis=0)[2:]))
  d, j = cKDTree(b[:, :2]).query(a[:, :2])
  # the grains of a large map by the ones both ports find, the others counted
  if op == "calcgrains_1C_1" and len(a) != len(b):
    ok = d <= 1e-3
    print(f"  {op}: {len(a)} and {len(b)} grains, {int((~ok).sum())} of this port's without a match")
    a, j, d = a[ok], j[ok], d[ok]
  if (len(a) != len(b) and op != "calcgrains_1C_1") or d.max() > 1e-3 or len(set(j)) != len(j):
    return float("nan")
  va, vb = a[:, 2:], b[j, 2:]
  if va.shape[1] == 4:
    from mtex import angle, crystalFrame, orientation, rotation
    cs = crystalFrame(SYM[op.split("_")[2]])
    return float(np.max(np.asarray(angle(orientation(rotation(va), cs), orientation(rotation(vb), cs)))))
  both = ~(np.isnan(va) | np.isnan(vb))
  if (np.isnan(va) != np.isnan(vb)).any():
    print(f"  {op}: NaN in {int(np.isnan(va).sum())} Python and {int(np.isnan(vb).sum())} MATLAB rows")
  return float(np.max(np.abs(va - vb)[both]) / np.max(np.abs(vb[both])))


# the table; for the denoising every pair of ports by the median and largest misorientation and
# the share of pixels apart by more than 0.1 degree, and how far each port moved the measured map
def compare():
  crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)
  from scipy.spatial import cKDTree
  from mtex import angle, crystalFrame, orientation, rotation
  load = lambda f: np.fromfile(f, "<f8").reshape(-1, 6) if os.path.exists(f) else None

  # the pixels both have, at one position and finite in both
  def apart(a, b, CS):
    d, j = cKDTree(b[:, :2]).query(a[:, :2])
    ok = (d < 1e-6) & np.isfinite(a[:, 2]) & np.isfinite(b[j, 2])
    return np.degrees(np.asarray(angle(orientation(rotation(a[ok, 2:]), CS), orientation(rotation(b[j[ok], 2:]), CS))))

  print(f"\n{'denoising':22s} {'moved the measured map, median degree':>40s}   ports apart: median / max degree, share > 0.1, pixels compared")
  for op in DENOISE:
    m = op.split("_")[2]
    cs = crystalFrame(SYM[m])
    r = {p: load(os.path.join(OUT, f"{p}_{op}.bin")) for p in ("python", "matlab", "julia")}
    inp = load(os.path.join(OUT, f"input_{m}.bin"))
    moved = "  ".join(f"{p} {np.median(apart(r[p], inp, cs)):.3f}" for p in r if r[p] is not None and inp is not None)
    print(f"  {op:20s} {moved:>40s}")
    pairs = []
    for p, q in (("python", "matlab"), ("python", "julia"), ("julia", "matlab")):
      if r[p] is not None and r[q] is not None:
        w = apart(r[p], r[q], cs)
        print(f"  {'':20s} {p:>6s} ~ {q:6s} {np.median(w):.1e} / {w.max():5.2f} / {np.mean(w > 0.1):4.0%}, {len(w)} pixels")


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: matlab(names),
   "julia": lambda: crosscheck.julia("maps_ops.jl", ALGORITHM, names),
   "compare": compare}[cmd]()

# the metrics of metrics.py on cases with analytic answers: the pixel staircase of a straight line,
# of a disc, of a square on the pixel edges and of one off them, and two partitions; every expected
# value is computed here without planar.py, by integrals along the columns, quadrature or by hand
#
#   .venv/bin/python bench/grains/checks.py
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

import metrics as mt
import planar as pl


def staircase(lab):
  """the pixel edge boundary of a labelled square grid of unit spacing, pixel lab[j, i] at (i, j)"""
  L = np.pad(lab, 1)
  # between (i - 1, j) west and (i, j) east, walked upwards; between (i, j - 1) south and (i, j) north, walked westwards
  w, e = L[1:-1, :-1], L[1:-1, 1:]
  j, i = np.nonzero(w != e)
  V = np.column_stack([i - 0.5, j - 0.5, i - 0.5, j + 0.5, w[j, i], e[j, i]])
  s, n = L[:-1, 1:-1], L[1:, 1:-1]
  j, i = np.nonzero(s != n)
  H = np.column_stack([i + 0.5, j - 0.5, i - 0.5, j - 0.5, s[j, i], n[j, i]])
  return np.vstack([V, H]).astype(float)


def case(T, nx, ny):
  i, j = np.meshgrid(np.arange(nx), np.arange(ny))
  pos = np.column_stack([i.ravel(), j.ravel()]).astype(float)
  lab = pl.locate(T, pos)
  inst = dict(S=T, pos=pos, label=lab, junctions=pl.junctions(T))
  return inst, staircase(lab.reshape(ny, nx))


def intAbs(f0, f1, w):
  """the integral of |f| over an interval of length w where f is linear from f0 to f1"""
  f0, f1 = np.asarray(f0, float), np.asarray(f1, float)
  same = f0 * f1 >= 0
  return w * np.where(same, (np.abs(f0) + np.abs(f1)) / 2, (f0**2 + f1**2) / (2 * (np.abs(f0) + np.abs(f1)) + 1e-300))


def line(slope, c0, nx=64, ny=64):
  """y = slope x + c0, the grain below 1 and above 2"""
  u = np.array([1, slope]) / np.hypot(1, slope)
  p, q = np.array([0, c0]) - 1000 * u, np.array([0, c0]) + 1000 * u
  T = pl.clipToBox(np.array([[*p, *q, 2, 1]]), (-100, nx + 100, -100, ny + 100),
                   lambda xy: np.where(xy[:, 1] > slope * xy[:, 0] + c0, 2, 1))
  inst, S = case(T, nx, ny)
  # the staircase height in every column, the misassigned area and the staircase's distances
  x = np.arange(nx)
  top = np.ceil(slope * x + c0) - 1
  h = top + 0.5
  f = lambda xx, yy: yy - slope * xx - c0
  M = intAbs(f(x - 0.5, h), f(x + 0.5, h), 1).sum()
  ell = nx * np.hypot(1, slope)
  dh = np.abs(np.diff(h))
  vert = intAbs(f(x[:-1] + 0.5, h[:-1]), f(x[:-1] + 0.5, h[1:]), dh)
  stair = nx + dh.sum()
  verts = np.concatenate([np.column_stack([x - 0.5, h]), np.column_stack([x + 0.5, h])])
  aTb = slope * ((nx - 0.5)**2 - 0.25) / 2 + (c0 + 0.5) * nx
  aRb = (top + 1).sum()
  th = np.degrees(np.arctan(abs(slope)))
  # the distance along the staircase is |f| and changes sign inside Simpson pieces of 1/32 step
  dm = (M + vert.sum()) / np.hypot(1, slope) / stair
  want = dict(disp=M / ell, d_rt_mean=(dm, 1e-4 * dm),
              d_rt_max=np.abs(f(*verts.T)).max() / np.hypot(1, slope), lengthRatio=stair / ell,
              dirErr=(nx * th + dh.sum() * (90 - th)) / stair,
              areaBias=((aRb - aTb) / aTb + (aTb - aRb) / (nx * ny - aTb)) / 2, pixelWrong=0.0)
  return inst, S, want


def disc(r, c, n=None, tol=1e-6):
  """a disc of radius r about c in one matrix grain"""
  nx = ny = int(np.ceil(2 * r + 12))
  m = int(np.ceil(2 * np.pi * np.sqrt(r / (8 * tol))))
  t = np.linspace(0, 2 * np.pi, m + 1)
  P = c + r * np.column_stack([np.cos(t), np.sin(t)])
  T = pl.clipToBox(np.column_stack([P[:-1], P[1:], np.full(m, 2), np.ones(m)]), (-5, nx + 5, -5, ny + 5), lambda xy: np.ones(len(xy)))
  inst, S = case(T, nx, ny)
  # the area of the disc in every pixel the circle crosses, by quadrature of its chords
  def inPixel(i, j):
    g = lambda u: max(0.0, min(j + 0.5, c[1] + np.sqrt(max(r * r - (u - c[0])**2, 0))) - max(j - 0.5, c[1] - np.sqrt(max(r * r - (u - c[0])**2, 0))))
    lo, hi = max(i - 0.5, c[0] - r), min(i + 0.5, c[0] + r)
    return quad(g, lo, hi, epsabs=1e-13, limit=200)[0] if hi > lo else 0.0
  inside = inst["label"] == 2
  M = 0.0
  for (x, y), isIn in zip(inst["pos"], inside):
    d = np.hypot(x - c[0], y - c[1])
    if abs(d - r) < 0.75:
      a = inPixel(x, y)
      M += 1 - a if isIn else a
  # the distance of the staircase to the circle, along it at 2000 points per pixel edge
  B = S[(S[:, 4] > 0) & (S[:, 5] > 0)]
  t = (np.arange(2000) + 0.5) / 2000
  P = B[:, None, :2] + t[None, :, None] * (B[:, None, 2:4] - B[:, None, :2])
  d = np.abs(np.hypot(P[..., 0] - c[0], P[..., 1] - c[1]) - r)
  nIn = inside.sum()
  want = dict(disp=M / (2 * np.pi * r), d_rt_mean=d.mean(), areaBias=((nIn - np.pi * r * r) / (np.pi * r * r) +
              (np.pi * r * r - nIn) / (nx * ny - np.pi * r * r)) / 2, pixelWrong=0.0)
  return inst, S, want


def square(shift, nx=48, ny=56):
  """the square [10.5, 30.5] x [12.5, 40.5] shifted by shift, in one matrix grain"""
  x0, y0 = 10.5 + shift[0], 12.5 + shift[1]
  P = np.array([[x0, y0], [x0 + 20, y0], [x0 + 20, y0 + 28], [x0, y0 + 28], [x0, y0]])
  T = pl.clipToBox(np.column_stack([P[:-1], P[1:], np.full(4, 2), np.ones(4)]), (-5, nx + 5, -5, ny + 5), lambda xy: np.ones(len(xy)))
  inst, S = case(T, nx, ny)
  # by hand: the pixel centres of the shifted square fill [10.5, 30.5] x [12.5, 40.5] for shifts in
  # [0, 0.5); the nearest true edge is the perpendicular one along 2 (sx + sy) + |sx - sy| of the
  # reconstructed edges near two corners, and Simpson's rule on 1/32 pieces is within a third of a
  # piece at each of the five points where it switches
  sx, sy = shift
  sym = 2 * (560 - (20 - sx) * (28 - sy))
  turn = 2 * (sx + sy) + abs(sx - sy) if min(sx, sy) > 0 else 0.0
  want = dict(disp=sym / 96, areaBias=0.0, lengthRatio=1.0, dirErr=(90 * turn / 96, 5 * 90 / 3 / 32 / 96), dirW1=0.0, pixelWrong=0.0,
              d_rt_max=np.hypot(*shift), d_tr_max=np.hypot(*shift))
  return inst, S, want


def partitionCases():
  """identical partitions, and two equal grains merged into one"""
  i, j = np.meshgrid(np.arange(20), np.arange(10))
  t = np.where(i.ravel() < 10, 1, 2)
  inst = dict(label=t, phase=np.ones(200), wild=np.zeros(200, bool), pos=np.column_stack([i.ravel(), j.ravel()]).astype(float),
              grainQ=np.array([[1, 0, 0, 0], [1, 0, 0, 0], [np.cos(0.2), np.sin(0.2), 0, 0]]), grainPhase=np.ones(4, int), phases=["Iron"])
  yield "identical", inst, t, dict(ari=1.0, vi=0.0, f1_50=1.0, f1_75=1.0, splits=0, merges=0, sizeW1=0.0)
  yield "merged", inst, np.ones(200, np.int64), dict(ari=0.0, vi=1.0, f1_50=0.0, merges=1, splits=0)
  yield "split", inst, np.where(i.ravel() < 5, 3, t), dict(vi=0.5, f1_50=0.4, f1_75=0.4, splits=1, merges=0)


def main():
  rows = []
  for name, (inst, S, want) in [("line 0.137", line(0.137, 20.3141)), ("line 0.577", line(0.5773, 15.2718)), ("line 1", line(1.0, 0.1592, 40, 64)),
                                ("disc 10", disc(10.0, np.array([16.3172, 15.7183]))), ("disc 3", disc(3.0, np.array([8.4142, 9.1732]))),
                                ("square on edges", square((0.0, 0.0))), ("square off edges", square((0.3, 0.2)))]:
    got = mt.geometry(inst, S)
    rows += [(name, k, got[k], v) for k, v in want.items()]
  for name, inst, gid, want in partitionCases():
    got = mt.partition(inst, gid)
    rows += [(name, k, got[k], v) for k, v in want.items()]
  bad = 0
  print(f"{'case':18s} {'metric':12s} {'metric.py':>14s} {'analytic':>14s}")
  for name, k, g, w in rows:
    # the disc is a polygon within 1e-6 of the circle
    w, tol = w if isinstance(w, tuple) else (w, (2e-4 if name.startswith("disc") else 1e-9) * max(1, abs(w)))
    ok = abs(g - w) <= tol
    bad += not ok
    print(f"{name:18s} {k:12s} {g:14.9f} {w:14.9f} {'' if ok else 'FAIL'}")
  print("all passed" if not bad else f"{bad} failed")
  return bad


if __name__ == "__main__":
  raise SystemExit(main())

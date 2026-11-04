# the true grain structures: each generator returns a partition (planar.py) of a box in the plane,
# its regions numbered from 1, drawn from a random generator
#
#   G1 voronoi     Poisson Voronoi, mean cell area pi D^2 / 4
#   G2 laguerre    Laguerre cells of Neper, equivalent diameters log-normal around D
#   G3 inclusions  discs and ellipses in one matrix grain, radii log-uniform between rmin and rmax
#   G5 stretched   Poisson Voronoi stretched by aspect along x at constant cell area
#   G5L lamellae   bands along x, their thickness log-uniform between tmin and tmax
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
from scipy.spatial import Voronoi, cKDTree

import planar as pl


def generate(rng, box, generator, **p):
  S = {"G1": voronoi, "G2": laguerre, "G3": inclusions, "G5": stretched, "G5L": lamellae}[generator](rng, box, **p)
  # the regions present numbered 1, 2, ... in the order of their labels
  lab, inv = np.unique(S[:, 4:6], return_inverse=True)
  inv = inv.reshape(-1, 2) + (lab[0] != 0)
  S[:, 4:6] = np.where(S[:, 4:6] == 0, 0, inv)
  return S


def voronoi(rng, box, D):
  x0, x1, y0, y1 = box
  m = 3 * D
  n = rng.poisson((x1 - x0 + 2 * m) * (y1 - y0 + 2 * m) / (np.pi * D**2 / 4))
  sites = rng.uniform([x0 - m, y0 - m], [x1 + m, y1 + m], (n, 2))
  vor = Voronoi(sites)
  rv, rp = np.array(vor.ridge_vertices), vor.ridge_points
  fin = (rv >= 0).all(axis=1)
  P, Q, a, b = vor.vertices[rv[fin, 0]], vor.vertices[rv[fin, 1]], rp[fin, 0], rp[fin, 1]
  # the site a on the left of P -> Q, or else b
  d, e = Q - P, sites[a] - P
  left = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0] > 0
  S = np.column_stack([P, Q, np.where(left, a, b) + 1, np.where(left, b, a) + 1])
  tree = cKDTree(sites)
  return pl.clipToBox(S, box, lambda xy: tree.query(xy)[1] + 1)


def laguerre(rng, box, D, sigma=0.5, itermax=3000):
  x0, x1, y0, y1 = box
  w, h = x1 - x0, y1 - y0
  n = max(2, round(w * h / (np.pi * D**2 / 4)))
  with tempfile.TemporaryDirectory() as tmp:
    # one thread, so that the optimisation of the cell sizes is the same on every run; Neper cuts
    # an output path at its first dot, so it writes into the working directory
    subprocess.run(["neper", "-T", "-n", str(n), "-dim", "2", "-domain", f"square({w},{h})", "-morpho", f"diameq:lognormal(1,{sigma})",
                    "-morphooptistop", f"itermax={itermax}", "-id", str(int(rng.integers(1, 2**31 - 1))), "-o", "t"],
                   check=True, capture_output=True, cwd=tmp, env=os.environ | {"OMP_NUM_THREADS": "1"})
    V, faces = readTess(os.path.join(tmp, "t.tess"))
  V = V + [x0, y0]
  # every face anticlockwise, an edge met in both senses has a face on either side
  half = {}
  for f, loop in enumerate(faces, 1):
    x, y = V[loop, 0], V[loop, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0:
      loop = loop[::-1]
    for a, b in zip(loop, np.roll(loop, -1)):
      half[a, b] = f
  rows = [[*V[a], *V[b], f, half.get((b, a), 0)] for (a, b), f in half.items() if half.get((b, a), 0) < f or (b, a) not in half]
  return np.array(rows, float)


def readTess(fname):
  """the vertices and the vertex loops of the faces of a two dimensional Neper tessellation"""
  lines = open(fname).read().split("\n")
  at = {l.strip(): k for k, l in enumerate(lines) if l.strip().startswith("**")}
  k = at["**vertex"]
  V = np.array([[float(t) for t in lines[k + 2 + i].split()[1:3]] for i in range(int(lines[k + 1]))])
  k, faces = at["**face"], []
  for i in range(int(lines[k + 1])):
    t = lines[k + 2 + 4 * i].split()
    faces.append(np.array([int(v) - 1 for v in t[2:2 + int(t[1])]]))
  return V, faces


def inclusions(rng, box, rmin, rmax, fraction=0.3, ellipses=0.5, aspect=3.0, gap=1.0, tol=1e-4):
  x0, x1, y0, y1 = box
  shapes, area = [], 0.0
  for _ in range(20000):
    if area >= fraction * (x1 - x0) * (y1 - y0):
      break
    # an ellipse of the area of the disc of radius r
    r = np.exp(rng.uniform(np.log(rmin), np.log(rmax)))
    s = np.sqrt(rng.uniform(1, aspect)) if rng.random() < ellipses else 1.0
    a, b, phi = r * s, r / s, rng.uniform(0, np.pi)
    c = rng.uniform([x0 + a + gap, y0 + a + gap], [x1 - a - gap, y1 - a - gap])
    if all(np.hypot(*(c - c2)) > a + a2 + gap for c2, a2, *_ in shapes):
      shapes.append((c, a, b, phi))
      area += np.pi * a * b
  rows = []
  for k, (c, a, b, phi) in enumerate(shapes, 2):
    # vertices on the ellipse, the chords within tol of it where it is most curved
    n = int(np.ceil(2 * np.pi * a * np.sqrt(a / (8 * b * b * tol))))
    t = np.linspace(0, 2 * np.pi, n + 1)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    P = c + np.column_stack([a * np.cos(t), b * np.sin(t)]) @ R.T
    rows.append(np.column_stack([P[:-1], P[1:], np.full(n, k), np.ones(n)]))
  S = np.vstack(rows) if rows else np.zeros((0, 6))
  return pl.clipToBox(S, box, lambda xy: np.ones(len(xy), np.int64))


def stretched(rng, box, D, aspect=4.0):
  x0, x1, y0, y1 = box
  s = np.sqrt(aspect)
  S = voronoi(rng, (x0 / s, x1 / s, y0 * s, y1 * s), D)
  S[:, [0, 2]] *= s
  S[:, [1, 3]] /= s
  return S


def lamellae(rng, box, tmin, tmax):
  x0, x1, y0, y1 = box
  y = [y0]
  while y[-1] < y1:
    y.append(y[-1] + np.exp(rng.uniform(np.log(tmin), np.log(tmax))))
  y = np.array(y[1:-1])
  # walked eastwards, the band above on the left
  k = np.arange(len(y)) + 1
  S = np.column_stack([np.full(len(y), x0), y, np.full(len(y), x1), y, k + 1, k])
  return pl.clipToBox(S, box, lambda xy: np.searchsorted(y, xy[:, 1]) + 1)


def transform(S, phi, centre, offset):
  """the partition turned by phi about the centre and shifted by offset"""
  R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
  out = S.copy()
  for c in (slice(0, 2), slice(2, 4)):
    out[:, c] = (S[:, c] - centre) @ R.T + centre + offset
  return out


def neighbours(S):
  """the pairs of regions sharing a boundary, each once"""
  p = np.sort(S[:, 4:6].astype(np.int64), axis=1)
  return np.unique(p[(p[:, 0] > 0) & (p[:, 0] != p[:, 1])], axis=0)

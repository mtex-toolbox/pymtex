# the halfwidth selectors on S2 against the oracle (docs/plans/halfwidth.md, section 7): exact
# samples of zonal mixtures whose degree energies are known in closed form, all methods on the
# same samples, the exact risk at the selected halfwidth against the oracle's, the realised ISE,
# the share of replicates below the oracle, the fluctuation of the estimate against the C3 bound
#
#   .venv/bin/python bench/halfwidth_s2.py [--reps R] [--n 1e2,1e3,...] [--models A,B,...] [--fluct]
#   .venv/bin/python bench/halfwidth_s2.py timing
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))

from mtex import degree, vector3d
from mtex.functions.halfwidth import (candidateHalfwidths, degreeEnergies, familyMultipliers, lowerBoundHalfwidth,
                                      noiseFloorHalfwidth, ucvHalfwidth, _legendre)
from mtex.functions.kernelfunctions import _dlvpKappa, _dlvpRule
from mtex.functions.s2fun import evenize
from mtex.functions.transforms import S2FourierPlan

ALGORITHM = {
  "name": "halfwidth selection on S2",
  "entry": {"python": ["mtex.functions.halfwidth.selectHalfwidth", "mtex.functions.halfwidth.degreeEnergies"]},
  "cases": {
    "A": "0.4 uniform + 0.4 dlVP 10 deg + 0.2 dlVP 5 deg, 37 deg apart",
    "B": "0.7 uniform + three dlVP 20 deg bumps on the axes",
    "peak5": "one dlVP bump of 5 deg", "peak10": "one of 10 deg", "peak20": "one of 20 deg",
    "sharp": "0.95 uniform + 0.05 dlVP 3 deg",
    "girdle": "a great circle smoothed by a Gaussian of 8 deg in the polar angle",
    "Aaxial": "model A made axial",
  },
}

LT = 256          # the bandwidth of the realised ISE and of UCV
H = np.array(candidateHalfwidths("S2", 1 * degree, 60 * degree, 60))


# ------------------------------------------------------------------ the test densities
# a model: the uniform weight and zonal components (weight, centre, alpha_l, sampler of t)

def _dlvp(hw):
  kappa = _dlvpKappa(hw * degree)
  a = _dlvpRule(kappa, LT) / (2 * np.arange(LT + 1) + 1)
  return a, lambda u: 2 * u ** (1 / (kappa + 1)) - 1


# a zonal profile p(t) on [-1, 1] with mean one over the sphere: its Legendre means and the
# inverse of its distribution function in t, tabulated
def _zonal(p):
  x, wq = np.polynomial.legendre.leggauss(2000)
  px = p(x)
  px /= np.sum(wq * px) / 2
  P = _legendre(x, LT)
  a = (P * (wq * px / 2)).sum(axis=1)
  t = np.linspace(-1, 1, 20001)
  cdf = np.concatenate([[0], np.cumsum((p(t[1:]) + p(t[:-1])) / 2)])
  cdf /= cdf[-1]
  return a, lambda u: np.interp(u, cdf, t)


def _unit(v):
  v = np.asarray(v, dtype=float)
  return v / np.linalg.norm(v)


def model(name):
  z, x, y = np.eye(3)[2], np.eye(3)[0], np.eye(3)[1]
  c37 = np.array([math.sin(37 * degree), 0, math.cos(37 * degree)])
  if name in ("A", "Aaxial"):
    comps = [(0.4, z, *_dlvp(10)), (0.2, c37, *_dlvp(5))]
    u = 0.4
  elif name == "B":
    comps = [(0.1, v, *_dlvp(20)) for v in (x, y, z)]
    u = 0.7
  elif name.startswith("peak"):
    comps, u = [(1.0, z, *_dlvp(float(name[4:])))], 0.0
  elif name == "sharp":
    comps, u = [(0.05, _unit([1, 2, 3]), *_dlvp(3))], 0.95
  elif name == "girdle":
    s = 8 * degree
    comps, u = [(1.0, z, *_zonal(lambda t: np.exp(-(np.arcsin(np.clip(t, -1, 1))) ** 2 / (2 * s * s))))], 0.0
  else:
    raise ValueError(name)
  if name.endswith("axial"):
    comps = [(w / 2, s * c, a, smp) for (w, c, a, smp) in comps for s in (1, -1)]
  return u, comps


# the exact degree energies A_l = (2l+1) sum_jk w_j w_k a_lj a_lk P_l(c_j . c_k)
def trueEnergies(m):
  u, comps = m
  W = np.array([c[0] for c in comps])
  C = np.stack([c[1] for c in comps])
  Al = np.stack([c[2] for c in comps])
  P = _legendre(C @ C.T, LT)
  A = (2 * np.arange(LT + 1) + 1) * np.einsum("j,k,lj,lk,ljk->l", W, W, Al.T, Al.T, P)
  A[0] = 1
  return A


# a zonal density about c: t from the sampler, the azimuth uniform
def _around(c, t, rng):
  n = len(t)
  phi = rng.uniform(0, 2 * np.pi, n)
  s = np.sqrt(np.maximum(1 - t * t, 0))
  e1 = _unit(np.cross(c, [1, 0, 0] if abs(c[0]) < 0.9 else [0, 1, 0]))
  e2 = np.cross(c, e1)
  return t[:, None] * c + (s * np.cos(phi))[:, None] * e1 + (s * np.sin(phi))[:, None] * e2


def sample(m, n, rng):
  u, comps = m
  k = rng.choice(len(comps) + 1, n, p=[u] + [c[0] for c in comps])
  out = rng.standard_normal((n, 3))
  out /= np.linalg.norm(out, axis=1, keepdims=True)
  for j, (_, c, _, smp) in enumerate(comps):
    sel = k == j + 1
    out[sel] = _around(c, smp(rng.uniform(size=int(sel.sum()))), rng)
  return out


# ------------------------------------------------------------------ the error measures


# the dimension of degree l of the estimate: 2l+1, and zero on the odd degrees of an axial one
def dims(L, axial):
  l = np.arange(L + 1)
  return (2 * l + 1.0) * ((l % 2 == 0) if axial else 1)


def exactRisk(A, n, h, axial=False):
  b = familyMultipliers("S2", h)
  d = dims(len(b) - 1, axial)
  L = min(len(b), len(A)) - 1
  r = (1 - b[:L + 1]) ** 2 * A[:L + 1] + b[:L + 1] ** 2 * (d[:L + 1] - A[:L + 1]) / n
  tail = np.sum(b[L + 1:] ** 2 * d[L + 1:]) / n + np.sum(A[L + 1:])
  return float(np.sum(r[1:]) + tail)


def oracle(A, n, axial=False):
  r = np.array([exactRisk(A, n, h, axial) for h in H])
  k = int(np.argmin(r))
  return H[k], r[k]


# the degree-wise oracle filter, the lower bound of every kernel family
def filterOracle(A, n, axial=False):
  V = np.maximum(dims(len(A) - 1, axial) - A, 0) / n
  return float(np.sum((A * V / np.maximum(A + V, 1e-300))[1:]))


# the realised ISE of every candidate: sum_l b^2 ||mu_l||^2 - 2 b <mu_l, f_l> + ||f_l||^2, with
# <mu_l, f_l> = (2l+1) sum_k w_k a_lk mean_i P_l(X_i . c_k)
def realisedISE(x, m, A, energy):
  u, comps = m
  cross = np.zeros(LT + 1)
  for (w, c, a, _) in comps:
    t = x @ c
    p0, p1 = np.ones_like(t), t
    s = np.empty(LT + 1)
    s[0], s[1] = 1, t.mean()
    for l in range(1, LT):
      p0, p1 = p1, ((2 * l + 1) * t * p1 - l * p0) / (l + 1)
      s[l + 1] = p1.mean()
    cross += w * a * s
  cross *= 2 * np.arange(LT + 1) + 1
  out = []
  for h in H:
    b = familyMultipliers("S2", h)
    k = min(len(b), LT + 1)
    e = b[:k] ** 2 * energy[:k] - 2 * b[:k] * cross[:k] + A[:k]
    out.append(np.sum(e[1:]) + np.sum(A[k:]))
  return np.array(out)


# ------------------------------------------------------------------ the study

def _cons(E0, tauMax):
  h2 = lowerBoundHalfwidth(E0, H)
  return max(h2, noiseFloorHalfwidth(E0, H, tauMax=tauMax, pilot=h2))


METHODS = {
  "UCV": lambda E1, E0: ucvHalfwidth(E1, H),
  "C2 z=2": lambda E1, E0: lowerBoundHalfwidth(E0, H, z=2),
  "C2 bonf": lambda E1, E0: lowerBoundHalfwidth(E0, H),
  "cons .05": lambda E1, E0: _cons(E0, 0.05),
  "cons .1": lambda E1, E0: _cons(E0, 0.1),
  "cons .2": lambda E1, E0: _cons(E0, 0.2),
}


# the estimate and the smoothed truth on a grid: the largest fluctuation beyond tauRel K_h*f,
# relative to the maximum of K_h*f, the quantity the C3 floor bounds by tauMax
class Fluctuation:

  def __init__(self, m, rng, nGrid=20000):
    self.u, self.comps = m
    G = rng.standard_normal((nGrid, 3))
    self.G = G / np.linalg.norm(G, axis=1, keepdims=True)
    self.P = [_legendre(self.G @ c, LT) for (_, c, _, _) in self.comps]
    self.plan = S2FourierPlan(vector3d(self.G), LT, real=True)
    self.deg = np.floor(np.sqrt(np.arange((LT + 1) ** 2))).astype(int)
    self._fh = {}

  def smoothed(self, h):
    if h not in self._fh:
      b = np.zeros(LT + 1)
      bb = familyMultipliers("S2", h)[:LT + 1]
      b[:len(bb)] = bb
      l = np.arange(LT + 1)
      self._fh[h] = self.u + sum(w * np.einsum("l,lg->g", (2 * l + 1) * b * a, P) for (w, _, a, _), P in zip(self.comps, self.P))
    return self._fh[h]

  def excess(self, a, h, tauRel=0.5):
    b = np.zeros(LT + 1)
    bb = familyMultipliers("S2", h)[:LT + 1]
    b[:len(bb)] = bb
    fhat = self.plan(4 * np.pi * b[self.deg] * a)
    fh = self.smoothed(h)
    return float(np.max(np.abs(fhat - fh) - tauRel * fh) / np.max(fh))


def study(models, ns, reps, L0, fluct, seed=1):
  rng = np.random.default_rng(seed)
  for name in models:
    m = model(name)
    axial = name.endswith("axial")
    A = trueEnergies(m)
    fl = Fluctuation(m, rng) if fluct else None
    print(f"\nmodel {name}: {ALGORITHM['cases'][name]}")
    print("cell: median h [deg], median/worst MISE over the oracle's, share below the oracle" +
          (", median excess fluctuation over tauRel f_h in units of max f_h" if fluct else ""))
    print(f"{'n':>8} {'oracle':>7} {'filter':>6} | " + " | ".join(f"{k:>{30 if fluct else 24}}" for k in METHODS))
    for n in ns:
      hO, rO = oracle(A, n, axial)
      rows = {k: [] for k in METHODS}
      for _ in range(reps if n < 1e5 else max(reps // 5, 3)):
        x = sample(m, int(n), rng)
        if axial:
          x *= rng.choice([-1, 1], len(x))[:, None]
        v = vector3d(x, antipodal=axial)
        E1 = degreeEnergies(v, LT, folds=1, rng=rng)
        E0 = degreeEnergies(v, L0, rng=rng)
        a = S2FourierPlan(v, LT, real=True).adjoint(np.ones(len(x)), np.full(len(x), 1 / len(x))) if fluct else None
        if fluct and axial:
          a = evenize(a)
        for k, f in METHODS.items():
          h = f(E1, E0)
          rows[k].append((h, exactRisk(A, n, h, axial) / rO, fl.excess(a, h) if fluct else 0.0))
      cells = []
      for k in METHODS:
        hs, rr, ex = np.array(rows[k]).T
        c = f"{np.median(hs) / degree:5.1f} {np.median(rr):5.2f}/{np.max(rr):5.2f} {np.mean(hs < hO * 0.999) * 100:3.0f}%"
        cells.append(c + (f" {np.median(ex):5.2f}" if fluct else ""))
      print(f"{n:8.0e} {hO / degree:7.2f} {filterOracle(A, n, axial) / rO:6.2f} | " + " | ".join(f"{c:>{30 if fluct else 24}}" for c in cells))


def timing():
  rng = np.random.default_rng(2)
  m = model("A")
  for n in (10**5, 10**6, 10**7):
    x = vector3d(sample(m, n, rng))
    for L, label in ((32, "conservative"), (LT, "UCV")):
      t = time.perf_counter()
      E = degreeEnergies(x, L, folds=8 if L == 32 else 1, rng=rng)
      t1 = time.perf_counter()
      h = _cons(E, 0.05) if L == 32 else ucvHalfwidth(E, H)
      t2 = time.perf_counter()
      print(f"n {n:.0e} {label:>12}: energies {t1 - t:.3f} s, selection {t2 - t1:.3f} s, h {h / degree:.2f} deg")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "timing":
    timing()
    sys.exit()
  ap = argparse.ArgumentParser()
  ap.add_argument("--reps", type=int, default=20)
  ap.add_argument("--n", default="1e2,1e3,1e4,1e5,1e6")
  ap.add_argument("--models", default="A,B")
  ap.add_argument("--L0", type=int, default=32)
  ap.add_argument("--fluct", action="store_true")
  a = ap.parse_args()
  study(a.models.split(","), [float(s) for s in a.n.split(",")], a.reps, a.L0, a.fluct)

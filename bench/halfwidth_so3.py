# the halfwidth selectors on SO(3) against the oracle (docs/plans/halfwidth.md, sections 7 and
# 8, H2): samples of ODFs with crystal and specimen symmetry, the truth in its harmonic form, the
# realised ISE of every candidate from the projected empirical coefficients, the oracle the
# candidate of least mean ISE over the paired replicates; and grain-clustered pixels, selected
# with and without their grain ids
#
#   .venv/bin/python bench/halfwidth_so3.py [--reps R] [--n 1e3,1e4,1e5] [--models uni10,...]
#   .venv/bin/python bench/halfwidth_so3.py timing
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtex import (Miller, SantaFe, SO3FunHarmonic, calcKernel, crystalFrame, degree, discreteSample, fibreODF,
                  orientation, rotation, specimenFrame, uniformODF, unimodalODF, vector3d)
from mtex.functions.halfwidth import (candidateHalfwidths, degreeEnergies, familyMultipliers, lowerBoundHalfwidth,
                                      noiseFloorHalfwidth, ucvHalfwidth)
from mtex.functions.harmonics import deg2dim, invariantProjection, project
from mtex.functions.transforms import SO3FourierPlan

ALGORITHM = {
  "name": "halfwidth selection on SO(3)",
  "entry": {"python": ["mtex.functions.halfwidth.selectHalfwidth", "mtex.functions.halfwidth.degreeEnergies"]},
  "cases": {
    "uni5": "m-3m, one dlVP component of 5 deg", "uni10": "m-3m, 10 deg", "uni20": "m-3m, 20 deg",
    "tri10": "-1, one dlVP component of 10 deg", "hex10": "6/mmm with 222 specimen symmetry, 10 deg",
    "santafe": "the SantaFe ODF, m-3m with mmm specimen symmetry",
    "mix": "m-3m, 0.5 uniform + 0.3 dlVP 7 deg + 0.2 dlVP 15 deg",
    "grains": "m-3m pixels of lognormal grains drawn from 'mix', 1 deg intragranular scatter",
  },
}

LT = 96
H = np.array([h for h in candidateHalfwidths("SO3", 2 * degree, 40 * degree, 40) if len(familyMultipliers("SO3", h)) - 1 <= LT])


def model(name, rng):
  cs = crystalFrame("m-3m")
  ori = lambda c, s=None: orientation(rotation.rand(rng=rng), c) if s is None else orientation(rotation.rand(rng=rng), c, s)
  if name.startswith("uni"):
    return unimodalODF(ori(cs), halfwidth=float(name[3:]) * degree)
  if name == "tri10":
    return unimodalODF(ori(crystalFrame("-1")), halfwidth=10 * degree)
  if name == "hex10":
    c6, s2 = crystalFrame("6/mmm"), specimenFrame("222")
    return unimodalODF(ori(c6, s2), halfwidth=10 * degree)
  if name == "santafe":
    return SantaFe()
  if name in ("mix", "grains"):
    return 0.5 * uniformODF(cs) + 0.3 * unimodalODF(ori(cs), halfwidth=7 * degree) + 0.2 * unimodalODF(ori(cs), halfwidth=15 * degree)
  raise ValueError(name)


# grains of lognormal sizes with orientations from the model, their pixels scattered about them
def grainSample(odf, n, rng, meanSize=50, scatter=1 * degree):
  sizes = np.maximum(1, np.round(rng.lognormal(np.log(meanSize) - 0.5, 1.0, size=4 * n // meanSize))).astype(int)
  sizes = sizes[:np.searchsorted(np.cumsum(sizes), n) + 1]
  g = discreteSample(odf, len(sizes), rng=rng)
  gid = np.repeat(np.arange(len(sizes)), sizes)
  omega = np.abs(rng.normal(0, scatter, len(gid)))
  pix = rotation(g.data.reshape(-1, 4)[gid]) * rotation.byAxisAngle(vector3d.rand(len(gid), rng=rng), omega)
  return orientation(pix, g.CS, g.SS), gid


class Truth:

  def __init__(self, odf):
    self.f = SO3FunHarmonic(odf, bandwidth=LT)
    self.CS, self.SS = self.f.CS, self.f.SS
    # the truth's coefficients, zero above its own bandwidth
    self.fl = np.zeros(deg2dim(LT + 1), dtype=complex)
    fh = np.asarray(self.f._fhat).ravel()[:deg2dim(LT + 1)]
    self.fl[:len(fh)] = fh
    self.P, self.Q = invariantProjection(self.SS, LT), invariantProjection(self.CS, LT)

  # the realised ISE of every candidate: sum_l ||b_l mu_l - f_l||^2
  def ise(self, o):
    q = o.data.reshape(-1, 4)
    mu = project(SO3FourierPlan(rotation(q), LT, real=True).adjoint(np.ones(len(q)), np.full(len(q), 1 / len(q))), self.P, self.Q)
    out = []
    for h in H:
      b = np.zeros(LT + 1)
      bb = familyMultipliers("SO3", h)
      b[:len(bb)] = bb
      d = np.concatenate([np.full((2 * l + 1) ** 2, b[l]) for l in range(LT + 1)])
      e = np.abs(d * mu - self.fl) ** 2
      e[0] = 0
      out.append(np.sum(e))
    return np.array(out)


def _floored(E, h, tauMax):
  return max(h, noiseFloorHalfwidth(E, H, tauMax=tauMax, pilot=h))


def methods(clustered):
  m = {
    "UCV": lambda E1, E0, o: ucvHalfwidth(E1, H),
    "UCV+f .05": lambda E1, E0, o: _floored(E1, ucvHalfwidth(E1, H), 0.05),
    "UCV+f .1": lambda E1, E0, o: _floored(E1, ucvHalfwidth(E1, H), 0.1),
    "UCV+f .2": lambda E1, E0, o: _floored(E1, ucvHalfwidth(E1, H), 0.2),
    "C2": lambda E1, E0, o: lowerBoundHalfwidth(E0, H),
    "cons .1": lambda E1, E0, o: _floored(E0, lowerBoundHalfwidth(E0, H), 0.1),
    "magicRule": lambda E1, E0, o: calcKernel(o, "magicRule").halfwidth(),
    "10deg": lambda E1, E0, o: 10 * degree,
  }
  if not clustered:
    m["KLCV"] = lambda E1, E0, o: calcKernel(o, "klcv", rng=np.random.default_rng(0)).halfwidth() if o.size <= 20000 else np.nan
  return m


def study(names, ns, reps, L0, seed=1):
  rng = np.random.default_rng(seed)
  for name in names:
    odf = model(name, rng)
    truth = Truth(odf)
    clustered = name == "grains"
    variants = [("groups", True), ("iid", False)] if clustered else [("", False)]
    print(f"\nmodel {name}: {ALGORITHM['cases'][name]}")
    print("cell: median h [deg], median/worst ISE over the oracle's in the same replicate, share below the oracle")
    for n in ns:
      rows, curves = {}, []
      for _ in range(reps):
        if clustered:
          o, gid = grainSample(odf, int(n), rng)
        else:
          o, gid = discreteSample(odf, int(n), rng=rng), None
        ise = truth.ise(o)
        curves.append(ise)
        for label, useGroups in variants:
          g = gid if useGroups else None
          E1 = degreeEnergies(o, LT, groups=g, folds=8 if useGroups else 1, rng=rng)
          E0 = degreeEnergies(o, L0, groups=g, rng=rng)
          for k, f in methods(clustered).items():
            if label == "iid" and k in ("magicRule", "10deg"):
              continue
            h = f(E1, E0, o)
            key = f"{k} {label}".strip()
            rows.setdefault(key, []).append((h, np.interp(h, H, ise) if np.isfinite(h) else np.nan))
      mean = np.mean(curves, axis=0)
      hO = H[int(np.argmin(mean))]
      head = f"n {n:.0e}, oracle {hO / degree:.2f} deg"
      if clustered:
        head += f", {len(np.unique(gid))} grains"
      print(head)
      for key, r in rows.items():
        hs, e = np.array(r).T
        if np.all(np.isnan(hs)):
          continue
        ratio = e / np.array([np.interp(hO, H, c) for c in curves])
        print(f"  {key:>12}: {np.nanmedian(hs) / degree:6.2f}  {np.nanmedian(ratio):6.2f}/{np.nanmax(ratio):6.2f}  {np.nanmean(hs < hO * 0.999) * 100:3.0f}%")


def timing():
  rng = np.random.default_rng(2)
  cs = crystalFrame("m-3m")
  odf = model("mix", rng)
  for n in (10**5, 10**6, 10**7):
    o = discreteSample(odf, n, rng=rng) if n <= 10**6 else orientation(rotation(np.tile(discreteSample(odf, 10**6, rng=rng).data, (10, 1))), cs)
    for L, label in ((32, "conservative"), (64, "UCV")):
      t = time.perf_counter()
      E = degreeEnergies(o, L, folds=8 if L == 32 else 1, rng=rng)
      t1 = time.perf_counter()
      Hs = candidateHalfwidths("SO3")
      h0 = lowerBoundHalfwidth(E, Hs) if L == 32 else ucvHalfwidth(E, Hs)
      h = max(h0, noiseFloorHalfwidth(E, Hs, pilot=h0))
      t2 = time.perf_counter()
      print(f"n {n:.0e} {label:>12}: energies {t1 - t:.3f} s, selection {t2 - t1:.3f} s, h {h / degree:.2f} deg", flush=True)


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "timing":
    timing()
    sys.exit()
  ap = argparse.ArgumentParser()
  ap.add_argument("--reps", type=int, default=10)
  ap.add_argument("--n", default="1e3,1e4,1e5")
  ap.add_argument("--models", default="uni10,mix")
  ap.add_argument("--L0", type=int, default=32)
  a = ap.parse_args()
  study(a.models.split(","), [float(s) for s in a.n.split(",")], a.reps, a.L0)

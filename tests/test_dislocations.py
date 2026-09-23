# M7: the dislocation content, the Julia I9b
import numpy as np

from mtex import (EBSD, phaseList, crystalFrame, rotation, orientation, degree, dislocationSystem, fitDislocationSystems, calcGND, curvature,
                  setMTEXpref, getMTEXpref)
from mtex.analysis.dislocations import systemCurvature
from conftest import close
from test_intragranular import linearMap


def test_I9b_the_dislocation_content():
  cs = crystalFrame("m-3m", mineral="test")
  dS = dislocationSystem.fcc(cs)
  # a curvature made of one system is explained by it, at the density it was made with,
  # since a single system is the cheapest way to carry its own curvature
  d = dS[0]
  kappa1 = systemCurvature(d.b.data[0], d.l.data[0])
  rho0 = 0.004
  rho, factor = fitDislocationSystems(rho0 * kappa1, dS)
  assert np.all(np.isfinite(rho)) and np.sum(np.abs(rho) * dS.u) <= rho0 * d.u[0] * (1 + 1e-6) and factor == 1
  # whatever combination it found, it reproduces the curvature
  M = np.einsum("j,jkl->kl", rho, systemCurvature(dS.b.data, dS.l.data))
  assert np.max(np.abs(M - rho0 * kappa1)) < 1e-9
  # the unknown components of a map's tensor are left out of the fit
  partial = rho0 * kappa1
  partial[:, 2] = np.nan
  assert np.all(np.isfinite(fitDislocationSystems(partial, dS)[0]))
  # on a map: a constant orientation has no dislocations at all
  i, j = np.meshgrid(np.arange(6), np.arange(6), indexing="xy")
  pos = np.stack([i.reshape(-1), j.reshape(-1), np.zeros(36)], axis=-1).astype(float)
  flat = EBSD(pos, rotation(np.tile(rotation.id.data, (36, 1))), np.ones(36, dtype=np.int8), phaseList(cs))
  gnd, dens = flat.calcGND(dS)
  assert gnd.shape == flat.shape and np.nanmax(np.abs(dens)) < 1e-9 and np.nanmax(gnd) < 1e-9 * 1e16
  # and the linear field has a curvature the systems do reproduce
  e = linearMap(np.eye(2), 6, phases=phaseList(cs))
  gnd2, dens2 = calcGND(e, dS)
  assert np.all(gnd2 > 0) and dens2.shape == (e.size, len(dS))
  kappa = curvature(e)
  assert kappa.unit == "1/um" and kappa.shape == e.shape
  kap = kappa.M.reshape(-1, 3, 3)
  known = ~np.isnan(kap[0])
  # the worked route of MATLAB's page: the systems rotated into every pixel, the densities in
  # 1/(um au) and the factor 1e16 to 1/m^2, the content exactly calcGND's
  rho, factor = fitDislocationSystems(kappa, e.orientations.reshape(-1, 1) * dS)
  assert factor == 1e16 and np.array_equal(rho, dens2, equal_nan=True)
  assert np.array_equal(factor * np.sum(np.abs(rho * dS.u), axis=1), gnd2.reshape(-1))
  for k in (7, 14):
    o = orientation._fromRotation(e.rotations.data[k], (cs, e.frame))
    M2 = np.einsum("j,jkl->kl", dens2[k], systemCurvature((o * dS.b).data, (o * dS.l).data))
    assert np.max(np.abs(M2[known] - kap[k][known])) < 1e-9


def test_the_simplex_and_HiGHS_agree_on_the_dislocation_fit():
  cs = crystalFrame("m-3m", mineral="test")
  dS = dislocationSystem.fcc(cs)
  rng = np.random.default_rng(4)
  e = linearMap(np.eye(2), 6, phases=phaseList(cs), shift=lambda i, j: 0.02 * rng.random(len(i)))
  before = getMTEXpref("kernels", "numba")
  try:
    setMTEXpref("kernels", "numba")
    g1, r1 = e.calcGND(dS)
    setMTEXpref("kernels", "numpy")
    g2, r2 = e.calcGND(dS)
  finally:
    setMTEXpref("kernels", before)
  # the energy is the value of the program, which both solvers reach; the densities may
  # differ where the optimum is not unique, but both reproduce the curvature
  assert np.array_equal(np.isnan(g1), np.isnan(g2)) and np.allclose(g1, g2, rtol=1e-6, atol=1e-12, equal_nan=True)
  kap = curvature(e).M.reshape(-1, 9)
  known = ~np.isnan(kap[0])
  for k in np.flatnonzero(~np.isnan(g1)):
    o = orientation._fromRotation(e.rotations.data[k], (cs, e.frame))
    K = systemCurvature((o * dS.b).data, (o * dS.l).data).reshape(len(dS), 9)
    assert np.max(np.abs(r1[k] @ K[:, known] - kap[k][known])) < 1e-8 and np.max(np.abs(r2[k] @ K[:, known] - kap[k][known])) < 1e-8

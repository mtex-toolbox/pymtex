# fast multiscale clustering, gbcFMC and calcGrains(ebsd, fmc=...)
import os

import numpy as np
import pytest

from mtex import (EBSD, calcGrains, crystalFrame, degree, gbcAngle, gbcFMC, phaseList, rotation, setMTEXpref, vector3d, xvector,
                  zvector)
from mtex.maps import fmc as F

CS = crystalFrame("m-3m", mineral="test")


# nb x nb blocks of bs x bs pixels, each block turned by step against the last, with scatter
# and a bending of the lattice along x inside every block
def blockMap(nb=4, bs=20, step=7 * degree, noise=0.3 * degree, bend=0.0, seed=0, phases=None):
  rng = np.random.default_rng(seed)
  n = nb * bs
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  base = rotation.byAxisAngle(zvector, (i // bs + (j // bs) * nb) * step)
  scatter = rotation.byAxisAngle(vector3d(rng.normal(size=(n * n, 3))), np.abs(rng.normal(size=n * n)) * noise)
  rot = scatter * base * rotation.byAxisAngle(xvector, bend * (i % bs))
  ph = np.ones(n * n, dtype=np.int8) if phases is None else phases(i, j)
  return EBSD(pos, rot, ph, phaseList([CS, crystalFrame("6/mmm", mineral="hex")]) if phases is not None else phaseList(CS))


# at a low angle the boundary may land a pixel or two off, as MATLAB's notes on gbcFMC say
def test_the_blocks_of_a_map_are_found():
  for step, tol in ((7 * degree, 0), (2 * degree, 12)):
    g = calcGrains(blockMap(step=step), fmc=3)
    assert len(g) == 16 and np.all(np.abs(g.numPixel - 400) <= tol)


def test_calcGrains_with_fmc_is_the_criterion():
  a = calcGrains(blockMap(), fmc=2, minPixel=5)
  b = calcGrains(blockMap(), gbcFMC(2), minPixel=5)
  assert np.array_equal(a.numPixel, b.numPixel)
  crit = gbcFMC(2, quatmax=4 * degree)
  assert crit.handlesMinPixel and not gbcAngle().handlesMinPixel
  assert crit.setMinPixel(7).minPixel == 7 and crit.minPixel == 1 and crit.setMinPixel(7).quatmax == crit.quatmax
  assert "cmaha 2" in repr(crit)


# every phase is clustered on its own over its own pixels, the not indexed ones left to calcGrains
def test_every_phase_is_clustered_on_its_own():
  e = blockMap(nb=2, bs=20, phases=lambda i, j: np.select([i < 10, i < 20], [0, 1], 2).astype(np.int8))
  g = calcGrains(e, fmc=3)
  assert sorted(zip(g.phaseId.tolist(), g.numPixel.tolist())) == [(1, 200), (1, 200), (2, 400), (2, 400)]


def test_the_seed_loop_and_its_fallback_agree():
  e = blockMap(nb=3, bs=12, step=3 * degree, bend=0.2 * degree)
  a = calcGrains(e.copy(), fmc=3).numPixel
  setMTEXpref("kernels", "numpy")
  b = calcGrains(e.copy(), fmc=3).numPixel
  assert np.array_equal(a, b)


def test_the_report(capsys):
  calcGrains(blockMap(), fmc=3, verbose=True)
  out = capsys.readouterr().out
  assert "fast multiscale clustering of test" in out and "16 grains" in out


def test_the_tangent_of_a_small_rotation():
  a = np.array([[1.0, 0, 0, 0]])
  b = rotation.byAxisAngle(zvector, 2 * degree).data.reshape(1, 4)
  assert np.allclose(F.tangent(a, b), [[0, 0, 2]]) and np.allclose(F.tangent(a, -b), [[0, 0, 2]])


LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")


# the counts of MATLAB's doc/Grains/GrainReconstruction.m, read through bench/matlab/probe_fmc.m;
# the seeds are taken in order of coupling and size, whose ties the order of the pixels
# decides, so the map is taken in MATLAB's order, y running fastest
@pytest.mark.skipif(not os.path.isfile(os.path.join(LOCAL, "EBSD", "EMSphinx.h5")), reason="the MTEX data folder is not here")
def test_fmc_counts_as_MATLAB():
  from mtex import inpolygon, mtexdata
  ebsd = mtexdata("EMSphinx", root=LOCAL, download=False)["Iron fcc"]
  ebsd = ebsd[inpolygon(ebsd, [40, 30, 80, 60])]
  P = ebsd.pos.data.reshape(-1, 3)
  order = np.lexsort((P[:, 1], P[:, 0]))
  e = EBSD(P[order], rotation(ebsd.rotations.data.reshape(-1, 4)[order]), ebsd.phaseId.reshape(-1)[order], ebsd.CSList)
  assert [len(calcGrains(e.copy(), fmc=c, minPixel=10)) for c in (0.5, 1.5)] == [55, 107]

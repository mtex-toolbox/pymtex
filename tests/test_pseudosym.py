# the correction of a pseudo symmetry, MATLAB cleanUpPseudoSym and the Julia pseudosym.jl
import numpy as np
import pytest

from mtex import EBSD, Miller, calcGrains, cleanUpPseudoSym, crystalFrame, degree, orientation, phaseList, rotation
from mtex.maps.pseudosym import pseudoOperators

CS = crystalFrame("mmm", [4.8, 10.2, 6.0], mineral="Forsterite")


# one crystal of 30 x 30 pixels, part of it indexed as its pseudo-symmetric twin inside a
# comb of ragged outline, so the boundary between the two is an indexing artefact
def combMap(psSym):
  n = 30
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  o0 = rotation.byEuler(20 * degree, 35 * degree, 50 * degree)
  twin = (i >= 8) & (i < 22) & (j >= 6) & (j < 24) & ((j < 9) | (i % 3 != 0))
  q = np.tile(o0.data.reshape(1, 4), (n * n, 1))
  from mtex.geometry.kernels import qmul
  q[twin] = qmul(q[twin], psSym.data.reshape(1, 4))
  pos = np.stack([i, j, np.zeros(n * n)], axis=-1).astype(float)
  return EBSD(pos, rotation(q), np.ones(n * n, dtype=np.int8), phaseList(CS)), twin


def test_the_operators_are_the_cosets_of_the_pseudo_symmetry():
  psSym = orientation.byAxisAngle(Miller(1, 0, 0, CS, "uvw"), 60 * degree)
  ops = pseudoOperators(psSym)
  assert len(ops) == 2
  # none lies in the proper group, and no two in one coset
  G = CS.sym.properGroup().rotations().data.reshape(-1, 4)
  from mtex.geometry.kernels import qmul
  for k, c in enumerate(ops):
    assert np.max(np.abs(qmul(c[None], G)[:, 0])) < 1 - 1e-6
    for d in ops[k + 1:]:
      assert np.max(np.abs(qmul(c[None], G) @ d)) < 1 - 1e-6
  # a true symmetry leaves nothing to correct
  assert len(pseudoOperators(orientation.byAxisAngle(Miller(1, 0, 0, CS, "uvw"), 180 * degree))) == 0


def test_a_ragged_pseudo_symmetric_boundary_is_undone():
  psSym = orientation.byAxisAngle(Miller(1, 0, 0, CS, "uvw"), 60 * degree)
  ebsd, twin = combMap(psSym)
  grains = calcGrains(ebsd, angle=10 * degree)
  assert grains.size == 2
  out, grainsM, numChanged = cleanUpPseudoSym(ebsd, grains, psSym)
  # the map is corrected in place: one grain, one orientation, the twin moved back
  assert out is ebsd and grainsM.size == 1 and numChanged == twin.sum()
  assert np.all(ebsd.grainId == 1) and np.array_equal(grains.parentId, [1, 1])
  o = orientation(ebsd.rotations, CS, ebsd.frame)
  assert np.max(o.angle(o.flatten()[0])) < 1e-6
  # a boundary that is not the pseudo symmetry is left alone
  ebsd2, _ = combMap(orientation.byAxisAngle(Miller(0, 0, 1, CS, "uvw"), 40 * degree))
  grains2 = calcGrains(ebsd2, angle=10 * degree)
  _, g2, n2 = cleanUpPseudoSym(ebsd2, grains2, psSym)
  assert g2.size == 2 and n2 == 0
  with pytest.raises(ValueError):
    cleanUpPseudoSym(ebsd, grains, rotation(psSym.data))

# G7: slip and dislocation systems, the Julia slipsystem.jl
import numpy as np
import pytest

from mtex import slipSystem, dislocationSystem, crystalFrame, Miller, orientation, vector3d, zvector, degree
from conftest import close


def test_slip_systems_of_a_family_and_their_Schmid_factors():
  cs = crystalFrame("m-3m", [3.6, 3.6, 3.6], mineral="Cu")
  s = slipSystem.fcc(cs)
  assert isinstance(s, slipSystem) and s.CS is cs and s.isCrystal and len(s) == 1
  with pytest.raises(ValueError):
    slipSystem(Miller(1, 0, 0, cs, "uvw"), Miller(1, 0, 0, cs))
  sym = s.symmetrise()
  assert len(sym) == 24 and len(s.symmetrise(antipodal=True)) == 12 and len(slipSystem.bcc(cs).symmetrise()) == 2 * (12 + 12 + 24)
  assert np.all(np.abs(np.sum(sym.b.normalize().data * sym.n.normalize().data, axis=-1)) < 1e-10)
  assert len(sym[3:7]) == 4 and len(list(sym)) == 24 and sym[0] == sym[0] and not (sym[0] == sym[1]) and not ((-sym[0]) == sym[0]) and sym[0] != sym[1]
  # the Schmid factor of tension along [001] for the fcc systems: 0 or 1 / sqrt(6)
  m = sym.SchmidFactor(Miller(0, 0, 1, cs))
  assert m.shape == (24,) and np.all((np.abs(m) < 1e-10) | (np.abs(np.abs(m) - 1 / np.sqrt(6)) < 1e-10)) and close(m.max(), 1 / np.sqrt(6))
  assert sym.SchmidFactor(Miller([[0, 0, 1], [1, 0, 0]], cs)).shape == (24, 2)
  assert close(slipSystem.fcc(cs, CRSS=2).symmetrise().SchmidFactor(Miller(0, 0, 1, cs), relative=True).max(), 1 / np.sqrt(6) / 2)
  # rotated into the specimen, the frame follows and a crystal direction no longer fits
  o = orientation.byEuler(0, 0, 0, cs)
  sr = o * s
  assert not sr.isCrystal and sr.CS is None and close(sr.SchmidFactor(vector3d(0, 0, 1)), s.SchmidFactor(Miller(0, 0, 1, cs)))
  assert close(s.rotate(o).b.data, sr.b.data)
  with pytest.raises(ValueError):
    sr.SchmidFactor(Miller(0, 0, 1, cs))
  # tensors and the pairwise measures
  M = s.deformationTensor()
  assert M.shape == (1, 3, 3) and abs(np.trace(M[0])) < 1e-12 and close(s.SchmidTensor()[0], (M[0] + M[0].T) / 2)
  assert close(s.mPrime(s), 1) and s.residualBurgersVector(s)[0, 0] == 0 and abs(s.trace()[0].dot(s.n[0], noSymmetry=True)) < 1e-10
  assert s.trace().antipodal and close(s.trace(zvector).norm(), 1)
  assert "111" in repr(s) and "CRSS" in repr(slipSystem.fcc(cs, CRSS=2))
  # the hexagonal families and the refusal of a default
  csh = crystalFrame("6/mmm", [3.2, 3.2, 5.2], mineral="Mg")
  assert len(slipSystem.basal(csh).symmetrise(antipodal=True)) == 3 and len(slipSystem.pyramidalCA(csh).symmetrise(antipodal=True)) == 12
  assert len(slipSystem.prismaticA(csh).symmetrise(antipodal=True)) == 3 and len(slipSystem.twinT1(csh).symmetrise(antipodal=True)) == 6
  with pytest.raises(ValueError):
    slipSystem.hcp(csh)


def test_dislocation_systems_of_fcc():
  cs = crystalFrame("m-3m", [3.6, 3.6, 3.6], mineral="Cu")
  ds = dislocationSystem.fcc(cs)
  # 12 edge and 6 screw systems, with the half Burgers vector
  assert np.count_nonzero(ds.isEdge) == 12 and np.count_nonzero(ds.isScrew) == 6 and len(ds) == 18 and ds.CS is cs
  assert close(ds.b.norm(), 3.6 * np.sqrt(2) / 2) and np.all(ds.u == np.where(ds.isEdge, 2, 1))
  assert ds.tensor().shape == (18, 3, 3) and ds.l.antipodal and len(ds[ds.isScrew]) == 6
  edge = ds[ds.isEdge]
  assert np.all(np.abs(np.sum(edge.b.data * edge.l.data, axis=-1)) < 1e-10)
  # from vectors of one's own, and a line vector that is neither along nor across
  d = dislocationSystem(vector3d(1, 0, 0), vector3d(0, 1, 0), 3)
  assert d.isEdge[0] and d.u[0] == 3 and dislocationSystem(vector3d(1, 0, 0), vector3d(2, 0, 0)).isScrew[0]
  with pytest.raises(ValueError):
    dislocationSystem(vector3d(1, 0, 0), vector3d(1, 1, 0))
  # rotated into the specimen
  o = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs)
  r = o * ds
  assert r.CS is None and close(r.b.norm(), ds.b.norm()) and "edge: 12" in repr(ds)
  assert len(dislocationSystem.bcc(cs)) > 18
  with pytest.raises(ValueError):
    dislocationSystem.hcp()

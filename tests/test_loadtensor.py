# reading the tensor files of the sample data (docs/plans/io.md I7)
import os

import numpy as np
import pytest

from mtex import Voigt, checkSymmetry, complianceTensor, crystalFrame, loadTensor, stiffnessTensor, tensor

LOCAL = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "tensor")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "tensor")


def test_the_olivine_stiffness_file():
  cs = crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="Olivine")
  C = stiffnessTensor.load(os.path.join(FIXTURES, "Olivine1997PC.GPa"), cs)
  assert isinstance(C, stiffnessTensor) and C.frame is cs and C.rank == 4
  assert np.isclose(Voigt(C)[0, 0], 320.5) and np.isclose(Voigt(C)[1, 2], 76.8) and np.isclose(Voigt(C)[5, 5], 78.7)
  assert np.isclose(C.density, 3.355) and checkSymmetry(C)
  # the plain loader reads the rank off the block, and a density given wins
  T = loadTensor(os.path.join(FIXTURES, "Olivine1997PC.GPa"), cs)
  assert type(T) is tensor and T.rank == 4 and np.allclose(Voigt(T), Voigt(C))
  assert stiffnessTensor.load(os.path.join(FIXTURES, "Olivine1997PC.GPa"), cs, density=3.3).density == 3.3
  assert complianceTensor.load(os.path.join(FIXTURES, "Olivine1997PC.GPa"), cs).doubleConvention


def test_the_piezoelectric_file():
  cs = crystalFrame("32", [4.916, 4.916, 5.4054], mineral="Quartz")
  P = tensor.load(os.path.join(FIXTURES, "Single_RH_quartz_poly.P"), cs, doubleConvention=True)
  assert P.rank == 3 and P.doubleConvention and Voigt(P).shape == (3, 6)
  assert np.isclose(Voigt(P)[1, 0], 2.3) and np.isclose(Voigt(P)[0, 3], -0.67) and np.isclose(Voigt(P)[1, 4], 0.67)
  assert np.isclose(P.M[1, 0, 2], 0.67 / 2) and np.isclose(P.M[1, 2, 0], 0.67 / 2) and P.frame is cs
  assert np.isclose(tensor.load(os.path.join(FIXTURES, "Single_RH_quartz_poly.P"), cs).M[1, 0, 2], 0.67)


@pytest.mark.skipif(not os.path.isdir(LOCAL), reason="the MTEX data folder is not here")
def test_the_rank_two_file():
  cs = crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="Olivine")
  T = tensor.load(os.path.join(LOCAL, "Fosterite_PC_model.TEXP"), cs)
  assert T.rank == 2 and np.allclose(np.diag(T.M), [9.435, 14.846, 11.808]) and T.frame is cs

# the linear program and the Taylor model: the crystal solve, the solve over orientations
# and the fields of the factor and the spin (docs/plans/functions.md P4e: X2, X3, X7 and
# the Julia Taylor tests)
import numpy as np
import pytest

from mtex import (SO3FunHarmonic, SO3VectorFieldHarmonic, calcTaylor, crystalFrame, degree, inv, linearProgram, orientation,
                  rotation, slipSystem, specimenFrame, spinTensor, strainTensor, vector3d)
from mtex.analysis.taylor import TaylorModel

rng = np.random.default_rng(31)


def test_linearProgram_findsAKnownOptimum():
  # a program with a known optimum: min x1 + 2x2 + 3x3 with x1 + x2 + x3 = 1, x >= 0
  assert np.allclose(linearProgram([1.0, 2, 3], np.ones((1, 3)), [1.0]), [1, 0, 0])
  # one that needs both variables
  A = np.array([[1.0, 1, 0], [0, 1, 1]])
  x = linearProgram([1.0, 5, 1], A, [1.0, 1])
  assert np.allclose(A @ x, [1, 1]) and np.all(x >= -1e-12) and np.isclose([1.0, 5, 1] @ x, 2)


def test_linearProgram_diagnosticsSayWhyThereIsNoSolution():
  # an infeasible one has no solution, and the diagnostics say why
  assert linearProgram([1.0, 1], [[1.0, 1], [1, 1]], [1.0, 2]) is None
  assert linearProgram([1.0], np.ones((2, 1)), [1.0, 2.0], diagnostics=True).status == "infeasible"
  assert linearProgram([-1.0], np.zeros((1, 1)), [0.0], diagnostics=True).status == "unbounded"
  assert linearProgram([2.0, 1.0], np.ones((1, 2)), [1.0], maxIter=1, diagnostics=True).status == "iteration_limit"


def test_linearProgram_solvesRedundantAndBadlyScaledRows():
  assert np.allclose(linearProgram([1.0], np.ones((2, 1)), [2.0, 2.0]), [2.0])         # a redundant row
  assert np.allclose(linearProgram([1.0, 2.0], np.array([[1e8, 1e8]]), [1e8]), [1.0, 0])  # badly scaled


def test_linearProgram_refusesNaN():
  with pytest.raises(ValueError):
    linearProgram([np.nan], np.ones((1, 1)), [1.0])


def test_linearProgram_isOptimalAgainstEveryBasis():
  # a random feasible one is solved to optimality, checked against every basis
  A = rng.standard_normal((2, 5))
  b = A @ np.abs(rng.standard_normal(5))
  c = np.abs(rng.standard_normal(5))
  x = linearProgram(c, A, b)
  assert np.allclose(A @ x, b) and np.all(x >= -1e-9)
  best = np.inf
  for i in range(4):
    for j in range(i + 1, 5):
      M = A[:, [i, j]]
      if abs(np.linalg.det(M)) < 1e-10:
        continue
      y = np.zeros(5)
      y[[i, j]] = np.linalg.solve(M, b)
      if np.all(y >= -1e-9):
        best = min(best, c @ y)
  assert np.isclose(c @ x, best, atol=1e-8)


def fccTension():
  cs = crystalFrame("m-3m")
  sS = slipSystem.fcc(cs)
  return cs, sS, TaylorModel(sS), strainTensor(np.diag([1.0, -0.5, -0.5]), cs)


# a strain of the principal values (1, -0.4, -0.6) along the axes of a random rotation
def turnedStrain(frame):
  r = rotation.rand(rng=rng)
  return strainTensor(rotation(r.data).matrix() @ np.diag([1.0, -0.4, -0.6]) @ rotation(r.data).matrix().T, frame)


def test_TaylorModel_countsTheSymmetrisedSystems():
  cs, sS, m, _ = fccTension()
  assert len(m) == 24 and len(TaylorModel(sS.symmetrise())) == 24 and len(TaylorModel(slipSystem.cat(sS, slipSystem.fcc(cs, 2)))) == 48


def test_calcTaylor_ofTheCubeInTensionIs2_45():
  # the cube orientation of an fcc crystal in tension has the Taylor factor 2.45
  cs, sS, m, eps = fccTension()
  M, b, W = calcTaylor(eps, sS)
  assert np.isclose(M, 2.449, atol=5e-3) and np.count_nonzero(b > 1e-9) <= 5
  assert np.isclose(calcTaylor(eps, m).M, M) and np.isclose(calcTaylor(eps, sS, plasticWork=True).M, M)
  assert np.isclose(calcTaylor(eps, sS, "plasticWork").M, M)


def test_calcTaylor_shearsReproduceTheStrain():
  # the shears reproduce the strain they were computed for, the spin is antisymmetric
  cs, sS, m, eps = fccTension()
  M, b, W = calcTaylor(eps, sS)
  D = np.einsum("k,kij->ij", b, m.systems.SchmidTensor().M)
  assert np.allclose((D + D.T) / 2, eps.M, atol=1e-8) and isinstance(W, spinTensor) and W.isSkewSymmetric


def test_calcTaylor_minimumNormKeepsTheFactorAndReducesTheShears():
  # the minimum norm selection keeps the factor, reduces the shears and makes the spin unique
  cs, sS, m, eps = fccTension()
  M = calcTaylor(eps, sS).M
  bal = calcTaylor(eps, sS, spinSelection="minimumNorm")
  assert np.isclose(bal.M, M) and np.isclose(np.sum(bal.b ** 2), 0.75, atol=1e-9) and bal.W.norm() < 1e-9
  for _ in range(5):
    er = turnedStrain(cs)
    basic, balanced = calcTaylor(er, m), calcTaylor(er, m, spinSelection="minimumNorm")
    assert np.isclose(basic.M, balanced.M, atol=1e-7)
    assert np.allclose(m.A @ balanced.b, [er.M[0, 0], er.M[1, 0], er.M[2, 0], er.M[1, 1], er.M[2, 1]], atol=1e-7)
    assert np.linalg.norm(balanced.b) <= np.linalg.norm(basic.b) + 1e-7


def test_calcTaylor_rejectsWhatDoesNotFit():
  # what does not fit is rejected
  cs, sS, m, eps = fccTension()
  with pytest.raises(ValueError):
    calcTaylor(strainTensor(np.eye(3), cs), m)
  with pytest.raises(ValueError):
    calcTaylor(strainTensor(eps.M, crystalFrame("222")), m)
  with pytest.raises(ValueError):
    TaylorModel(slipSystem.fcc(cs, 0))


def test_calcTaylor_ofZeroStrainIsZero():
  cs, sS, m, eps = fccTension()
  assert calcTaylor(strainTensor(np.zeros((3, 3)), cs), m).M == 0


def test_calcTaylor_overARandomTextureIs3_06():
  # the average over a random texture is the classical 3.06
  cs, sS, m, eps = fccTension()
  oris = orientation.rand(150, cs, specimenFrame.default, rng=rng)
  Ms, bs, Ws = calcTaylor(inv(oris) * strainTensor(np.diag([1.0, -0.5, -0.5])), sS)
  assert Ms.shape == (150,) and bs.shape == (150, 24) and Ws.shape == (150,) and abs(Ms.mean() - 3.06) < 0.08


@pytest.mark.parametrize("kind", ["hexagonal", "bcc"])
def test_calcTaylor_minimumNormKeepsTheFactorOfOtherModels(kind):
  # a hexagonal model with several families, and the bcc one
  if kind == "hexagonal":
    hex = crystalFrame("6/mmm", [3.2, 3.2, 5.2])
    model = TaylorModel(slipSystem.cat(slipSystem.basal(hex, 1), slipSystem.prismatic2A(hex, 4), slipSystem.pyramidalCA(hex, 7)))
  else:
    model = TaylorModel(slipSystem.bcc(crystalFrame("m-3m")))
  er = turnedStrain(model.frame)
  basic, balanced = calcTaylor(er, model, plasticWork=True), calcTaylor(er, model, plasticWork=True, spinSelection="minimumNorm")
  assert np.isclose(basic.M, balanced.M, atol=1e-7) and np.linalg.norm(balanced.b) <= np.linalg.norm(basic.b) + 1e-7


def fccAtOneOrientation():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  sS = slipSystem.fcc(cs).symmetrise()
  o = orientation.byEuler(20 * degree, 35 * degree, 10 * degree, cs, ss)
  eps = strainTensor(np.diag([1.0, -0.5, -0.5]))
  return cs, ss, sS, o, eps, calcTaylor(inv(o) * eps, sS)


def test_calcTaylor_overOrientationsIsTheSingleSolveEach():
  cs, ss, sS, o, eps, single = fccAtOneOrientation()
  batch = calcTaylor(inv(orientation(rotation.cat(rotation(o.data), rotation(o.data)), cs, ss)) * eps, sS)
  assert np.isclose(batch.M[0], single.M) and np.allclose(batch.b[0], single.b) and batch.b.shape == (2, len(sS))
  assert batch.W[1].isclose(single.W)


def test_exp_ofTheSpinTurnsTheCrystalByItsAngle():
  # the spin turns the crystal: exp(ori, -W) moves by the angle of the spin vector
  cs, ss, sS, o, eps, single = fccAtOneOrientation()
  moved = o.exp(-single.W)
  assert np.isclose(o.angle(moved, noSymmetry=True), single.W.vector().norm(), atol=1e-8)
  assert (o * (-single.W).exp()).angle(moved, noSymmetry=True) < 1e-10 and isinstance((-single.W).exp(), rotation)


def test_calcTaylor_refusesAStrainOfAnotherCrystal():
  cs, ss, sS, o, eps, single = fccAtOneOrientation()
  with pytest.raises(ValueError):
    calcTaylor(inv(orientation(rotation(o.data), crystalFrame("6/mmm"), ss)) * eps, sS)


# the harmonic Taylor factor and spin of an fcc crystal in plane strain, bandwidth 8
@pytest.fixture(scope="module")
def harmonicTaylor():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  sS = slipSystem.fcc(cs)
  eps = strainTensor(np.diag([1.0, 0.0, -1.0]))
  MF, b, WF = calcTaylor(eps, sS, bandwidth=8)
  return cs, ss, sS, eps, MF, b, WF, orientation.rand(12, cs, ss, rng=rng)


def test_calcTaylor_withABandwidthGivesTheHarmonicFields(harmonicTaylor):
  cs, ss, sS, eps, MF, b, WF, os = harmonicTaylor
  assert b is None and isinstance(MF, SO3FunHarmonic) and isinstance(WF, SO3VectorFieldHarmonic)
  assert WF.tangentSpace == "left" and WF.CS is cs and WF.SS is ss and MF.bandwidth == WF.bandwidth == 8


def test_calcTaylor_harmonicFieldsApproximateTheDirectSolve(harmonicTaylor):
  cs, ss, sS, eps, MF, b, WF, os = harmonicTaylor
  direct = calcTaylor(inv(os) * eps, sS, spinSelection="minimumNorm")
  exactV = np.stack([(os[k] * vector3d(direct.W[k].vector().data)).data.reshape(3) for k in range(12)])
  assert np.linalg.norm(np.real(MF(os)) - direct.M) / np.linalg.norm(direct.M) < 0.15
  assert np.linalg.norm(WF(os).data.reshape(12, 3) - exactV) / np.linalg.norm(exactV) < 0.4


def test_calcTaylor_harmonicFieldsAreSymmetryInvariant(harmonicTaylor):
  # the fields are invariant under the crystal group
  cs, ss, sS, eps, MF, b, WF, os = harmonicTaylor
  s = cs.rotations()[5]
  turned = orientation(rotation(os.data) * s, cs, ss)
  assert np.allclose(np.real(MF(turned)), np.real(MF(os))) and np.allclose(WF(turned).data, WF(os).data)


def test_calcTaylor_harmonicFactorAtThePageOrientation(harmonicTaylor):
  # a page value: the factor of the (0, 30, 15) orientation of the fcc crystal
  cs, ss, sS, eps, MF, b, WF, os = harmonicTaylor
  assert abs(np.real(MF(orientation.byEuler(0, 30 * degree, 15 * degree, cs, ss))) - 2.52) < 0.15


def test_the_compiled_solve_against_the_simplex():
  pytest.importorskip("numba")
  from mtex import getMTEXpref, setMTEXpref
  cs = crystalFrame("432")
  sS = slipSystem.bcc(cs).symmetrise()
  eps = strainTensor(np.diag([1.0, 0.0, -1.0]))
  os = orientation.rand(40, cs, rng=rng)
  before = getMTEXpref("kernels", "numba")
  for selection in ("basic", "minimumNorm"):
    try:
      setMTEXpref("kernels", "numba")
      fast = calcTaylor(inv(os) * eps, sS, spinSelection=selection)
      setMTEXpref("kernels", "numpy")
      slow = calcTaylor(inv(os) * eps, sS, spinSelection=selection)
    finally:
      setMTEXpref("kernels", before)
    assert np.allclose(fast.M, slow.M, atol=1e-10) and np.allclose(fast.W.M, slow.W.M, atol=1e-7)


def test_calcTaylor_compiledSolveLeavesTheDegenerateVertex():
  # the degenerate vertex on which the active set of the compiled form cycles at the tolerance
  cs = crystalFrame("432")
  sS = slipSystem.bcc(cs).symmetrise()
  E = np.array([[-0.98720962, 0.12760682, -0.02018955], [0.12760682, 0.62712738, 0.48429689],
                [-0.02018955, 0.48429689, 0.36008224]])
  E -= np.trace(E) / 3 * np.eye(3)
  single = calcTaylor(strainTensor(np.stack([E, E]), cs), sS, spinSelection="minimumNorm")
  assert np.isclose(single.M[0], 2.17502016, atol=1e-6)

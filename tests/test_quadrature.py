# the quadrature rules and the product grids of docs/plans/functions.md F2
import numpy as np
import pytest
from numpy import pi

from mtex.functions.harmonics import _WignerD
from mtex import (angle, bandwidthFor, clenshawCurtis, crystalFrame, degreeBlock, deg2dim, folds, gaussLegendre,
                  legendreP, orientation, quadratureS1Grid, quadratureS2Grid, quadratureSO3Grid, specimenFrame)


def test_clenshawCurtis_integratesPolynomials():
  x, w = clenshawCurtis(9)
  assert np.isclose(np.sum(w), 2)
  assert np.isclose(np.sum(w * x ** 8), 2 / 9)
  assert abs(np.sum(w * x ** 7)) < 1e-14


def test_gaussLegendre_integratesPolynomials():
  x, w = gaussLegendre(5)
  assert np.isclose(np.sum(w * x ** 8), 2 / 9) and np.isclose(np.sum(w), 2)


def test_legendreP_isATableOfDegreesOverPoints():
  x, w = gaussLegendre(5)
  P = legendreP(4, x)
  assert P.shape == (5, 5) and np.allclose(P[2], (3 * x * x - 1) / 2)


def test_legendreP_isOrthogonalUnderBothRules():
  # orthogonality of the Legendre polynomials under both rules
  for x, w in (clenshawCurtis(11), gaussLegendre(6)):
    P = legendreP(5, x)
    assert np.allclose((P * w) @ P.T, np.diag(2 / (2 * np.arange(6) + 1)))


def test_quadratureSO3Grid_integratesTheOrthonormalBasisExactly():
  for L in (3, 6):
    g = quadratureSO3Grid(L)
    rots, w = g.nodes(), g.weights
    assert np.isclose(np.sum(w), 1)
    # Clenshaw-Curtis is the default and needs 2L+1 polar nodes where Gauss-Legendre needs L+1
    assert len(g) == rots.size == (2 * L + 2) ** 2 * (2 * L + 1) and len(g.beta) == 2 * L + 1
    assert len(quadratureSO3Grid(L, "GaussLegendre")) == (2 * L + 2) ** 2 * (L + 1)
    # the rule integrates the orthonormal basis exactly
    D = _WignerD(rots, bandwidth=2)
    B = np.concatenate([np.sqrt(2 * n + 1) * D[n].reshape(rots.size, -1).T for n in range(3)])
    assert np.allclose((B * w) @ B.conj().T, np.eye(deg2dim(3)))


# the folded rule of a 4/mmm crystal and a 222 specimen at a bandwidth the folds divide
def foldedGrid():
  cs, ss = crystalFrame("4/mmm"), specimenFrame("222")
  L = bandwidthFor(3, cs, ss)
  return cs, ss, L, quadratureSO3Grid(L, cs, ss)


def test_quadratureSO3Grid_ofFramesIsFoldedIntoOrientationsOfThoseFrames():
  # frames give folds, orientations in those frames and a bandwidth the folds divide
  cs, ss = crystalFrame("4/mmm"), specimenFrame("222")
  assert folds(cs, ss) == (4, 2)
  assert bandwidthFor(10, cs, ss) == 11 == bandwidthFor(10, 4, 2)
  cs, ss, L, g = foldedGrid()
  assert (2 * L + 2) % 4 == 0
  assert g.folds == (4, 2) and len(g.gamma) == (2 * L + 2) // 4 and len(g.alpha) == (2 * L + 2) // 2
  assert isinstance(g.nodes(), orientation) and g.nodes().CS is cs and g.nodes().SS is ss
  assert np.isclose(np.sum(g.weights), 1)
  assert repr(g).startswith("quadratureSO3Grid (ClenshawCurtis)")


def test_quadratureSO3Grid_refusesAFoldTheBandwidthDoesNotAdmit():
  with pytest.raises(ValueError):
    quadratureSO3Grid(10, rA=4)


def test_quadratureSO3Grid_foldedAgreesWithTheFullRuleOnInvariantFunctions():
  cs, ss, L, g = foldedGrid()
  # the folded rule agrees with the full rule on the invariant functions of degree 2
  P, Q = _WignerD(ss, bandwidth=2), _WignerD(cs, bandwidth=2)
  gram = []
  for grid in (g, quadratureSO3Grid(L)):
    rots, w = grid.nodes(), grid.weights
    D = _WignerD(rots, 2)
    B = np.einsum("ab,ibc,cd->iad", P[2], D, Q[2]).reshape(rots.size, -1).T
    gram.append((B * w) @ B.conj().T)
  assert np.allclose(gram[0], gram[1], atol=1e-12) and np.linalg.norm(gram[0]) > 1e-3



# a 2-fold axis perpendicular to z on either side maps the nodes beta <= pi/2 onto the rest, so
# the values of an invariant function there give all of them; a group without one has no half
@pytest.mark.parametrize("cs, ss, scheme", [("432", "1", "ClenshawCurtis"), ("321", "1", "GaussLegendre"),
                                            ("1", "mmm", "ClenshawCurtis"), ("2/m", "1", "GaussLegendre")])
def test_quadratureSO3Grid_halfNodesGiveAllValuesOfAnInvariantFunction(cs, ss, scheme):
  CS, SS = crystalFrame(cs), specimenFrame(ss) if ss != "1" else specimenFrame.default
  g = quadratureSO3Grid(bandwidthFor(9, CS, SS), scheme, CS, SS)
  centre = orientation.rand(1, CS, SS)
  f = lambda o: np.cos(angle(o, centre))
  assert np.allclose(g.fromHalf(f(g.halfNodes())), f(g.nodes()), atol=1e-13)
  assert quadratureSO3Grid(bandwidthFor(9, crystalFrame("4")), crystalFrame("4")).halfNodes() is None

def test_quadratureS2Grid_integratesTheSphericalHarmonicsExactly():
  for L in (3, 10):
    for scheme in ("ClenshawCurtis", "GaussLegendre"):
      g = quadratureS2Grid(L, scheme)
      v, w = g.nodes(), g.weights
      assert np.isclose(np.sum(w), 4 * pi) and v.size == len(g)
      assert len(g.thetaGrid) == (2 * L + 1 if scheme == "ClenshawCurtis" else L + 1)
      from mtex import sphericalY
      Y = np.concatenate([sphericalY(l, v) for l in range(4)])
      assert np.allclose((Y * w) @ Y.conj().T, np.eye(16))


def test_quadratureS1Grid_integratesTheFourierModesExactly():
  g = quadratureS1Grid(4)
  x, w = g.nodes(), g.weights
  assert len(g) == 10 and np.isclose(np.sum(w), 1)
  k = np.arange(-4, 5)
  E = np.exp(1j * np.outer(k, x))
  assert np.allclose((E * w) @ E.conj().T, np.eye(9))

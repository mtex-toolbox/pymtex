# the statements of docs/design/harmonics.md, one test per statement
import numpy as np
import pytest
from numpy import pi

from mtex.functions.harmonics import _WignerD, _sphericalY
from mtex import (Miller, S2FunHarmonic, SO3FunHarmonic, WignerD, clebschGordan, clebschGordanTensor, crystalFrame, deg2dim, degree, degreeBlock,
                  degreeRange, dim2deg, normalizedLegendre, rotation, specimenFrame, sphericalY, symmetry, vector3d,
                  wigner3j, wignerSeries, wignerd, xvector)
from conftest import close

rng = np.random.default_rng(1)


# the Jacobi polynomial P_s^{(a,b)}(x) by its three-term recurrence, the oracle for H2
def jacobiP(s, a, b, x):
  if s == 0:
    return 1.0
  p0, p1 = 1.0, ((a + b + 2) * x + a - b) / 2
  for n in range(1, s):
    c = 2 * n + a + b
    A = 2 * (n + 1) * (n + a + b + 1) * c
    B = (c + 1) * ((c + 2) * c * x + a * a - b * b)
    C = 2 * (n + a) * (n + b) * (c + 2)
    p0, p1 = p1, (B * p1 - C * p0) / A
  return p1


def wignerdJacobi(n, k, l, beta):
  from math import comb
  a, b = abs(k - l), abs(k + l)
  s = n - (a + b) // 2
  B = np.sqrt(comb(2 * n - s, s + a) / comb(s + b, b)) * ((1 - np.cos(beta)) / 2) ** (a / 2) * ((1 + np.cos(beta)) / 2) ** (b / 2)
  return (-1 if k > l and (k - l) % 2 else 1) * B * jacobiP(s, a, b, np.cos(beta))


# a product rule on SO(3) exact for polynomials of degree 2L in the Matthies angles
def so3Quadrature(L):
  x, w = np.polynomial.legendre.leggauss(L + 1)
  m = 2 * L + 2
  i = np.arange(m)
  X, I, J = np.meshgrid(np.arange(L + 1), i, i, indexing="ij")
  nodes = rotation.byEuler(2 * pi * I.ravel() / m, np.arccos(x[X.ravel()]), 2 * pi * J.ravel() / m, "Matthies")
  return nodes, w[X.ravel()] / 2 / m ** 2


def test_deg2dim_countsTheCoefficientsBelowADegree():
  assert [deg2dim(n) for n in range(6)] == [0, 1, 10, 35, 84, 165]


def test_dim2deg_isTheDegreeOfAnIndex():
  for n in range(101):
    assert dim2deg(deg2dim(n + 1)) == n
    assert dim2deg(deg2dim(n + 1) + 3) == n


def test_degreeRange_coversTheBlockOfADegree():
  fhat = np.arange(deg2dim(4))
  for n in range(4):
    assert fhat[degreeRange(n)].size == (2 * n + 1) ** 2


def test_degreeBlock_storesTheBlockColumnByColumn():
  fhat = np.arange(deg2dim(4))
  for n in range(4):
    for l in range(-n, n + 1):
      for k in range(-n, n + 1):
        assert degreeBlock(fhat, n)[k + n, l + n] == deg2dim(n) + (k + n) + (2 * n + 1) * (l + n)


def test_degreeBlock_isAView():
  fhat = np.zeros(deg2dim(3))
  degreeBlock(fhat, 2)[1, 3] = 7
  assert fhat[deg2dim(2) + 1 + 5 * 3] == 7


def test_degreeBlock_keepsBatchAxesInFront():
  batch = np.arange(2 * deg2dim(3)).reshape(2, -1)
  assert degreeBlock(batch, 2).shape == (2, 5, 5)
  assert np.array_equal(degreeBlock(batch, 2)[1], degreeBlock(batch[1], 2))


betas = (0.3, 1.2, 2.9, pi / 2)


@pytest.mark.parametrize("beta", betas)
def test_wignerd_isTheJacobiForm(beta):
  for n in range(13):
    ref = np.array([[wignerdJacobi(n, k, l, beta) for l in range(-n, n + 1)] for k in range(-n, n + 1)])
    assert close(wignerd(beta, n), ref)


@pytest.mark.parametrize("beta", betas)
def test_wignerd_isOrthogonal(beta):
  for n in range(13):
    d = wignerd(beta, n)
    assert close(d @ d.T, np.eye(2 * n + 1))


@pytest.mark.parametrize("beta", betas)
def test_wignerd_ofMinusBetaIsTheTranspose(beta):
  for n in range(13):
    assert close(wignerd(-beta, n), wignerd(beta, n).T)


@pytest.mark.parametrize("beta", betas)
def test_wignerd_reversedIsTheSignedMatrix(beta):
  for n in range(13):
    d = wignerd(beta, n)
    assert close(d[::-1, ::-1], np.array([[(-1.0) ** (k - l) for l in range(-n, n + 1)] for k in range(-n, n + 1)]) * d)


def test_wignerd_atZeroIsTheIdentity():
  assert close(wignerd(0.0, 5), np.eye(11))


def test_wignerd_ofDegreeOneIsTheClosedForm():
  c, s = np.cos(0.7), np.sin(0.7)
  assert close(wignerd(0.7, 1), [[(1 + c) / 2, s / np.sqrt(2), (1 - c) / 2], [-s / np.sqrt(2), c, s / np.sqrt(2)], [(1 - c) / 2, -s / np.sqrt(2), (1 + c) / 2]])


def test_wignerd_withABandwidthListsAllDegrees():
  d64 = wignerd(1.1, bandwidth=40)
  assert len(d64) == 41 and close(d64[7], wignerd(1.1, 7))


# single precision follows the recursion to its own accuracy
def test_wignerd_inSinglePrecisionFollowsTheDoubleRecursion():
  d64 = wignerd(1.1, bandwidth=40)
  d32 = wignerd(np.float32(1.1), bandwidth=40)
  assert d32[3].dtype == np.float32
  assert all(np.max(np.abs(d32[n] - d64[n])) < 2e-5 for n in range(41))


# the Fourier identity behind the transforms of F2
def test_wignerd_isAFourierSumOverTheMatricesAtHalfPi():
  beta = 0.83
  for n in range(7):
    d, h = wignerd(beta, n), wignerd(pi / 2, n)
    j = np.arange(-n, n + 1)
    for l in range(-n, n + 1):
      for k in range(-n, n + 1):
        s = np.sum(h[:, k + n] * h[:, l + n] * np.exp(-1j * j * beta))
        assert abs(1j ** ((k - l) % 4) * s - d[k + n, l + n]) < 1e-12


def test_wignerd_withoutADegreeRaises():
  with pytest.raises(TypeError):
    wignerd(0.3)


def randomWignerD():
  r1, r2 = rotation.rand(rng=rng), rotation.rand(rng=rng)
  return r1, r2, _WignerD(r1, bandwidth=4)


def test_WignerD_isAHomomorphism():
  for _ in range(10):
    r1, r2, D1 = randomWignerD()
    D2, D12 = _WignerD(r2, bandwidth=4), _WignerD(r1 * r2, bandwidth=4)
    for n in range(5):
      assert close(D12[n], D1[n] @ D2[n])


def test_WignerD_ofTheInverseIsTheAdjoint():
  for _ in range(10):
    r1, _r2, D1 = randomWignerD()
    Dinv = _WignerD(r1.inv(), bandwidth=4)
    for n in range(5):
      assert close(Dinv[n], D1[n].conj().T)


def test_WignerD_isUnitary():
  for _ in range(10):
    _r1, _r2, D1 = randomWignerD()
    for n in range(5):
      assert close(D1[n] @ D1[n].conj().T, np.eye(2 * n + 1))


def test_WignerD_ofTheIdentityIsTheIdentity():
  assert all(close(_WignerD(rotation.id, n), np.eye(2 * n + 1)) for n in range(4))


# the definition in the Matthies angles
def matthiesWignerD():
  d = wignerd(1.1, 3)
  D = np.array([[np.exp(-1j * k * 0.4) * d[k + 3, l + 3] * np.exp(-1j * l * 2.3) for l in range(-3, 4)] for k in range(-3, 4)])
  return rotation.byEuler(0.4, 1.1, 2.3, "Matthies"), d, D


def test_WignerD_isTheDefinitionInTheMatthiesAngles():
  r, _d, D = matthiesWignerD()
  assert close(_WignerD(r, 3), D)


# the mapping to MTEX's WignerD.m
def test_WignerD_transposedIsMTEXsWignerD():
  _r, d, D = matthiesWignerD()
  Dmtex = np.array([[np.exp(-1j * k * 2.3) * d[l + 3, k + 3] * np.exp(-1j * l * 0.4) for l in range(-3, 4)] for k in range(-3, 4)])
  assert close(Dmtex, D.T)


def test_WignerD_stacksTheMatricesOfAContainerInFront():
  rs = rotation.rand(2, 3, rng=rng)
  Ds = _WignerD(rs, 2)
  assert Ds.shape == (2, 3, 5, 5) and close(Ds[1, 2], _WignerD(rs[1, 2], 2))


def test_WignerD_inSinglePrecisionIsComplex64():
  r = rotation.byEuler(0.4, 1.1, 2.3, "Matthies")
  assert _WignerD(r, 3, dtype=np.float32).dtype == np.complex64


# MATLAB's matrices, the basis of the public coefficients: its run of WignerFunctions
def test_WignerD_publicAtNinetyDegreesIsMATLABs():
  d = WignerD(90 * degree, 1)
  assert close(d, [[0.5, -np.sqrt(0.5), -0.5], [np.sqrt(0.5), 0, np.sqrt(0.5)], [-0.5, -np.sqrt(0.5), 0.5]])


def test_WignerD_publicNormalizedIsMATLABs():
  R = rotation.byEuler(20 * degree, 40 * degree, 10 * degree)
  W = WignerD(R, 1, normalize=True)
  assert np.allclose(W[2, 0], 0.1995 + 0.0352j, atol=1e-4) and np.allclose(W[0, 0], 1.3245 + 0.7647j, atol=1e-4)


def test_SO3FunHarmonic_evalSumsTheCoefficientsAgainstThePublicWignerD():
  R = rotation.byEuler(20 * degree, 40 * degree, 10 * degree)
  for n in range(4):
    c = rng.standard_normal((2 * n + 1) ** 2) + 1j * rng.standard_normal((2 * n + 1) ** 2)
    fhat = np.concatenate([np.zeros(deg2dim(n)), c])
    # MATLAB stores a block column by column
    assert close(SO3FunHarmonic(fhat).eval(R), np.sum(c.reshape(2 * n + 1, 2 * n + 1, order="F") * WignerD(R, n, normalize=True)))


def test_sphericalY_ofLowDegreesIsTheClosedForm():
  for _ in range(5):
    v = vector3d.rand(rng=rng)
    theta, rho = v.theta, v.rho
    assert close(_sphericalY(0, v), [1 / np.sqrt(4 * pi)])
    assert close(_sphericalY(1, v), [np.sqrt(3 / 8 / pi) * np.sin(theta) * np.exp(-1j * rho), np.sqrt(3 / 4 / pi) * np.cos(theta), -np.sqrt(3 / 8 / pi) * np.sin(theta) * np.exp(1j * rho)])
    assert close(_sphericalY(2, v)[4], np.sqrt(15 / 32 / pi) * np.sin(theta) ** 2 * np.exp(2j * rho))


def test_sphericalY_ofTheNegativeOrderIsTheSignedConjugate():
  for _ in range(5):
    v = vector3d.rand(rng=rng)
    for l in range(6):
      Y = _sphericalY(l, v)
      assert all(close(Y[l - m], (-1) ** m * np.conj(Y[l + m])) for m in range(l + 1))


# orthonormality by a product rule exact to degree 12
def test_sphericalY_isOrthonormal():
  x, w = np.polynomial.legendre.leggauss(8)
  m = 16
  X, I = np.meshgrid(x, np.arange(m), indexing="ij")
  nodes = vector3d(np.sqrt(1 - X ** 2) * np.cos(2 * pi * I / m), np.sqrt(1 - X ** 2) * np.sin(2 * pi * I / m), X).flatten()
  weights = np.repeat(w, m) * 2 * pi / m
  Y = np.concatenate([_sphericalY(l, nodes) for l in range(6)])
  assert close(Y @ np.diag(weights) @ Y.conj().T, np.eye(36))


# rotation: Y_l(R xi) = conj(D_l(R)) Y_l(xi), and none of the other placements
def test_sphericalY_ofARotatedDirectionIsTheConjugateWignerDTimesY():
  for _ in range(5):
    r, v = rotation.rand(rng=rng), vector3d.rand(rng=rng)
    for l in range(1, 5):
      Y, D = _sphericalY(l, v), _WignerD(r, l)
      assert close(_sphericalY(l, r * v), np.conj(D) @ Y)
      assert close(_sphericalY(l, r.inv() * v), D.T @ Y)
      assert not close(_sphericalY(l, r * v), D @ Y) and not close(_sphericalY(l, r * v), D.conj().T @ Y)


def test_sphericalY_ofAFramedDirectionIsThatOfItsVector():
  cs = crystalFrame("m-3m")
  assert close(_sphericalY(2, Miller(1, 0, 0, cs)), _sphericalY(2, xvector))


def test_sphericalY_ofAContainerStacksTheOrders():
  assert _sphericalY(3, vector3d.rand(7, rng=rng)).shape == (7, 7)


def test_normalizedLegendre_isATableOverPointsAndOrders():
  assert normalizedLegendre(4, np.linspace(-1, 1, 5)).shape == (5, 5)


# MTEX's basis, the one of the public coefficients: MATLAB's values at x
def test_sphericalY_publicAtXIsMATLABs():
  assert close(sphericalY(1, xvector), [0.345494149471335, 0, 0.345494149471335])


# MTEX's basis: Y_l^{-m} = conj(Y_l^m)
def test_sphericalY_publicOfTheNegativeOrderIsTheConjugate():
  for l in range(5):
    Y = sphericalY(l, vector3d.rand(3, rng=rng))
    assert all(close(Y[l - m], np.conj(Y[l + m])) for m in range(l + 1))


def test_S2FunHarmonic_evalSumsTheCoefficientsAgainstThePublicSphericalY():
  for l in range(5):
    v = vector3d.rand(3, rng=rng)
    c = rng.standard_normal(2 * l + 1) + 1j * rng.standard_normal(2 * l + 1)
    assert close(S2FunHarmonic(np.concatenate([np.zeros(l * l), c])).eval(v), c @ sphericalY(l, v))


# coefficients of degree 3, two rotations, and fhat acted on by s from the left and the right
def seriesSetup():
  L = 3
  fhat = rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))
  r, s = rotation.rand(rng=rng), rotation.rand(rng=rng)
  left, right = fhat.copy(), fhat.copy()
  for n in range(L + 1):
    Dt = _WignerD(s, n).T
    degreeBlock(left, n)[...] = Dt @ degreeBlock(fhat, n)
    degreeBlock(right, n)[...] = degreeBlock(fhat, n) @ Dt
  return fhat, r, s, left, right


def test_wignerSeries_ofTheLeftActionIsTheSeriesAtSR():
  fhat, r, s, left, _right = seriesSetup()
  assert close(wignerSeries(left, r), wignerSeries(fhat, s * r))


def test_wignerSeries_ofTheRightActionIsTheSeriesAtRS():
  fhat, r, s, _left, right = seriesSetup()
  assert close(wignerSeries(right, r), wignerSeries(fhat, r * s))


def test_wignerSeries_ofAContainerIsElementwise():
  fhat = seriesSetup()[0]
  rs = rotation.rand(4, rng=rng)
  assert close(wignerSeries(fhat, rs), [wignerSeries(fhat, rs[i]) for i in range(4)])


def test_wignerSeries_ofABatchOfCoefficientsIsOnePerRow():
  fhat, _r, _s, left, _right = seriesSetup()
  rs = rotation.rand(4, rng=rng)
  assert close(wignerSeries(np.stack([fhat, left]), rs)[1], wignerSeries(left, rs))


# orthonormality of the basis under the normalised Haar measure
def test_WignerD_isOrthonormalUnderTheHaarMeasure():
  L = 3
  nodes, weights = so3Quadrature(2 * L)
  D = _WignerD(nodes, bandwidth=L)
  B = np.concatenate([np.sqrt(2 * n + 1) * D[n].swapaxes(-1, -2).reshape(nodes.size, -1).T for n in range(L + 1)])
  assert close(B @ np.diag(weights) @ B.conj().T, np.eye(deg2dim(L + 1)))
  assert close(np.sum(weights), 1)


groupNames = ("m-3m", "622", "-3m", "2/m", "-1", "4mm")


@pytest.mark.parametrize("name", groupNames + ("321",))
def test_WignerD_publicOfAGroupIsTheMeanOverItsRotations(name):
  pg = symmetry(name)
  for n in range(5):
    assert close(WignerD(pg, n), np.mean(WignerD(pg.properGroup().rotations().flatten(), n), axis=0))


@pytest.mark.parametrize("name", groupNames)
def test_WignerD_ofAGroupIsAnOrthogonalProjection(name):
  P = _WignerD(symmetry(name), bandwidth=3)
  for n in range(4):
    assert close(P[n] @ P[n], P[n])
    assert close(P[n], P[n].conj().T)


@pytest.mark.parametrize("name", groupNames)
def test_WignerD_ofAGroupProjectsOntoInvariantFunctions(name):
  L = 3
  fhat = rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))
  r = rotation.rand(rng=rng)
  pg = symmetry(name)
  P = _WignerD(pg, bandwidth=L)
  sym = fhat.copy()
  for n in range(L + 1):
    degreeBlock(sym, n)[...] = P[n] @ degreeBlock(fhat, n) @ P[n]
  for i in range(len(pg)):
    e = pg.rot[i].rot
    assert close(wignerSeries(sym, r * e), wignerSeries(sym, r))
    assert close(wignerSeries(sym, e * r), wignerSeries(sym, r))


# a frame delegates to its group, whose elements are written in the frame's axes
@pytest.mark.parametrize("name", groupNames)
def test_WignerD_ofAFrameIsThatOfItsGroup(name):
  assert close(_WignerD(crystalFrame(name), 2), _WignerD(crystalFrame(name).sym, 2))


def test_WignerD_ofAGroupWithoutRotationsIsTheIdentity():
  assert close(_WignerD(symmetry("1"), 2), np.eye(5))
  assert close(_WignerD(specimenFrame("-1"), 2), np.eye(5))


def test_WignerD_ofAGroupInSinglePrecisionIsComplex64():
  assert _WignerD(symmetry("622"), 2, dtype=np.float32).dtype == np.complex64


def test_clebschGordan_isTheTabulatedValue():
  assert close(clebschGordan(1, 1, 1, -1, 0, 0), 1 / np.sqrt(3))
  assert close(clebschGordan(1, 0, 1, 0, 2, 0), np.sqrt(2 / 3))
  assert close(clebschGordan(1, 1, 1, 0, 1, 1), 1 / np.sqrt(2))
  assert close(clebschGordan(1, 1, 1, 0, 2, 1), 1 / np.sqrt(2))
  assert close(clebschGordan(2, 2, 2, -2, 0, 0), 1 / np.sqrt(5))
  assert clebschGordan(1, 1, 1, 1, 1, 1) == 0


def test_wigner3j_isTheTabulatedValue():
  assert close(wigner3j(1, 1, 0, 1, -1, 0), 1 / np.sqrt(3))


def test_clebschGordanTensor_isOrthogonalInItsLastIndex():
  for l1, l2 in ((1, 1), (2, 1), (2, 2)):
    G = np.concatenate([clebschGordanTensor(l1, l2, J).reshape(-1, 2 * J + 1) for J in range(abs(l1 - l2), l1 + l2 + 1)], axis=1)
    assert close(G.T @ G, np.eye(G.shape[1]))


# the expansion of a product of Wigner-D matrices, check_clebschGordan.m
def test_clebschGordanTensor_expandsAProductOfWignerD():
  g = rotation.byEuler(-72 * degree, 88 * degree, 134 * degree, "Bunge")
  for l1, l2 in ((1, 1), (1, 2), (2, 1), (2, 2)):
    D1, D2 = _WignerD(g, l1), _WignerD(g, l2)
    ref = np.kron(D1, D2)
    got = np.zeros_like(ref)
    for J in range(abs(l1 - l2), l1 + l2 + 1):
      C = clebschGordanTensor(l1, l2, J).reshape(-1, 2 * J + 1)
      got += C @ _WignerD(g, J) @ C.T
    assert close(got, ref, 1e-10)
    assert np.linalg.norm(ref) > 1e-6


# the compiled projection over the nonzeros of the blocks equals the matrix products, for
# sparse (dihedral) and dense (cubic) groups, one side or both, on lists of coefficients
def test_sparse_projection_against_the_products():
  from mtex import crystalFrame, invariantProjection, setMTEXpref, specimenFrame
  from mtex.functions.harmonics import project
  rng = np.random.default_rng(8)
  for cs, ss, L in (("622", "222", 20), ("-3m", "2", 17), ("1", "mmm", 12), ("m-3m", "1", 9), ("321", "1", 0), ("222", "1", 1)):
    CS, SS = crystalFrame(cs), specimenFrame(ss) if ss != "1" else specimenFrame.default
    P, Q = invariantProjection(SS, L), invariantProjection(CS, L)
    f = rng.standard_normal((3, deg2dim(L + 1))) + 1j * rng.standard_normal((3, deg2dim(L + 1)))
    setMTEXpref("kernels", "numpy")
    expected = np.stack([project(r.copy(), P, Q) for r in f])
    setMTEXpref("kernels", "numba")
    assert np.max(np.abs(project(f.copy(), P, Q) - expected)) < 1e-13


# the compiled Wigner-d table at any angle equals the recursion, the ends 0 and pi included
def test_wignerd_table_against_the_recursion():
  pytest.importorskip("numba")
  from mtex.functions.harmonics import wignerdDegrees
  from mtex.functions.kernels import wignerdTable
  for beta in (0.0, 0.4, np.pi / 2, 2.7, np.pi):
    T = wignerdTable(beta, 25)
    for n, d in wignerdDegrees(np.float64(beta), 25):
      assert np.max(np.abs(T[deg2dim(n):deg2dim(n + 1)].reshape(2 * n + 1, 2 * n + 1) - d)) < 1e-13


# the degree of every coefficient, spherical and SO(3), against the block layout
def test_degreeVectors_followTheLayout():
  from mtex.functions.harmonics import _s2Degrees, _so3Degrees
  assert np.array_equal(_s2Degrees(3), np.floor(np.sqrt(np.arange(16))).astype(int))
  assert np.array_equal(_so3Degrees(3), np.concatenate([np.full((2 * n + 1) ** 2, n) for n in range(4)]))


# a value per degree lands on every coefficient of its degree
def test_perDegree_spreadsAValuePerDegree():
  from mtex.functions.harmonics import _s2PerDegree, _so3PerDegree
  a = np.array([2.0, 3.0, 5.0])
  assert np.array_equal(_s2PerDegree(a), [2, 3, 3, 3, 5, 5, 5, 5, 5])
  assert np.array_equal(_so3PerDegree(a)[[0, 1, 9, 10, 34]], [2, 3, 3, 5, 5]) and _so3PerDegree(a).size == deg2dim(3)


# the sign (-1)^(k-l) of a degree block
def test_checkerboard_isTheParityOfKMinusL():
  from mtex.functions.harmonics import _checkerboard
  k = np.arange(-2, 3)
  assert np.array_equal(_checkerboard(2), (-1.0) ** (k[:, None] - k[None, :]))


# the Clebsch-Gordan tensor only couples M = m1 + m2
def test_clebschGordanTensor_isTheCoefficientsAtEveryIndex():
  C = clebschGordanTensor(2, 1, 2)
  full = np.array([[[clebschGordan(2, m1, 1, m2, 2, M) for M in range(-2, 3)] for m2 in range(-1, 2)] for m1 in range(-2, 3)])
  assert np.array_equal(C, full)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("g", ["rotations", "beta", "frame"])
@pytest.mark.parametrize("normalize", [False, True])
def test_WignerD_returnsTheDtypeAskedFor(dtype, g, normalize):
  # every matrix, of one degree or of all up to a bandwidth, is of the type asked for
  from mtex import WignerD, crystalFrame, rotation
  x = {"rotations": rotation.rand(4, rng=np.random.default_rng(2)), "beta": 0.3, "frame": crystalFrame("432")}[g]
  kw = {} if g == "frame" else {"normalize": normalize}
  assert WignerD(x, 3, dtype=dtype, **kw).dtype == dtype
  assert all(D.dtype == dtype for D in WignerD(x, bandwidth=2, dtype=dtype, **kw))

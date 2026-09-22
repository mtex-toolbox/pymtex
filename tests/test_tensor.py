# the tensor type with its classes, the Voigt and Kelvin notation, the Einstein sum, the
# group action and the factories (docs/plans/functions.md P4c: T1 to T3 and T9 after the
# Julia tests; check_EinsteinSum, check_tensorFactories)
import numpy as np
import pytest

from mtex import (EinsteinSum, Kelvin, Voigt, checkSymmetry, colon, complianceTensor, crystalFrame, degree, directionalMagnitude,
                  dyad, eig, inv, mean, orientation, rotate, rotation, specimenFrame, stiffnessTensor, strainTensor, stressTensor,
                  symmetrise, tensor, trace, transformReferenceFrame, vector3d, xvector, zvector)

rng = np.random.default_rng(20)
CUBIC = np.array([[231.0, 134, 134, 0, 0, 0], [134, 231, 134, 0, 0, 0], [134, 134, 231, 0, 0, 0], [0, 0, 0, 116, 0, 0],
                  [0, 0, 0, 0, 116, 0], [0, 0, 0, 0, 0, 116]])


# the isotropic stiffness of the Lame parameters, the reference every modulus is checked against
def isotropicStiffness(lam, mu, fr=None):
  V = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0], [lam, lam + 2 * mu, lam, 0, 0, 0], [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]], dtype=float)
  return stiffnessTensor(V, fr)


def test_T1_construction_classes_and_notation():
  cs = crystalFrame("m-3m")
  C = stiffnessTensor(CUBIC, cs)
  assert C.rank == 4 and isinstance(C, stiffnessTensor) and C.frame is cs and C.CS is cs and C.unit == "GPa"
  assert np.allclose(Voigt(C), CUBIC)                      # the round trip through the notation
  assert C.M[0, 0, 1, 1] == 134 and C.M[1, 2, 1, 2] == 116   # and the pair index
  assert np.isclose(Kelvin(C)[3, 3], 2 * 116)               # the Kelvin factor is sqrt(2) on each side
  assert stiffnessTensor(Kelvin(C), cs, Kelvin=True).isclose(C)
  assert stiffnessTensor(C.M, cs).isclose(C) and C.shape == () and C.size == 1
  # the class carries the Voigt factors: a strain has 2 on its shear components, a stress none
  eps = strainTensor([[1.0, 0.2, 0], [0.2, 2, 0], [0, 0, 3]])
  assert np.allclose(Voigt(eps), [1, 2, 3, 0, 0, 0.4]) and eps.doubleConvention
  assert np.allclose(Voigt(stressTensor(eps.M)), [1, 2, 3, 0, 0, 0.2])
  assert eps.frame is specimenFrame.default and eps.isSymmetric and not eps.isSkewSymmetric
  # a compliance is a stiffness inverted, and the class follows with its convention
  S = inv(C)
  assert isinstance(S, complianceTensor) and S.doubleConvention and np.allclose(Voigt(S), np.linalg.inv(CUBIC))
  assert inv(S).isclose(C) and isinstance(inv(S), stiffnessTensor)
  assert np.allclose(Voigt(colon(C, S)), Voigt(stiffnessTensor.eye(cs)))
  # a plain tensor reads its rank off the shape or takes it as given
  assert tensor(np.ones((3, 3))).rank == 2 and tensor(np.ones((3, 3, 3))).rank == 3 and tensor(np.ones((5, 3, 3)), rank=2).shape == (5,)
  assert tensor(CUBIC, rank=4).rank == 4 and tensor(np.zeros((3, 6)), rank=3).rank == 3 and tensor(5.0, rank=0).rank == 0
  assert tensor(xvector).rank == 1 and tensor(rotation.rand(rng=rng)).rank == 2
  assert tensor(np.ones((2, 4, 3, 3)), rank=2).shape == (2, 4)
  # the plain tensor takes the double convention as a keyword or a flag
  P = np.array([[0, 0, 0, 0.17, 0, 0], [0, 0, 0, 0, 0.17, 0], [0, 0, 0, 0, 0, 5.17]]) * 1e-11
  P0 = tensor(P, rank=3, name="piezoelectric modulus")
  assert P0.rank == 3 and np.allclose(Voigt(P0), P) and np.isclose(P0.M[2, 0, 1], 5.17e-11) and P0.M[2, 0, 1] == P0.M[2, 1, 0]
  assert np.isclose(tensor(P, rank=3, doubleConvention=True).M[2, 0, 1], 5.17e-11 / 2) and tensor(P, "doubleConvention", rank=3).doubleConvention
  # what does not fit is rejected
  with pytest.raises(ValueError):
    tensor(np.ones((3, 4)), rank=2)
  with pytest.raises(ValueError):
    stiffnessTensor(np.ones((3, 3)), cs)
  with pytest.raises(ValueError):
    strainTensor(np.ones((3, 3)), cs) + strainTensor(np.ones((3, 3)), specimenFrame.default)
  assert "Voigt" in repr(C) and "231" in repr(C) and "rank: 2" in repr(eps)


def test_T2_the_Einstein_sum():
  # check_EinsteinSum: a contraction over two indices, on arrays that are not 3 x 3
  A, B = rng.random((3, 5, 4)), rng.random((4, 3, 5))
  ref = np.einsum("jkl,lik->ij", A, B)
  assert np.allclose(EinsteinSum(A, [2, -1, -2], B, [-2, 1, -1]).M, ref)
  assert np.allclose(EinsteinSum(A, [1, 2, -1], B, [-1, 3, 4]), np.einsum("abl,lcd->abcd", A, B))   # not 3 x 3: an array
  # the outer product of two vectors
  v, w = vector3d.rand(rng=rng), vector3d.rand(rng=rng)
  assert np.allclose(EinsteinSum(v, [1], w, [2]).M, np.outer(v.data, w.data))
  assert np.allclose(dyad(v, w).M, np.outer(v.data, w.data)) and dyad(v, w).rank == 2
  assert np.allclose(dyad(xvector, 2).M, np.outer(xvector.data, xvector.data))
  assert np.isclose(EinsteinSum(v, [-1], w, [-1]), v.data @ w.data)
  # a label repeated inside one operand is a trace over those indices
  T = rng.random((3, 3, 3, 3))
  assert np.isclose(EinsteinSum(T, [-1, -2, -1, -2]), np.einsum("klkl", T))
  x = vector3d.rand(rng=rng)
  assert np.isclose(EinsteinSum(T, [-1, -2, -3, -3], x, [-1], x, [-2]), np.einsum("klmm,k,l", T, x.data, x.data))
  # a tensor operand keeps its frame and loses its class: a contracted stiffness is none
  cs = crystalFrame("m-3m")
  C = isotropicStiffness(60, 40, cs)
  t = EinsteinSum(C, [1, 2, -1, -1])
  assert type(t) is tensor and t.frame is cs and t.rank == 2
  # the double contraction, MATLAB's colon, and the products
  eps = strainTensor(np.diag([1.0, 0, -1]), cs)
  sigma = C * eps
  assert isinstance(sigma, stressTensor) and sigma.isclose(colon(C, eps)) and np.allclose(sigma.M, np.diag([80.0, 0, -80]))
  assert np.isclose(colon(sigma, eps), 160) and np.isclose(EinsteinSum(C, [-1, -2, -3, -4], eps, [-1, -2], eps, [-3, -4]), 160)
  T1, T2 = tensor(rng.standard_normal((3, 3))), tensor(rng.standard_normal((3, 3)))
  assert np.allclose((T1 * T2).M, T1.M @ T2.M) and np.isclose(colon(T1, T2), trace(T1 * T2.T))
  assert np.allclose((T1 * xvector).data, T1.M[:, 0]) and np.isclose(T1.det(), np.linalg.det(T1.M))
  assert np.allclose((T1 + 2 * T2).M, T1.M + 2 * T2.M) and np.allclose((T1 - T2).M, T1.M - T2.M) and np.allclose((T1 / 2).M, T1.M / 2)
  # lists of tensors combine elementwise over their leading dimensions
  Ts = tensor(rng.standard_normal((4, 3, 3)))
  assert (Ts * T1).shape == (4,) and np.allclose((Ts * T1)[2].M, Ts[2].M @ T1.M)
  assert EinsteinSum(Ts, [1, -1], xvector, [-1]).shape == (4,)
  with pytest.raises(ValueError):
    EinsteinSum(A, [1, -1, -2], B, [-1, 2, -2])
  with pytest.raises(ValueError):
    C * C


def test_T3_the_group_action():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  C = isotropicStiffness(60, 40, cs)
  # an isotropic tensor is invariant under every rotation
  r = rotation.rand(rng=rng)
  assert rotate(C, r).isclose(C, atol=1e-9)
  # a cubic one under its own group only, and it says so
  Cc = stiffnessTensor(CUBIC, cs)
  assert checkSymmetry(Cc)
  assert all(rotate(Cc, e).isclose(Cc, atol=1e-9) for e in cs.rotations())
  assert not rotate(Cc, r).isclose(Cc, atol=1e-6)
  # a tensor turned by anything but an element of its group keeps no group
  assert rotate(Cc, r).frame is not cs and rotate(Cc, r).frame.numSym() == 1
  assert rotate(Cc, cs.rotations()[1]).frame is cs
  # symmetrise projects onto the invariant tensors, and what it returns is invariant
  t = tensor(rng.standard_normal((3, 3)), cs)
  assert checkSymmetry(symmetrise(t))
  assert np.allclose(symmetrise(t).M, trace(t) / 3 * np.eye(3))       # cubic: a multiple of delta
  # an orientation carries the tensor into the specimen frame, without its group
  o = orientation.rand(cs, specimenFrame("222"), rng=rng)
  Cr = o * Cc
  assert Cr.frame is ss and np.isclose(Cr.norm(), Cc.norm()) and rotate(Cc, o).isclose(Cr)
  assert (r * Cc).isclose(rotate(Cc, r))
  # a list of orientations gives a list of tensors, indexable and averaged
  os = orientation.rand(5, cs, rng=rng)
  Cs = os * Cc
  assert Cs.shape == (5,) and Cs[3].isclose(os[3] * Cc) and isinstance(Cs, stiffnessTensor)
  assert mean(Cs).shape == () and mean(Cs, weights=[1, 0, 0, 0, 0]).isclose(Cs[0])
  assert mean(Cs).isclose(stiffnessTensor(Cs.M.mean(axis=0), ss))
  # the same coefficients read in another frame
  a = crystalFrame("2/m", [9.585, 8.776, 5.26], np.array([90, 105.86, 90]) * degree, "X||a*", "Z||c")
  b = crystalFrame("2/m", [9.75, 8.89, 5.25], np.array([90, 106, 90]) * degree, "X||a*", "Z||c")
  V = np.array([[228.1, 78.8, 70.2, 0, 7.9, 0], [78.8, 181.1, 61.1, 0, 5.9, 0], [70.2, 61.1, 245.4, 0, 39.7, 0], [0, 0, 0, 78.9, 0, 6.4],
                [7.9, 5.9, 39.7, 0, 68.2, 0], [0, 0, 0, 6.4, 0, 78.1]])
  Cm = stiffnessTensor(V, a)
  D = transformReferenceFrame(Cm, b)
  assert D.frame is b and np.isclose(D.norm(), Cm.norm()) and transformReferenceFrame(D, a).isclose(Cm, atol=1e-9)
  with pytest.raises(ValueError):
    transformReferenceFrame(Cm, crystalFrame("-1", [1, 2, 3], np.array([40, 60, 80]) * degree))
  # the directional magnitude, at directions and as the function on the sphere
  assert np.isclose(directionalMagnitude(Cc, xvector), 231) and np.isclose(directionalMagnitude(Cc, vector3d(1, 1, 1)), 321)
  assert directionalMagnitude(Cc, vector3d.rand(4, rng=rng)).shape == (4,) and directionalMagnitude(Cs, zvector).shape == (5,)
  f = directionalMagnitude(Cc)
  assert f.bandwidth == 4 and f.antipodal and np.isclose(np.real(f(vector3d(1, 1, 1))), 321)


def test_T9_factories():
  # check_tensorFactories: rand gives a symmetric, positive definite tensor of the right rank
  cs = crystalFrame("m-3m")
  for K in (stiffnessTensor, complianceTensor):
    T = K.rand(cs, rng=rng)
    assert T.rank == 4 and isinstance(T, K) and T.frame is cs
    assert T.isSymmetric and np.all(T.eigvals() > 0)
    assert checkSymmetry(T)                              # and it carries the symmetry of its frame
  assert stressTensor.rand(rng=rng).rank == 2 and stressTensor.rand(rng=rng).isSymmetric
  assert tensor.rand(3, rng=rng).shape == (3,) and tensor.rand(2, 2, rank=4, rng=rng).shape == (2, 2)
  # the identity tensors, ones, zeros and the Levi-Civita symbol
  assert np.allclose(tensor.eye().M, np.eye(3)) and np.allclose(Voigt(stiffnessTensor.eye()), np.diag([1, 1, 1, 0.5, 0.5, 0.5]))
  assert np.allclose(Voigt(tensor.eye(rank=4)), np.diag([1, 1, 1, 0.5, 0.5, 0.5]))
  assert np.all(tensor.ones(rank=2).M == 1) and np.all(tensor.zeros(2, rank=1).M == 0) and tensor.zeros(2, rank=1).shape == (2,)
  L = tensor.leviCivita
  assert L.rank == 3 and L.M[0, 1, 2] == 1 and L.M[1, 0, 2] == -1 and L.M[0, 0, 1] == 0
  # the mean of a list, weighted or not, and the eigenvectors as axes of the frame
  ts = tensor(rng.standard_normal((5, 3, 3)), cs)
  assert mean(ts).isclose(tensor(ts.M.mean(axis=0), cs)) and mean(ts, weights=[1, 0, 0, 0, 0]).isclose(ts[0])
  vals, vecs = eig(stressTensor(np.diag([3.0, 1, 2])))
  assert np.allclose(vals, [1, 2, 3]) and vecs.antipodal and np.isclose(abs(vecs[2].data[0]), 1)
  vals4, _ = eig(stiffnessTensor(CUBIC, cs))
  assert np.allclose(np.sort(vals4)[-1], 231 + 2 * 134)
  # a uniaxial stress along a direction, in its frame
  s = stressTensor.uniaxial(zvector)
  assert isinstance(s, stressTensor) and np.allclose(s.M, np.diag([0, 0, 1.0]))
  assert np.allclose(stressTensor.uniaxial(vector3d(1, 1, 0), 2).M, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))

# the tensor type with its classes, the Voigt and Kelvin notation, the Einstein sum, the
# group action and the factories (docs/plans/functions.md P4c: T1 to T3 and T9 after the
# Julia tests; check_EinsteinSum, check_tensorFactories)
import numpy as np
import pytest
import scipy.linalg

from mtex import (EinsteinSum, Kelvin, Voigt, cat, checkSymmetry, colon, complianceTensor, crystalFrame, degree, directionalMagnitude,
                  dyad, eig, expm, inv, logm, mean, orientation, polar, rotate, rotation, setMTEXpref, specimenFrame, sqrtm, stiffnessTensor, strainTensor,
                  stressTensor, svd, symmetrise, tensor, times, trace, transformReferenceFrame, vector3d, xvector, zvector)

rng = np.random.default_rng(20)
CUBIC = np.array([[231.0, 134, 134, 0, 0, 0], [134, 231, 134, 0, 0, 0], [134, 134, 231, 0, 0, 0], [0, 0, 0, 116, 0, 0],
                  [0, 0, 0, 0, 116, 0], [0, 0, 0, 0, 0, 116]])


# the isotropic stiffness of the Lame parameters, the reference every modulus is checked against
def isotropicStiffness(lam, mu, fr=None):
  V = np.array([[lam + 2 * mu, lam, lam, 0, 0, 0], [lam, lam + 2 * mu, lam, 0, 0, 0], [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]], dtype=float)
  return stiffnessTensor(V, fr)


def test_stiffnessTensor_carriesItsRankClassFrameAndUnit():
  cs = crystalFrame("m-3m")
  C = stiffnessTensor(CUBIC, cs)
  assert C.rank == 4 and isinstance(C, stiffnessTensor) and C.frame is cs and C.CS is cs and C.unit == "GPa"
  assert stiffnessTensor(C.M, cs).isclose(C) and C.shape == () and C.size == 1


def test_Voigt_and_Kelvin_roundTripTheCoefficients():
  cs = crystalFrame("m-3m")
  C = stiffnessTensor(CUBIC, cs)
  assert np.allclose(Voigt(C), CUBIC)                      # the round trip through the notation
  assert C.M[0, 0, 1, 1] == 134 and C.M[1, 2, 1, 2] == 116   # and the pair index
  assert np.isclose(Kelvin(C)[3, 3], 2 * 116)               # the Kelvin factor is sqrt(2) on each side
  assert stiffnessTensor(Kelvin(C), cs, Kelvin=True).isclose(C)


def test_Voigt_appliesTheFactorsOfTheClass():
  # the class carries the Voigt factors: a strain has 2 on its shear components, a stress none
  eps = strainTensor([[1.0, 0.2, 0], [0.2, 2, 0], [0, 0, 3]])
  assert np.allclose(Voigt(eps), [1, 2, 3, 0, 0, 0.4]) and eps.doubleConvention
  assert np.allclose(Voigt(stressTensor(eps.M)), [1, 2, 3, 0, 0, 0.2])
  assert eps.frame is specimenFrame.default and eps.isSymmetric and not eps.isSkewSymmetric


def test_inv_ofAStiffnessIsACompliance():
  # a compliance is a stiffness inverted, and the class follows with its convention
  cs = crystalFrame("m-3m")
  C = stiffnessTensor(CUBIC, cs)
  S = inv(C)
  assert isinstance(S, complianceTensor) and S.doubleConvention and np.allclose(Voigt(S), np.linalg.inv(CUBIC))
  assert inv(S).isclose(C) and isinstance(inv(S), stiffnessTensor)
  assert np.allclose(Voigt(colon(C, S)), Voigt(stiffnessTensor.eye(cs)))


def test_inv_ofManyTensorsIsCompiledAsNumpysVoigtInverse():
  # the compiled loop over the tensors against NumPy's three passes, both ways of the convention
  cs = crystalFrame("mmm", [4.7646, 10.2296, 5.9942])
  A = rng.random((6, 6))
  R = orientation.rand(50, cs, rng=rng) * stiffnessTensor(10 * (A + A.T) + 200 * np.eye(6), cs)
  S, T = inv(R), inv(inv(R))
  setMTEXpref("kernels", "numpy")
  try:
    Sn, Tn = inv(R), inv(inv(R))
  finally:
    setMTEXpref("kernels", "numba")
  assert np.allclose(S.data, Sn.data, rtol=1e-12, atol=0) and np.allclose(T.data, Tn.data, rtol=1e-12, atol=1e-12)


def test_inv_ofManySingularTensorsRaisesAsOfOne():
  # a singular Voigt matrix is refused in a list as it is alone
  M = np.zeros((6, 6))
  M[:3, :3] = 1.0
  with pytest.raises(np.linalg.LinAlgError):
    inv(stiffnessTensor(np.stack([np.eye(6), M])))


def test_tensor_readsItsRankOffTheShapeOrTakesItAsGiven():
  # a plain tensor reads its rank off the shape or takes it as given
  assert tensor(np.ones((3, 3))).rank == 2 and tensor(np.ones((3, 3, 3))).rank == 3 and tensor(np.ones((5, 3, 3)), rank=2).shape == (5,)
  assert tensor(CUBIC, rank=4).rank == 4 and tensor(np.zeros((3, 6)), rank=3).rank == 3 and tensor(5.0, rank=0).rank == 0
  assert tensor(xvector).rank == 1 and tensor(rotation.rand(rng=rng)).rank == 2
  assert tensor(np.ones((2, 4, 3, 3)), rank=2).shape == (2, 4)


def test_tensor_takesTheDoubleConventionAsKeywordOrFlag():
  # the plain tensor takes the double convention as a keyword or a flag
  P = np.array([[0, 0, 0, 0.17, 0, 0], [0, 0, 0, 0, 0.17, 0], [0, 0, 0, 0, 0, 5.17]]) * 1e-11
  P0 = tensor(P, rank=3, name="piezoelectric modulus")
  assert P0.rank == 3 and np.allclose(Voigt(P0), P) and np.isclose(P0.M[2, 0, 1], 5.17e-11) and P0.M[2, 0, 1] == P0.M[2, 1, 0]
  assert np.isclose(tensor(P, rank=3, doubleConvention=True).M[2, 0, 1], 5.17e-11 / 2) and tensor(P, "doubleConvention", rank=3).doubleConvention


def test_tensor_rejectsWhatDoesNotFit():
  # what does not fit is rejected
  cs = crystalFrame("m-3m")
  with pytest.raises(ValueError):
    tensor(np.ones((3, 4)), rank=2)
  with pytest.raises(ValueError):
    stiffnessTensor(np.ones((3, 3)), cs)
  with pytest.raises(ValueError):
    strainTensor(np.ones((3, 3)), cs) + strainTensor(np.ones((3, 3)), specimenFrame.default)


def test_repr_namesTheNotationAndTheRank():
  C = stiffnessTensor(CUBIC, crystalFrame("m-3m"))
  eps = strainTensor([[1.0, 0.2, 0], [0.2, 2, 0], [0, 0, 3]])
  assert "Voigt" in repr(C) and "231" in repr(C) and "rank: 2" in repr(eps)


def test_EinsteinSum_contractsArraysThatAreNot3x3():
  # check_EinsteinSum: a contraction over two indices, on arrays that are not 3 x 3
  A, B = rng.random((3, 5, 4)), rng.random((4, 3, 5))
  ref = np.einsum("jkl,lik->ij", A, B)
  assert np.allclose(EinsteinSum(A, [2, -1, -2], B, [-2, 1, -1]).M, ref)
  assert np.allclose(EinsteinSum(A, [1, 2, -1], B, [-1, 3, 4]), np.einsum("abl,lcd->abcd", A, B))   # not 3 x 3: an array


def test_EinsteinSum_ofTwoVectorsIsTheOuterOrInnerProduct():
  # the outer product of two vectors
  v, w = vector3d.rand(rng=rng), vector3d.rand(rng=rng)
  assert np.allclose(EinsteinSum(v, [1], w, [2]).M, np.outer(v.data, w.data))
  assert np.allclose(dyad(v, w).M, np.outer(v.data, w.data)) and dyad(v, w).rank == 2
  assert np.allclose(dyad(xvector, 2).M, np.outer(xvector.data, xvector.data))
  assert np.isclose(EinsteinSum(v, [-1], w, [-1]), v.data @ w.data)


def test_EinsteinSum_traceOverALabelRepeatedInOneOperand():
  # a label repeated inside one operand is a trace over those indices
  T = rng.random((3, 3, 3, 3))
  assert np.isclose(EinsteinSum(T, [-1, -2, -1, -2]), np.einsum("klkl", T))
  x = vector3d.rand(rng=rng)
  assert np.isclose(EinsteinSum(T, [-1, -2, -3, -3], x, [-1], x, [-2]), np.einsum("klmm,k,l", T, x.data, x.data))


def test_EinsteinSum_keepsTheFrameAndDropsTheClass():
  # a tensor operand keeps its frame and loses its class: a contracted stiffness is none
  cs = crystalFrame("m-3m")
  C = isotropicStiffness(60, 40, cs)
  t = EinsteinSum(C, [1, 2, -1, -1])
  assert type(t) is tensor and t.frame is cs and t.rank == 2


def test_colon_isTheDoubleContraction():
  # the double contraction, MATLAB's colon, and the products
  cs = crystalFrame("m-3m")
  C = isotropicStiffness(60, 40, cs)
  eps = strainTensor(np.diag([1.0, 0, -1]), cs)
  sigma = C * eps
  assert isinstance(sigma, stressTensor) and sigma.isclose(colon(C, eps)) and np.allclose(sigma.M, np.diag([80.0, 0, -80]))
  assert np.isclose(colon(sigma, eps), 160) and np.isclose(EinsteinSum(C, [-1, -2, -3, -4], eps, [-1, -2], eps, [-3, -4]), 160)


def test_rankTwoArithmeticIsTheMatrixArithmetic():
  T1, T2 = tensor(rng.standard_normal((3, 3))), tensor(rng.standard_normal((3, 3)))
  assert np.allclose((T1 * T2).M, T1.M @ T2.M) and np.isclose(colon(T1, T2), trace(T1 * T2.T))
  assert np.allclose((T1 * xvector).data, T1.M[:, 0]) and np.isclose(T1.det(), np.linalg.det(T1.M))
  assert np.allclose((T1 + 2 * T2).M, T1.M + 2 * T2.M) and np.allclose((T1 - T2).M, T1.M - T2.M) and np.allclose((T1 / 2).M, T1.M / 2)


def test_listsOfTensorsCombineElementwise():
  # lists of tensors combine elementwise over their leading dimensions
  T1 = tensor(rng.standard_normal((3, 3)))
  Ts = tensor(rng.standard_normal((4, 3, 3)))
  assert (Ts * T1).shape == (4,) and np.allclose((Ts * T1)[2].M, Ts[2].M @ T1.M)
  assert EinsteinSum(Ts, [1, -1], xvector, [-1]).shape == (4,)


def test_EinsteinSum_refusesMismatchedContractions():
  A, B = rng.random((3, 5, 4)), rng.random((4, 3, 5))
  C = isotropicStiffness(60, 40, crystalFrame("m-3m"))
  with pytest.raises(ValueError):
    EinsteinSum(A, [1, -1, -2], B, [-1, 2, -2])
  with pytest.raises(ValueError):
    C * C


def test_EinsteinSum_refusesASummedIndexNotUsedTwice():
  # MATLAB's check: a summed index pairs exactly two indices
  v = tensor(np.array([1.0, 2.0, 3.0]), rank=1)
  with pytest.raises(ValueError, match="appears 1 times"):
    EinsteinSum(v, [-1])
  with pytest.raises(ValueError, match="appears 3 times"):
    EinsteinSum(v, [-1], v, [-1], v, [-1])
  assert EinsteinSum(v, [-1], v, [-1]) == 14.0


def test_rotate_leavesAnIsotropicTensorInvariant():
  # an isotropic tensor is invariant under every rotation
  C = isotropicStiffness(60, 40, crystalFrame("m-3m"))
  r = rotation.rand(rng=rng)
  assert rotate(C, r).isclose(C, atol=1e-9)


def test_rotate_leavesACubicTensorInvariantUnderItsGroupOnly():
  # a cubic one under its own group only, and it says so
  cs = crystalFrame("m-3m")
  r = rotation.rand(rng=rng)
  Cc = stiffnessTensor(CUBIC, cs)
  assert checkSymmetry(Cc)
  assert all(rotate(Cc, e).isclose(Cc, atol=1e-9) for e in cs.rotations())
  assert not rotate(Cc, r).isclose(Cc, atol=1e-6)


def test_rotate_byANonSymmetryKeepsNoGroup():
  # a tensor turned by anything but an element of its group keeps no group
  cs = crystalFrame("m-3m")
  r = rotation.rand(rng=rng)
  Cc = stiffnessTensor(CUBIC, cs)
  assert rotate(Cc, r).frame is not cs and rotate(Cc, r).frame.numSym() == 1
  assert rotate(Cc, cs.rotations()[1]).frame is cs


def test_symmetrise_projectsOntoTheInvariantTensors():
  # symmetrise projects onto the invariant tensors, and what it returns is invariant
  t = tensor(rng.standard_normal((3, 3)), crystalFrame("m-3m"))
  assert checkSymmetry(symmetrise(t))
  assert np.allclose(symmetrise(t).M, trace(t) / 3 * np.eye(3))       # cubic: a multiple of delta


def test_orientationTimesTensor_carriesItIntoTheSpecimenFrame():
  # an orientation carries the tensor into the specimen frame, without its group
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  Cc = stiffnessTensor(CUBIC, cs)
  r = rotation.rand(rng=rng)
  o = orientation.rand(cs, specimenFrame("222"), rng=rng)
  Cr = o * Cc
  assert Cr.frame is ss and np.isclose(Cr.norm(), Cc.norm()) and rotate(Cc, o).isclose(Cr)
  assert (r * Cc).isclose(rotate(Cc, r))


def test_orientationsTimesTensor_giveAnIndexableListAndItsMean():
  # a list of orientations gives a list of tensors, indexable and averaged
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  Cc = stiffnessTensor(CUBIC, cs)
  os = orientation.rand(5, cs, rng=rng)
  Cs = os * Cc
  assert Cs.shape == (5,) and Cs[3].isclose(os[3] * Cc) and isinstance(Cs, stiffnessTensor)
  assert mean(Cs).shape == () and mean(Cs, weights=[1, 0, 0, 0, 0]).isclose(Cs[0])
  assert mean(Cs).isclose(stiffnessTensor(Cs.M.mean(axis=0), ss))


def test_transformReferenceFrame_readsTheCoefficientsInAnotherFrame():
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


def test_directionalMagnitude_atDirectionsAndAsASphericalFunction():
  # the directional magnitude, at directions and as the function on the sphere
  cs = crystalFrame("m-3m")
  Cc = stiffnessTensor(CUBIC, cs)
  Cs = orientation.rand(5, cs, rng=rng) * Cc
  assert np.isclose(directionalMagnitude(Cc, xvector), 231) and np.isclose(directionalMagnitude(Cc, vector3d(1, 1, 1)), 321)
  assert directionalMagnitude(Cc, vector3d.rand(4, rng=rng)).shape == (4,) and directionalMagnitude(Cs, zvector).shape == (5,)
  f = directionalMagnitude(Cc)
  assert f.bandwidth == 4 and f.antipodal and np.isclose(np.real(f(vector3d(1, 1, 1))), 321)


@pytest.mark.parametrize("R", [1, 2, 3, 4, 5])
def test_directionalMagnitude_ofManyDirectionsIsTheCompiledPolynomial(R):
  # one tensor at many directions of any length runs the compiled polynomial, the NumPy contraction its oracle
  T, v = tensor(rng.standard_normal((3,) * R), rank=R), vector3d(3 * rng.standard_normal((500, 3)))
  compiled = directionalMagnitude(T, v)
  setMTEXpref("kernels", "numpy")
  try:
    oracle = directionalMagnitude(T, v)
  finally:
    setMTEXpref("kernels", "numba")
  assert np.allclose(compiled, oracle, rtol=0, atol=1e-13)


@pytest.mark.parametrize("K", [stiffnessTensor, complianceTensor])
def test_rand_givesASymmetricPositiveDefiniteTensorOfItsFrame(K):
  # check_tensorFactories: rand gives a symmetric, positive definite tensor of the right rank
  cs = crystalFrame("m-3m")
  T = K.rand(cs, rng=rng)
  assert T.rank == 4 and isinstance(T, K) and T.frame is cs
  assert T.isSymmetric and np.all(T.eigvals() > 0)
  assert checkSymmetry(T)                              # and it carries the symmetry of its frame


def test_rand_givesTheRankAndShapeAskedFor():
  assert stressTensor.rand(rng=rng).rank == 2 and stressTensor.rand(rng=rng).isSymmetric
  assert tensor.rand(3, rng=rng).shape == (3,) and tensor.rand(2, 2, rank=4, rng=rng).shape == (2, 2)


def test_eyeOnesZerosAndLeviCivita():
  # the identity tensors, ones, zeros and the Levi-Civita symbol
  assert np.allclose(tensor.eye().M, np.eye(3)) and np.allclose(Voigt(stiffnessTensor.eye()), np.diag([1, 1, 1, 0.5, 0.5, 0.5]))
  assert np.allclose(Voigt(tensor.eye(rank=4)), np.diag([1, 1, 1, 0.5, 0.5, 0.5]))
  assert np.all(tensor.ones(rank=2).M == 1) and np.all(tensor.zeros(2, rank=1).M == 0) and tensor.zeros(2, rank=1).shape == (2,)
  L = tensor.leviCivita
  assert L.rank == 3 and L.M[0, 1, 2] == 1 and L.M[1, 0, 2] == -1 and L.M[0, 0, 1] == 0


def test_eig_givesTheEigenvectorsAsAxes():
  # the eigenvectors as axes of the frame
  vals, vecs = eig(stressTensor(np.diag([3.0, 1, 2])))
  assert np.allclose(vals, [1, 2, 3]) and vecs.antipodal and np.isclose(abs(vecs[2].data[0]), 1)
  vals4, _ = eig(stiffnessTensor(CUBIC, crystalFrame("m-3m")))
  assert np.allclose(np.sort(vals4)[-1], 231 + 2 * 134)


def test_stressTensor_uniaxialAlongADirection():
  # a uniaxial stress along a direction, in its frame
  s = stressTensor.uniaxial(zvector)
  assert isinstance(s, stressTensor) and np.allclose(s.M, np.diag([0, 0, 1.0]))
  assert np.allclose(stressTensor.uniaxial(vector3d(1, 1, 0), 2).M, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]]))


def test_the_display_shows_the_coefficients_at_every_rank():
  # a single tensor shows its coefficients, the options it carries and the density to MATLAB's five digits
  t0 = repr(tensor(5, rank=0))
  assert "rank: 0" in t0 and t0.rstrip().endswith("5")
  assert repr(tensor([1, 2, 3], rank=1)).split()[-3:] == ["1", "2", "3"]
  d = tensor(np.array([[0, 0, 0, 0.1, 0, 0]] * 3), rank=3, doubleConvention=True, name="piezo")
  assert "doubleConvention: true" in repr(d) and "compact matrix form" in repr(d) and ": piezo" in repr(d) and "  name " in repr(d)
  assert "density: 3.3449" in repr(stiffnessTensor(np.eye(6), density=3.3448528380995057))


def test_times_multiplies_corresponding_coefficients():
  # MATLAB's .* keeps the rank and refuses tensors of different ranks
  A, B = tensor.rand(3, rank=2, rng=rng), tensor.rand(3, rank=2, rng=rng)
  assert times(A, B).rank == 2 and np.allclose(times(A, B).M, A.M * B.M)
  with pytest.raises(ValueError):
    times(A, tensor([1, 2, 3], rank=1))


def test_a_rank_one_tensor_is_a_vector():
  v = vector3d(tensor(xvector))
  assert isinstance(v, vector3d) and np.allclose(v.data, [1, 0, 0])


def test_Hookes_law_drops_the_density_and_the_unit():
  # as MATLAB's: the stress carries neither the density nor the unit of the stiffness
  C = stiffnessTensor(CUBIC, density=3.3)
  sigma = C * strainTensor(np.diag([1.0, 0, -1]))
  assert sigma.density is None and sigma.unit is None
  assert colon(C.inv(), sigma).unit is None


def test_the_inverse_inverts_the_unit():
  # MATLAB's tensor/inv: GPa to 1 / GPa and back, a compliance in 1/GPa by default
  C = stiffnessTensor(np.eye(6) * 100 + np.ones((6, 6)), crystalFrame("m-3m"))
  assert C.unit == "GPa" and C.inv().unit == "1 / GPa" and C.inv().inv().unit == "GPa"
  assert complianceTensor(np.eye(6)).unit == "1/GPa" and complianceTensor(np.eye(6)).inv().unit == "GPa"
  assert tensor(np.eye(3), unit="m/s").inv().unit == "s / m"


def test_a_weighted_list_mean_gives_MATLABs_three_outputs():
  # the list keeps one density per tensor, and the Reuss mean is the inverse of the mean inverse
  A = stiffnessTensor(np.eye(6) * 100 + np.ones((6, 6)), density=3.0)
  B = stiffnessTensor(np.eye(6) * 200 + 2 * np.ones((6, 6)), density=4.0)
  L = cat(A, B)
  assert np.allclose(L.density, [3, 4]) and cat(A, A).density == 3.0
  V, R, H = mean(L, 'all', weights=[0.25, 0.75])
  assert np.allclose(Voigt(V), 0.25 * Voigt(A) + 0.75 * Voigt(B))
  assert np.allclose(Voigt(R.inv()), 0.25 * Voigt(A.inv()) + 0.75 * Voigt(B.inv()))
  assert np.allclose(H.M, (V.M + R.M) / 2) and np.isclose(V.density, 3.75) and np.isclose(H.density, 3.75)
  assert np.allclose(mean(L, 'Hill', weights=[0.25, 0.75]).M, H.M)


def test_matrix_functions_of_rank_two_and_four():
  A = rng.normal(size=(4, 3, 3))
  T = tensor(A, rank=2)
  E = expm(T)
  assert np.allclose(E.M[1], scipy.linalg.expm(A[1]))
  assert np.allclose(expm(logm(E)).M, E.M)
  P = tensor(A @ np.swapaxes(A, -1, -2), rank=2)
  assert np.allclose((sqrtm(P) * sqrtm(P)).M, P.M)
  # a rank four tensor through its Kelvin matrix: the square root squares back by the double contraction
  C = stiffnessTensor(CUBIC, crystalFrame("m-3m"), name="C")
  Q = sqrtm(C)
  assert isinstance(Q, stiffnessTensor) and np.allclose(EinsteinSum(Q, [1, 2, -1, -2], Q, [-1, -2, 3, 4]).M, C.M)
  L = logm(C)
  assert L.name == "log C" and np.allclose(expm(L).M, C.M)
  with pytest.raises(ValueError):
    expm(tensor(np.ones(3), rank=1))


def test_svd_givesDescendingSingularValues():
  A = rng.normal(size=(5, 3, 3))
  T = tensor(A, rank=2)
  u, s, v = svd(T)
  assert s.shape == (5, 3) and np.all(np.diff(s, axis=-1) <= 0)
  assert np.allclose(np.einsum("nij,nkj->nki", A, v.data), u.data * s[..., None])


def test_polar_givesARotationAndTwoSymmetricStretches():
  A = rng.normal(size=(5, 3, 3))
  R, SL, SR = polar(tensor(A, rank=2))
  assert np.allclose((R * SR).M, A) and np.allclose((SL * R).M, A)
  assert SL.isSymmetric and SR.isSymmetric and np.allclose(R.M @ np.swapaxes(R.M, -1, -2), np.eye(3))


def test_svd_ofRankFourTurnsTheCoreBackIntoTheTensor():
  # a rank four tensor: the rotation of its mode singular vectors turns the core back into it
  C = stiffnessTensor(CUBIC, crystalFrame("m-3m"))
  r, core, V = svd(C)
  assert isinstance(r, rotation) and V is None
  assert np.allclose(tensor(core.M, rank=4).rotate(r.matrix()).M, C.M)


def test_the_compiled_turn_agrees_with_the_kronecker_products():
  # one tensor by many rotations, many by one, many by as many, rank one to four
  from mtex import getMTEXpref, setMTEXpref
  from mtex.tensors.tensor import _applyMatrix
  pytest.importorskip("numba")
  Q = rotation.rand(50, rng=rng).matrix()
  before = getMTEXpref("kernels", "numba")
  try:
    for rank in (1, 2, 3, 4):
      one, many = rng.normal(size=(3,) * rank), rng.normal(size=(50,) + (3,) * rank)
      out = {}
      for be in ("numpy", "numba"):
        setMTEXpref("kernels", be)
        out[be] = [_applyMatrix(one, Q, rank), _applyMatrix(many, Q[0], rank), _applyMatrix(many, Q, rank)]
      for a, b in zip(out["numpy"], out["numba"]):
        assert a.shape == b.shape == (50, 3 ** rank) and np.allclose(a, b, atol=1e-12)
  finally:
    setMTEXpref("kernels", before)


# one tensor contracted on every index with lists of vectors, a trace in it taken first, is the
# compiled multilinear form, with einsum's values whatever the order of the labels
@pytest.mark.parametrize("labels", [[-1, -2, -3, -4], [-3, -1, -4, -2], [-1, -2, -5, -5]])
def test_EinsteinSum_ofOneTensorWithVectorsIsEinsums(labels):
  import sys
  module = sys.modules["mtex.tensors.tensor"]
  T = tensor(rng.standard_normal((3, 3, 3, 3)), rank=4)
  a, b = vector3d.rand(200), vector3d.rand(200)
  ops = [a, [-1], b, [-2]] + ([a, [-3], b, [-4]] if -3 in labels else [])
  got = EinsteinSum(T, labels, *ops)
  old, module._multilinearSum = module._multilinearSum, lambda *x: None
  try:
    ref = EinsteinSum(T, labels, *ops)
  finally:
    module._multilinearSum = old
  assert np.allclose(got, ref, rtol=1e-13, atol=1e-13)

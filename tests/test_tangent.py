# the tangent spaces of SO(3): check_logReference, the pair carriage of
# check_SO3TangentVectors, and the identities of the Julia RotationTangentSpace page
import numpy as np
import pytest
from numpy import pi

from mtex import (SO3TangentVector, orientation, rotation, crystalFrame, specimenFrame, vector3d, Miller,
                  xvector, degree, angle, log, exp, left, right, spinTensor, geodesic, inv, norm, cat, dot,
                  project2FundamentalRegion)
from conftest import close


def rng():
  return np.random.default_rng(0)


# ------------------------------------------------------------------ construction and frames

# a left tangent vector of 321 and 222 at a reference
def trigonalLog():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  oRef = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  o = orientation.byEuler(14 * degree, 23 * degree, 27 * degree, cs, ss)
  return cs, ss, oRef, log(o, oRef)


def test_log_leftVectorLivesInTheSpecimenFrame():
  cs, ss, _, v = trigonalLog()
  assert isinstance(v, SO3TangentVector) and v.side == "left" and v.frame is ss and v.CS is cs and v.SS is ss


def test_right_vectorLivesInTheCrystalFrame():
  cs, ss, _, v = trigonalLog()
  vr = right(v)
  assert vr.side == "right" and vr.frame is cs and vr.CS is cs and vr.SS is ss and vr.dispStyle == "UVTW"


def test_left_andRight_areInverseAndIdempotent():
  _, _, _, v = trigonalLog()
  vr = right(v)
  assert left(vr) == v and right(vr) is vr and left(v) is v


def test_SO3TangentVector_rebuiltFromTheReferenceKeepsBothSymmetries():
  cs, ss, _, v = trigonalLog()
  r = SO3TangentVector(v, v.oriRef, v.side)
  assert r.CS is cs and r.SS is ss


def test_SO3TangentVector_frameCannotBeAssigned():
  cs, _, _, v = trigonalLog()
  with pytest.raises(AttributeError):
    v.frame = cs


def test_vector3d_ofATangentVectorIsFramedLikeIt():
  _, ss, _, v = trigonalLog()
  p = vector3d(v)
  assert type(p) is vector3d and p.frame is ss and close(p.data, v.data)


def test_SO3TangentVector_framedVectorHasToFitItsSide():
  cs, _, oRef, _ = trigonalLog()
  with pytest.raises(ValueError):
    SO3TangentVector(Miller(1, 0, 0, cs, "uvw"), oRef, "left")
  assert SO3TangentVector(Miller(1, 0, 0, cs, "uvw"), oRef, "right").frame is cs


def test_SO3TangentVector_needsAReferenceAndASide():
  _, _, oRef, _ = trigonalLog()
  with pytest.raises(TypeError):
    SO3TangentVector(vector3d(1, 2, 3))
  with pytest.raises(ValueError):
    SO3TangentVector(vector3d(1, 2, 3), oRef, "middle")


R20 = rotation.byAxisAngle(xvector, 20 * degree)


def tangent20():
  return SO3TangentVector(0.2 * vector3d(1, 2, 3), R20)


def test_spinTensor_ofBothSidesDescribesOneTangentMatrix():
  # spinTensor is the tensor class of P4; its matrix is M
  t, M = tangent20(), R20.matrix()
  assert np.max(np.abs(spinTensor(t).M @ M - M @ spinTensor(right(t)).M)) < 1e-12


def test_SO3TangentVector_fromASpinTensorMatrix():
  t = tangent20()
  assert close(vector3d(SO3TangentVector(spinTensor(t).M, R20)).data, t.data)


def test_SO3TangentVector_takesTheAntisymmetricPartOfAMatrix():
  # the antisymmetric part is taken before the axial vector is read
  t = tangent20()
  S = spinTensor(t).M + 1e-3 * np.eye(3)
  assert close(SO3TangentVector(S, R20).data, t.data)


def test_SO3TangentVector_sameNumbersOnTheOtherSideDiffer():
  t = tangent20()
  assert norm(SO3TangentVector(0.2 * vector3d(1, 2, 3), R20, "right") - t) > 0.1 * norm(t)
  assert close(norm(t + right(t)), 2 * norm(t))


def test_norm_andAngle_ofATangentVector():
  t = tangent20()
  assert close(vector3d(t).norm(), t.norm()) and close(angle(t), norm(t))


def test_exp_ofATangentAtARotationIsARotation():
  t = tangent20()
  assert isinstance(exp(t), rotation) and not isinstance(exp(t), orientation)


def test_log_invertsExp():
  t = tangent20()
  assert close(vector3d(log(exp(t), R20)).data, t.data)


# ------------------------------------------------------------------ check_logReference

def refLog(ori, ref, side):
  # what log(ori, ref, side) is supposed to do, spelled out
  o = rotation(ori.project2FundamentalRegion(ref))
  m = inv(rotation(ref)) * o if side == "right" else o * inv(rotation(ref))
  return vector3d(log(m))


# 100 orientations of 432, a reference per orientation, and two copies of it with an identity
# and a symmetry element put in
def arrayReferences():
  cs = crystalFrame("432")
  g = rng()
  ori = orientation.rand(100, cs, rng=g)
  ref = orientation.rand(100, cs, rng=g)
  ref2 = ref.copy()
  ref2[17] = orientation.id(cs)
  ref3 = ref.copy()
  ref3[23] = orientation(cs.properGroup().rotations()[4], cs)
  return cs, ori, (ref, ref2, ref3)


@pytest.mark.parametrize("side", ["right", "left"])
def test_log_respects_an_array_valued_reference(side):
  cs, ori, refs = arrayReferences()
  for r in refs:
    v = log(ori, r, side)
    assert v.oriRef.shape == (100,) and v.CS is cs
    assert np.max(norm(vector3d(v) - refLog(ori, r, side))) < 1e-10


@pytest.mark.parametrize("side", ["right", "left"])
def test_exp_invertsLogAtAnArrayValuedReference(side):
  cs, ori, refs = arrayReferences()
  for r in refs:
    v = log(ori, r, side)
    e = exp(v)
    assert isinstance(e, orientation) and e.CS is cs
    assert np.max(angle(e, ori)) < 1e-4
    assert np.max(angle(exp(vector3d(v), r, side), ori)) < 1e-4


@pytest.mark.parametrize("side", ["right", "left"])
def test_log_of_rotations_respects_the_reference(side):
  g = rng()
  rot = rotation.rand(100, rng=g)
  rotRef = rotation.rand(100, rng=g)
  rotRef[3] = rotation.id
  v = vector3d(log(rot, rotRef, side))
  m = inv(rotRef) * rot if side == "right" else rot * inv(rotRef)
  assert np.max(norm(v - vector3d(log(m, side)))) < 1e-10
  assert np.max(angle(exp(log(rot, rotRef, side)), rot)) < 1e-10


def test_log_ofRotationsWithoutAReferenceIsAtTheIdentity():
  v = log(rotation.rand(100, rng=rng()))
  assert isinstance(v, SO3TangentVector) and v.oriRef.ndim == 0 and v.frame is None


def test_log_without_reference_reduces_by_both_symmetries():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  oriSS = orientation.rand(100, cs, ss, rng=rng())
  v = vector3d(log(oriSS))
  vRef = vector3d(log(project2FundamentalRegion(oriSS), symmetry=False))
  assert np.max(norm(v - vRef)) < 1e-10
  assert log(oriSS).oriRef.CS is cs and log(oriSS).oriRef.SS is ss


# an orientation 1 radian from a reference of 321 and 222
def trigonalPair():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  oRef = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  o = oRef * orientation.byAxisAngle(Miller(1, 2, -3, 3, cs), 1, cs, cs)
  return cs, ss, oRef, o


@pytest.mark.parametrize("side", ["right", "left"])
def test_log_and_exp_carry_the_symmetry_pair(side):
  cs, ss, oRef, o = trigonalPair()
  v = log(o, oRef, side)
  assert v.oriRef.CS is cs and v.oriRef.SS is ss
  e = exp(v)
  assert e.CS is cs and e.SS is ss and angle(e, o) < 1e-10
  e = exp(vector3d(v), oRef, side)
  assert e.CS is cs and e.SS is ss and angle(e, o) < 1e-10


def test_log_atABareRotationTakesTheFramesOfTheOrientation():
  cs, ss, oRef, o = trigonalPair()
  v = log(o, rotation(oRef))
  assert v.oriRef.CS is cs and v.oriRef.SS is ss


@pytest.mark.parametrize("name", ["m-3m", "622", "2/m"])
def test_log_measures_the_misorientation(name):
  cs = crystalFrame(name)
  g = rng()
  symOps = cs.properGroup().rotations()
  ref = orientation.rand(50, cs, rng=g)
  ori = orientation(rotation.byAxisAngle(vector3d.rand(50, rng=g), 1 * degree) * rotation(ref), cs)
  for side in ("right", "left"):
    v = vector3d(log(ori, ref, side))
    assert np.max(np.abs(norm(v) - angle(ori, ref))) < 1e-10
    for s in symOps:
      oriK = orientation(rotation(ori) * s, cs)
      assert np.max(norm(vector3d(log(oriK, ref, side)) - v)) < 1e-10
      # a right tangent vector turns with the crystal frame, a left one may not move
      refK = orientation(rotation(ref) * s, cs)
      vK = vector3d(log(ori, refK, side))
      dev = np.max(norm(vK - v)) if side == "left" else np.max(np.abs(norm(vK) - norm(v)))
      assert dev < 1e-10
  # the representatives themselves are far apart
  s = symOps[len(symOps) // 2]
  assert np.max(angle(rotation(ori) * s, rotation(ref))) > 10 * degree


def test_noSymmetry_takes_the_representatives():
  cs = crystalFrame("432")
  ref = orientation.rand(20, cs, rng=rng())
  ori = orientation(rotation(ref) * cs.properGroup().rotations()[7], cs)
  assert np.max(norm(log(ori, ref))) < 1e-10
  assert np.min(norm(log(ori, ref, symmetry=False))) > 10 * degree


# ------------------------------------------------------------------ algebra and lists

# five left and five right tangent vectors at one reference of m-3m and 222
def atOneReference():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  g = rng()
  ref = orientation.rand(cs, ss, rng=g)
  t = SO3TangentVector(vector3d.rand(5, rng=g), ref)
  u = SO3TangentVector(vector3d.rand(5, rng=g), ref, "right")
  return cs, ss, ref, t, u, g


def test_SO3TangentVector_sumTakesTheSideOfTheFirst():
  _, _, ref, t, u, _ = atOneReference()
  s = t + u
  assert isinstance(s, SO3TangentVector) and s.side == "left" and s.oriRef is ref
  assert close(s.data, t.data + left(u).data) and close((t - u).data, t.data - left(u).data)
  assert close((u + t).data, u.data + right(t).data) and (u + t).side == "right"


def test_SO3TangentVector_scalingAndNegation():
  _, _, _, t, _, _ = atOneReference()
  assert close((2 * t).data, 2 * t.data) and close((t * 2).data, 2 * t.data) and close((t / 2).data, t.data / 2)
  assert (-t).side == "left" and close((-t).data, -t.data)


def test_dot_crossAndAngle_ofTangentVectorsOnOneSide():
  _, _, _, t, u, _ = atOneReference()
  assert close(dot(t, u), np.sum(t.data * left(u).data, axis=-1))
  assert isinstance(t.cross(u), SO3TangentVector) and close(t.cross(u).data, np.cross(t.data, left(u).data))
  assert close(angle(t, u), vector3d(t).angle(vector3d(left(u)), symmetry=False))


def test_SO3TangentVector_equalityAcrossSides():
  _, _, _, t, u, _ = atOneReference()
  assert np.all(t == t) and np.all(t == left(t)) and not np.any(t == u)
  assert np.all(left(u + t) == t + u)


def test_mean_andSum_ofTangentVectors():
  _, _, ref, t, _, _ = atOneReference()
  m = t.mean()
  assert isinstance(m, SO3TangentVector) and m.oriRef is ref and close(m.data, np.mean(t.data, axis=0))
  assert close(t.sum().data, np.sum(t.data, axis=0))


def test_SO3TangentVector_refusesOtherObjects():
  _, _, _, t, u, _ = atOneReference()
  with pytest.raises(TypeError):
    t + vector3d(1, 0, 0)
  with pytest.raises(TypeError):
    vector3d(1, 0, 0) + t
  with pytest.raises(TypeError):
    t * u


def test_SO3TangentVector_refusesOtherReferences():
  cs, ss, ref, t, _, g = atOneReference()
  with pytest.raises(ValueError):
    t + SO3TangentVector(vector3d.rand(5, rng=g), orientation.rand(cs, ss, rng=g))
  with pytest.raises(ValueError):
    t + SO3TangentVector(t, orientation(rotation(ref), crystalFrame("432"), ss))


def test_SO3TangentVector_whatIsNoLongerATangentIsAFramedVector():
  _, ss, ref, t, _, _ = atOneReference()
  p = ref * right(t)
  assert type(p) is vector3d and p.frame is ss and close(p.data, (rotation(ref) * vector3d(right(t))).data)
  assert type(vector3d(t).normalize()) is vector3d and isinstance(t.normalize(), SO3TangentVector)


# four tangent vectors each at its own reference, and four at one shared reference
def listsOfTangents():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  g = rng()
  ref = orientation.rand(4, cs, ss, rng=g)
  ori = orientation.rand(4, cs, ss, rng=g)
  v = log(ori, ref)
  t = SO3TangentVector(vector3d.rand(4, rng=g), ref[0])
  return cs, ss, ref, v, t, g


def test_SO3TangentVector_indexingTakesThePerElementReference():
  _, _, ref, v, _, _ = listsOfTangents()
  assert v.oriRef.shape == (4,) and v[1].oriRef.ndim == 0 and angle(v[1].oriRef, ref[1]) < 1e-12
  assert close(v[1].data, v.data[1]) and v[1:3].oriRef.shape == (2,)
  assert v.reshape(2, 2).oriRef.shape == (2, 2) and v.reshape(2, 2).T[1, 0].oriRef == ref[1]


def test_SO3TangentVector_sharedReferenceStaysOneUnderIndexing():
  _, _, _, _, t, _ = listsOfTangents()
  assert t.oriRef.ndim == 0 and t[2].oriRef is t.oriRef


def test_SO3TangentVector_refusesReferencesOfAnotherShape():
  _, _, ref, _, _, g = listsOfTangents()
  with pytest.raises(ValueError):
    SO3TangentVector(vector3d.rand(4, rng=g), ref[:3])


def test_cat_keepsTheSymmetriesAndTheReferences():
  cs, ss, ref, v, t, _ = listsOfTangents()
  w = cat(v[0], v[1])
  assert w.oriRef.CS is cs and w.oriRef.SS is ss and w.oriRef.shape == (2,) and np.all(angle(w.oriRef, ref[:2]) < 1e-12)
  assert cat(t[0], t[1]).oriRef.ndim == 0
  w = cat(v[:2], right(v[2:]))
  assert w.side == "left" and close(w.data, v.data)


def test_mean_needsOneReference():
  # the mean over different references refuses, over one it works
  _, _, ref, v, t, _ = listsOfTangents()
  with pytest.raises(ValueError):
    v.mean()
  assert t.mean().oriRef is ref[0] or angle(t.mean().oriRef, ref[0]) < 1e-12


def test_SO3TangentVector_elementsCanBeStored():
  _, _, _, _, t, _ = listsOfTangents()
  t[0] = t[1]
  assert close(t.data[0], t.data[1])
  assert isinstance(t.unique(), SO3TangentVector) and t.unique().size == 3


# three left tangent vectors at a reference, and a rotation
def turnable():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  g = rng()
  ref = orientation.rand(cs, ss, rng=g)
  t = SO3TangentVector(vector3d.rand(3, rng=g), ref)
  return ref, t, rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree)


def test_rotation_ofTheSpecimenMovesALeftVectorWithItsReference():
  ref, t, r = turnable()
  s = r * t
  assert isinstance(s, SO3TangentVector) and s.side == "left"
  assert np.all(angle(exp(s), r * exp(t)) < 1e-10) and angle(s.oriRef, r * ref) < 1e-12
  assert isinstance(t.rotate(r), SO3TangentVector)


def test_rotation_ofTheCrystalMovesARightVectorWithItsReference():
  ref, t, r = turnable()
  u = right(t) * r
  assert isinstance(u, SO3TangentVector) and u.side == "right"
  assert np.all(angle(exp(u), exp(t) * r) < 1e-10) and angle(u.oriRef, ref * r) < 1e-12


def test_rotation_onTheWrongSideRefuses():
  _, t, r = turnable()
  with pytest.raises(TypeError):
    r * right(t)
  with pytest.raises(TypeError):
    t * r


# ------------------------------------------------------------------ exp

v90 = vector3d(0, 0, pi / 2)


def crystalTangent(CS):
  return Miller(1, 1, 1, CS, "uvw").normalize() * (30 * degree)


def test_exp_ofAPlainVectorIsARotation():
  assert exp(v90) == rotation.byAxisAngle(vector3d.Z, pi / 2) and type(exp(v90)) is rotation


def test_exp_ofACrystalDirectionIsAMisorientation():
  # a crystal direction is a misorientation of its frame with itself
  cs = crystalFrame("432")
  e = exp(crystalTangent(cs))
  assert isinstance(e, orientation) and e.CS is cs and e.SS is cs and close(e.angle(), 30 * degree)


def test_exp_atAReferenceUpdatesACrystalDirectionFromTheRightAndASpecimenVectorFromTheLeft():
  cs = crystalFrame("432")
  m = crystalTangent(cs)
  ref = orientation.rand(cs, rng=rng())
  assert angle(exp(m, ref), ref * exp(m)) < 1e-10
  assert angle(exp(v90, ref), rotation(exp(v90)) * ref) < 1e-10
  assert angle(exp(vector3d(v90.data), ref, "right"), ref * rotation(exp(v90))) < 1e-10
  assert angle(ref.exp(m), exp(m, ref)) < 1e-10


def test_exp_keepsTheFramesOfTheReference():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  m = crystalTangent(cs)
  ref = orientation.rand(cs, ss, rng=rng())
  assert exp(v90, ref).SS is ss and exp(v90, ref).CS is cs
  assert exp(m, ref).SS is ss and exp(m, ref).CS is cs


def test_exp_refusesACrystalDirectionOnTheLeft():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  ref = orientation.rand(cs, ss, rng=rng())
  with pytest.raises(ValueError):
    exp(crystalTangent(cs), ref, "left")


def test_exp_of_a_tangent_vector_at_its_own_reference():
  # ori.exp(v) with v an SO3TangentVector at ori, as the texture evolution steps write it
  cs = crystalFrame("432")
  o = orientation.rand(3, cs, rng=np.random.default_rng(5))
  t = SO3TangentVector(vector3d(np.full((3, 3), 0.01)), o)
  assert np.allclose(o.exp(-t).data, (-t).exp().data)


# ------------------------------------------------------------------ geodesic

def geodesicPath():
  cs = crystalFrame("432")
  oA = orientation.byEuler(20 * degree, 30 * degree, 10 * degree, cs)
  oB = orientation.byEuler(120 * degree, 60 * degree, 40 * degree, cs)
  return cs, oA, oB, geodesic(oA, oB, np.linspace(0, 1, 121))


def test_geodesic_runsFromEndToEnd():
  cs, oA, oB, path = geodesicPath()
  assert isinstance(path, orientation) and path.shape == (121,) and path.CS is cs
  assert angle(path[0], oA) < 1e-10 and angle(path[-1], oB) < 1e-10


def test_geodesic_runs_to_the_nearest_equivalent():
  _, oA, oB, path = geodesicPath()
  steps = np.sum(angle(path[:-1], path[1:]))
  assert abs(steps - angle(oA, oB)) < 1e-8
  assert angle(rotation(path[-1]), rotation(oB)) > 10 * degree


def test_geodesic_halfwayIsHalfTheAngle():
  _, oA, oB, _ = geodesicPath()
  assert angle(oA.geodesic(oB, 0.5), oA) - angle(oA, oB) / 2 < 1e-10


def test_geodesic_ofRotations():
  g = rng()
  rA, rB = rotation.rand(rng=g), rotation.rand(rng=g)
  assert angle(geodesic(rA, rB, 1), rB) < 1e-10 and angle(geodesic(rA, rB, 0), rA) < 1e-10


def test_geodesic_NaNEndpointYieldsTheOther():
  g = rng()
  rA, rB = rotation.rand(rng=g), rotation.rand(rng=g)
  assert angle(geodesic(rotation.nan(), rB, 0.3), rB) < 1e-10 and angle(geodesic(rA, rotation.nan(), 0.3), rA) < 1e-10


def tangentList():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  g = rng()
  ref = orientation.rand(cs, ss, rng=g)
  return log(orientation.rand(3, cs, ss, rng=g), ref)


def test_SO3TangentVector_repr():
  s = repr(tangentList())
  assert s.startswith("SO3TangentVector (left") and "reference: orientation" in s and "size     : 3" in s


def test_SO3TangentVector_reprOfARightVectorNamesTheCrystalAxes():
  # right components are x, y, z of the crystal frame, as MATLAB prints them
  t = tangentList()
  assert "reference: orientation" in repr(right(t)) and repr(right(t)).split("\n")[3].split() == ["x", "y", "z"]


def test_SO3TangentVector_reprAtARotation():
  assert "reference: rotation" in repr(log(rotation.rand(rng=rng())))

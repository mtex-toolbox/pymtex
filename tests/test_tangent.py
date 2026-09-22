# the tangent spaces of SO(3): check_logReference, the pair carriage of
# check_SO3TangentVectors, and the identities of the Julia RotationTangentSpace page
import numpy as np
import pytest
from numpy import pi

from mtex import (SO3TangentVector, orientation, rotation, crystalFrame, specimenFrame, vector3d, Miller,
                  xvector, degree, angle, log, exp, left, right, spinTensor, geodesic, inv, norm, cat, dot,
                  project2FundamentalRegion)
from conftest import close

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ construction and frames

def test_a_tangent_vector_reads_its_frame_from_the_reference():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  oRef = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  o = orientation.byEuler(14 * degree, 23 * degree, 27 * degree, cs, ss)
  v = log(o, oRef)
  assert isinstance(v, SO3TangentVector) and v.side == "left" and v.frame is ss and v.CS is cs and v.SS is ss
  vr = right(v)
  assert vr.side == "right" and vr.frame is cs and vr.CS is cs and vr.SS is ss and vr.dispStyle == "UVTW"
  assert left(vr) == v and right(vr) is vr and left(v) is v
  # rebuilding from the reference keeps both symmetries; the frame cannot be assigned
  r = SO3TangentVector(v, v.oriRef, v.side)
  assert r.CS is cs and r.SS is ss
  with pytest.raises(AttributeError):
    v.frame = cs
  # the plain vector is framed like the tangent vector
  p = vector3d(v)
  assert type(p) is vector3d and p.frame is ss and close(p.data, v.data)
  # a framed vector has to fit the side it is put on
  with pytest.raises(ValueError):
    SO3TangentVector(Miller(1, 0, 0, cs, "uvw"), oRef, "left")
  assert SO3TangentVector(Miller(1, 0, 0, cs, "uvw"), oRef, "right").frame is cs
  with pytest.raises(TypeError):
    SO3TangentVector(vector3d(1, 2, 3))
  with pytest.raises(ValueError):
    SO3TangentVector(vector3d(1, 2, 3), oRef, "middle")


def test_the_two_sides_describe_one_tangent_matrix():
  R = rotation.byAxisAngle(xvector, 20 * degree)
  t = SO3TangentVector(0.2 * vector3d(1, 2, 3), R)
  M = R.matrix()
  # spinTensor is the tensor class of P4; its matrix is M
  assert np.max(np.abs(spinTensor(t).M @ M - M @ spinTensor(right(t)).M)) < 1e-12
  assert close(vector3d(SO3TangentVector(spinTensor(t).M, R)).data, t.data)
  # the antisymmetric part is taken before the axial vector is read
  S = spinTensor(t).M + 1e-3 * np.eye(3)
  assert close(SO3TangentVector(S, R).data, t.data)
  # the same numbers on the other side are a different tangent vector
  assert norm(SO3TangentVector(0.2 * vector3d(1, 2, 3), R, "right") - t) > 0.1 * norm(t)
  assert close(norm(t + right(t)), 2 * norm(t))
  assert close(vector3d(t).norm(), t.norm()) and close(angle(t), norm(t))
  assert isinstance(exp(t), rotation) and not isinstance(exp(t), orientation)
  assert close(vector3d(log(exp(t), R)).data, t.data)


# ------------------------------------------------------------------ check_logReference

def refLog(ori, ref, side):
  # what log(ori, ref, side) is supposed to do, spelled out
  o = rotation(ori.project2FundamentalRegion(ref))
  m = inv(rotation(ref)) * o if side == "right" else o * inv(rotation(ref))
  return vector3d(log(m))


def test_log_respects_an_array_valued_reference():
  cs = crystalFrame("432")
  N = 100
  ori = orientation.rand(N, cs, rng=rng)
  ref = orientation.rand(N, cs, rng=rng)
  ref2 = ref.copy()
  ref2[17] = orientation.id(cs)
  ref3 = ref.copy()
  ref3[23] = orientation(cs.properGroup().rotations()[4], cs)
  for side in ("right", "left"):
    for r in (ref, ref2, ref3):
      v = log(ori, r, side)
      assert v.oriRef.shape == (N,) and v.CS is cs
      assert np.max(norm(vector3d(v) - refLog(ori, r, side))) < 1e-10
      e = exp(v)
      assert isinstance(e, orientation) and e.CS is cs
      assert np.max(angle(e, ori)) < 1e-4
      assert np.max(angle(exp(vector3d(v), r, side), ori)) < 1e-4


def test_log_of_rotations_respects_the_reference():
  N = 100
  rot = rotation.rand(N, rng=rng)
  rotRef = rotation.rand(N, rng=rng)
  rotRef[3] = rotation.id
  for side in ("right", "left"):
    v = vector3d(log(rot, rotRef, side))
    m = inv(rotRef) * rot if side == "right" else rot * inv(rotRef)
    assert np.max(norm(v - vector3d(log(m, side)))) < 1e-10
    assert np.max(angle(exp(log(rot, rotRef, side)), rot)) < 1e-10
  # without a reference the logarithm is taken at the identity
  v = log(rot)
  assert isinstance(v, SO3TangentVector) and v.oriRef.ndim == 0 and v.frame is None
  assert close(v.norm(), rot.angle()) and np.max(angle(exp(v), rot)) < 1e-10


def test_log_without_reference_reduces_by_both_symmetries():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  oriSS = orientation.rand(100, cs, ss, rng=rng)
  v = vector3d(log(oriSS))
  vRef = vector3d(log(project2FundamentalRegion(oriSS), noSymmetry=True))
  assert np.max(norm(v - vRef)) < 1e-10
  assert log(oriSS).oriRef.CS is cs and log(oriSS).oriRef.SS is ss


def test_log_and_exp_carry_the_symmetry_pair():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  oRef = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, cs, ss)
  o = oRef * orientation.byAxisAngle(Miller(1, 2, -3, 3, cs), 1, cs, cs)
  for side in ("right", "left"):
    v = log(o, oRef, side)
    assert v.oriRef.CS is cs and v.oriRef.SS is ss
    e = exp(v)
    assert e.CS is cs and e.SS is ss and angle(e, o) < 1e-10
    e = exp(vector3d(v), oRef, side)
    assert e.CS is cs and e.SS is ss and angle(e, o) < 1e-10
  # a bare rotation as the reference takes the frames of the orientation
  v = log(o, rotation(oRef))
  assert v.oriRef.CS is cs and v.oriRef.SS is ss


@pytest.mark.parametrize("name", ["m-3m", "622", "2/m"])
def test_log_measures_the_misorientation(name):
  cs = crystalFrame(name)
  symOps = cs.properGroup().rotations()
  ref = orientation.rand(50, cs, rng=rng)
  ori = orientation(rotation.byAxisAngle(vector3d.rand(50, rng=rng), 1 * degree) * rotation(ref), cs)
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
  ref = orientation.rand(20, cs, rng=rng)
  ori = orientation(rotation(ref) * cs.properGroup().rotations()[7], cs)
  assert np.max(norm(log(ori, ref))) < 1e-10
  assert np.min(norm(log(ori, ref, noSymmetry=True))) > 10 * degree


# ------------------------------------------------------------------ algebra and lists

def test_arithmetic_at_one_reference():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  ref = orientation.rand(cs, ss, rng=rng)
  t = SO3TangentVector(vector3d.rand(5, rng=rng), ref)
  u = SO3TangentVector(vector3d.rand(5, rng=rng), ref, "right")
  s = t + u
  assert isinstance(s, SO3TangentVector) and s.side == "left" and s.oriRef is ref
  assert close(s.data, t.data + left(u).data) and close((t - u).data, t.data - left(u).data)
  assert close((u + t).data, u.data + right(t).data) and (u + t).side == "right"
  assert close((2 * t).data, 2 * t.data) and close((t * 2).data, 2 * t.data) and close((t / 2).data, t.data / 2)
  assert (-t).side == "left" and close((-t).data, -t.data)
  assert close(dot(t, u), np.sum(t.data * left(u).data, axis=-1))
  assert isinstance(t.cross(u), SO3TangentVector) and close(t.cross(u).data, np.cross(t.data, left(u).data))
  assert close(angle(t, u), vector3d(t).angle(vector3d(left(u)), noSymmetry=True))
  assert np.all(t == t) and np.all(t == left(t)) and not np.any(t == u)
  assert np.all(left(u + t) == t + u)
  m = t.mean()
  assert isinstance(m, SO3TangentVector) and m.oriRef is ref and close(m.data, np.mean(t.data, axis=0))
  assert close(t.sum().data, np.sum(t.data, axis=0))
  # other objects and other references refuse
  with pytest.raises(TypeError):
    t + vector3d(1, 0, 0)
  with pytest.raises(TypeError):
    vector3d(1, 0, 0) + t
  with pytest.raises(TypeError):
    t * u
  with pytest.raises(ValueError):
    t + SO3TangentVector(vector3d.rand(5, rng=rng), orientation.rand(cs, ss, rng=rng))
  with pytest.raises(ValueError):
    t + SO3TangentVector(t, orientation(rotation(ref), crystalFrame("432"), ss))
  # what is no longer a tangent vector is a plain framed vector
  p = ref * right(t)
  assert type(p) is vector3d and p.frame is ss and close(p.data, (rotation(ref) * vector3d(right(t))).data)
  assert type(vector3d(t).normalize()) is vector3d and isinstance(t.normalize(), SO3TangentVector)


def test_lists_carry_one_reference_or_one_per_element():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  ref = orientation.rand(4, cs, ss, rng=rng)
  ori = orientation.rand(4, cs, ss, rng=rng)
  v = log(ori, ref)
  assert v.oriRef.shape == (4,) and v[1].oriRef.ndim == 0 and angle(v[1].oriRef, ref[1]) < 1e-12
  assert close(v[1].data, v.data[1]) and v[1:3].oriRef.shape == (2,)
  assert v.reshape(2, 2).oriRef.shape == (2, 2) and v.reshape(2, 2).T[1, 0].oriRef == ref[1]
  # a shared reference is broadcast, and it stays one under indexing
  t = SO3TangentVector(vector3d.rand(4, rng=rng), ref[0])
  assert t.oriRef.ndim == 0 and t[2].oriRef is t.oriRef
  with pytest.raises(ValueError):
    SO3TangentVector(vector3d.rand(4, rng=rng), ref[:3])
  # cat keeps both symmetries and the references
  w = cat(v[0], v[1])
  assert w.oriRef.CS is cs and w.oriRef.SS is ss and w.oriRef.shape == (2,) and np.all(angle(w.oriRef, ref[:2]) < 1e-12)
  assert cat(t[0], t[1]).oriRef.ndim == 0
  w = cat(v[:2], right(v[2:]))
  assert w.side == "left" and close(w.data, v.data)
  # the mean over different references refuses, over one it works
  with pytest.raises(ValueError):
    v.mean()
  assert t.mean().oriRef is ref[0] or angle(t.mean().oriRef, ref[0]) < 1e-12
  # elements can be stored
  t[0] = t[1]
  assert close(t.data[0], t.data[1])
  assert isinstance(t.unique(), SO3TangentVector) and t.unique().size == 3


def test_rotations_move_the_reference():
  cs, ss = crystalFrame("432"), specimenFrame("222")
  ref = orientation.rand(cs, ss, rng=rng)
  t = SO3TangentVector(vector3d.rand(3, rng=rng), ref)
  r = rotation.byAxisAngle(vector3d(1, 2, 3), 37 * degree)
  # a rotation of the specimen moves a left vector with its reference
  s = r * t
  assert isinstance(s, SO3TangentVector) and s.side == "left"
  assert np.all(angle(exp(s), r * exp(t)) < 1e-10) and angle(s.oriRef, r * ref) < 1e-12
  assert isinstance(t.rotate(r), SO3TangentVector)
  # a rotation of the crystal moves a right vector with its reference
  u = right(t) * r
  assert isinstance(u, SO3TangentVector) and u.side == "right"
  assert np.all(angle(exp(u), exp(t) * r) < 1e-10) and angle(u.oriRef, ref * r) < 1e-12
  with pytest.raises(TypeError):
    r * right(t)
  with pytest.raises(TypeError):
    t * r


# ------------------------------------------------------------------ exp

def test_exp_of_plain_vectors():
  cs = crystalFrame("432")
  v = vector3d(0, 0, pi / 2)
  assert exp(v) == rotation.byAxisAngle(vector3d.Z, pi / 2) and type(exp(v)) is rotation
  # a crystal direction is a misorientation of its frame with itself
  m = Miller(1, 1, 1, cs, "uvw").normalize() * (30 * degree)
  e = exp(m)
  assert isinstance(e, orientation) and e.CS is cs and e.SS is cs and close(e.angle(), 30 * degree)
  # with a reference, a crystal direction updates from the right and a specimen vector from the left
  ref = orientation.rand(cs, rng=rng)
  assert angle(exp(m, ref), ref * e) < 1e-10
  assert angle(exp(v, ref), rotation(exp(v)) * ref) < 1e-10
  assert angle(exp(vector3d(v.data), ref, "right"), ref * rotation(exp(v))) < 1e-10
  assert angle(ref.exp(m), exp(m, ref)) < 1e-10
  # the frames of the reference survive
  ss = specimenFrame("222")
  ref = orientation.rand(cs, ss, rng=rng)
  assert exp(v, ref).SS is ss and exp(v, ref).CS is cs
  assert exp(m, ref).SS is ss and exp(m, ref).CS is cs
  with pytest.raises(ValueError):
    exp(m, ref, "left")


# ------------------------------------------------------------------ geodesic

def test_geodesic_runs_to_the_nearest_equivalent():
  cs = crystalFrame("432")
  oA = orientation.byEuler(20 * degree, 30 * degree, 10 * degree, cs)
  oB = orientation.byEuler(120 * degree, 60 * degree, 40 * degree, cs)
  s = np.linspace(0, 1, 121)
  path = geodesic(oA, oB, s)
  assert isinstance(path, orientation) and path.shape == (121,) and path.CS is cs
  assert angle(path[0], oA) < 1e-10 and angle(path[-1], oB) < 1e-10
  steps = np.sum(angle(path[:-1], path[1:]))
  assert abs(steps - angle(oA, oB)) < 1e-8
  assert angle(rotation(path[-1]), rotation(oB)) > 10 * degree
  assert angle(oA.geodesic(oB, 0.5), oA) - angle(oA, oB) / 2 < 1e-10
  # rotations, and a NaN endpoint yields the other
  rA, rB = rotation.rand(rng=rng), rotation.rand(rng=rng)
  assert angle(geodesic(rA, rB, 1), rB) < 1e-10 and angle(geodesic(rA, rB, 0), rA) < 1e-10
  assert angle(geodesic(rotation.nan(), rB, 0.3), rB) < 1e-10 and angle(geodesic(rA, rotation.nan(), 0.3), rA) < 1e-10


def test_display():
  cs, ss = crystalFrame("321"), specimenFrame("222")
  ref = orientation.rand(cs, ss, rng=rng)
  t = log(orientation.rand(3, cs, ss, rng=rng), ref)
  s = repr(t)
  assert s.startswith("SO3TangentVector (left") and "reference: orientation" in s and "size: 3" in s
  assert "reference: orientation" in repr(right(t)) and "W" in repr(right(t)).split("\n")[3]
  assert "reference: rotation" in repr(log(rotation.rand(rng=rng)))

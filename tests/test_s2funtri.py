import numpy as np

from mtex import S2FunTri, dot, interp, mean, vector3d, zvector

rng = np.random.default_rng(5)

# the linear interpolant of z^2 at random directions, and directions in between
v = vector3d.rand(2000, rng=rng)
y = v.z ** 2
sF = S2FunTri(v, y)
w = vector3d.rand(200, rng=rng)


# it reproduces the values at its vertices and is close to the function in between
def test_S2FunTri_interpolatesItsValues():
  assert np.allclose(sF.eval(v), y)
  assert np.max(np.abs(sF(w) - w.z ** 2)) < 0.05


# a direction given twice takes the mean of its values
def test_S2FunTri_averagesADirectionGivenTwice():
  s2 = S2FunTri(vector3d(np.concatenate([v.data, v.data[:1]])), np.concatenate([y, [y[0] + 1]]))
  assert s2.vertices.size == v.size and np.isclose(s2.eval(v[0]), y[0] + 0.5)


# the mean of z^2 over the sphere is a third
def test_mean_ofTheInterpolantIsTheIntegral():
  assert abs(mean(sF) - 1 / 3) < 0.01


# the algebra acts on the values at the vertices
def test_S2FunTri_algebraActsOnTheVertexValues():
  assert np.allclose((2 * sF + 1).eval(v), 2 * y + 1) and np.allclose((sF * sF).eval(v), y ** 2)


# from a callable, on a grid of 1.5 degrees
def test_S2FunTri_fromACallableApproximatesIt():
  f = S2FunTri(lambda u: dot(u, zvector))
  assert np.max(np.abs(f(w) - w.z)) < 1e-3


def test_interp_byMethod():
  v = vector3d.rand(1500, rng=rng)
  y = v.x + v.z
  w = vector3d.rand(100, rng=rng)
  assert isinstance(interp(v, y, "linear"), S2FunTri)
  assert np.max(np.abs(interp(v, y, "harmonic")(w) - (w.x + w.z))) < 1e-3
  assert np.allclose(interp(v, y, "nearest")(v[:10]), y[:10])
  assert np.max(np.abs(interp(v, y, w, "linear") - (w.x + w.z))) < 0.02
  assert np.max(np.abs(interp(v, y, w) - (w.x + w.z))) < 0.1
  assert np.max(np.abs(interp(v, y, w, "inverseDistance") - (w.x + w.z))) < 0.2

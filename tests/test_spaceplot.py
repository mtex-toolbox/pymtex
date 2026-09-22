# the drawing of plots in space (docs/plans/plotting.md S5), on the Agg backend
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

from mtex import (S2AxisFieldHarmonic, S2FunHarmonic, crystalFrame, degree, discreteSample, fundamentalRegion, gcm, grad, orientation,
                  plot, plot3d, rotation, scatter, scatter3d, specimenFrame, unimodalODF, vector3d, zvector)

rng = np.random.default_rng(9)


@pytest.fixture(autouse=True)
def closeFigures():
  yield
  plt.close("all")


def axesOf():
  return gcm().children[-1]


def test_directions_functions_and_fields_on_the_sphere():
  ss = specimenFrame("1")
  vs = vector3d(vector3d.rand(30, rng=rng), frame=ss)
  h = plot3d(vs)
  ax = axesOf()
  assert ax.name == "3d" and hasattr(ax, "_mtexSpacePlot") and len(h.get_offsets()) == 30
  # the sphere, the grid and the axes labels are there
  assert any(isinstance(c, Poly3DCollection) for c in ax.collections) and len(ax.lines) == 17
  assert {t.get_text() for t in ax.texts} == {"X", "Y", "Z"}
  # axes are drawn at both ends, values through the colour map with a bar
  h = scatter3d(vector3d.rand(10, antipodal=True, rng=rng), rng.random(10), colorbar=True)
  assert len(h.get_offsets()) == 20 and len(gcm().fig.axes) == 2
  vs.plot3d(label=[str(k) for k in range(30)], noLabel=True)
  assert len(axesOf().texts) == 30
  # a function colours the sphere at its faces, radial pushes it out
  s = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True) + 3
  h = plot3d(s, colorbar=True)
  assert isinstance(h, Poly3DCollection) and h.get_array().size == len(h.get_paths()) if hasattr(h, "get_paths") else True
  h = s.plot3d(radial=True, resolution=10 * degree)
  assert isinstance(h, Poly3DCollection) and axesOf().get_xlim()[1] > 1.3
  # a field as arrows, an axis field as segments
  h = plot3d(grad(s))
  assert isinstance(h, Line3DCollection)
  A = S2AxisFieldHarmonic(lambda v: vector3d(v.cross(zvector).data, antipodal=True), bandwidth=12)
  h = A.plot3d()
  assert isinstance(h, Line3DCollection)


def test_orientation_space():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  oR = fundamentalRegion(cs, ss)
  h = plot(oR)
  ax = axesOf()
  assert isinstance(h, Poly3DCollection) and ax.get_xlabel() == "x / °"
  assert ax.get_xlim()[1] > 60 and len(ax.lines) > 10
  oR.plot(lines=False)
  assert len(axesOf().lines) == 0
  # a misorientation region has walls from the identity
  mR = fundamentalRegion(cs, cs)
  plot(mR)
  assert len([c for c in axesOf().collections if isinstance(c, Poly3DCollection)]) > 1
  # orientations as points in the region, in degrees, coloured by a value
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=10 * degree)
  ori = discreteSample(odf, 200, rng=rng)
  # an orientation scatters in the Euler region by default, as MATLAB does; axis angle on request
  h = scatter(ori, axisAngle=True)
  ax = axesOf()
  P = np.stack([np.asarray(a) for a in h._offsets3d], axis=1)
  assert P.shape == (200, 3) and np.all(np.linalg.norm(P, axis=1) <= 63) and ax.get_xlabel() == "x / °"
  assert any(isinstance(c, Poly3DCollection) for c in ax.collections)
  h = ori.scatter(np.degrees(ori.angle()), colorbar=True, region=False, points=50)
  assert len(h._offsets3d[0]) == 50 and not any(isinstance(c, Poly3DCollection) for c in axesOf().collections)
  # Euler and Rodrigues coordinates
  h = scatter(ori, Euler=True)
  ax = axesOf()
  P = np.stack([np.asarray(a) for a in h._offsets3d], axis=1)
  assert ax.get_xlabel() == "φ1 / °" and np.all(P[:, 1] <= 90 + 1e-6) and np.all(P[:, 2] <= 90 + 1e-6)
  h = scatter(ori, Rodrigues=True)
  assert axesOf().get_xlabel() == "r1"
  # the representatives as stored, and a single orientation
  h = scatter(ori, ignoreFundamentalRegion=True)
  assert len(h._offsets3d[0]) == 200
  plot3d(ori[0])
  assert len(axesOf().collections) >= 1

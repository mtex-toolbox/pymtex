# the vector table: read by column names, written back, the other columns alongside
import os

import numpy as np

from mtex import angle, degree, export, loadVector3d, mtexdatafile, vector3d


def test_load_by_names_and_positions():
  f = mtexdatafile("vectors")
  v = vector3d.load(f, columnNames=["polar angle", "azimuth angle"])
  assert v.shape == (1000,)
  w = vector3d.load(f, columnNames=["azimuth angle", "polar angle"], columns=[1, 0])
  assert np.max(angle(v, w)) < 1e-12
  # the names on the header line, a two word name read as one column
  u = vector3d.load(f)
  assert np.max(angle(v, u)) < 1e-12
  assert np.allclose(np.linalg.norm(v.data, axis=-1), 1)


def test_round_trip_with_a_column(tmp_path):
  v = vector3d.byPolar(np.arange(10, 91, 20) * degree, np.arange(-80, 81, 40) * degree)
  f = os.path.join(tmp_path, "v.txt")
  export(v, f)
  assert open(f).read().split("\n")[0].split() == ["x", "y", "z"]
  w = vector3d.load(f, columnNames=["x", "y", "z"])
  assert np.max(angle(v, w)) / degree < 1e-4
  export(v, f, polar=True)
  w = vector3d.load(f, columnNames=["polar angle", "azimuth angle"])
  assert np.max(angle(v, w)) / degree < 1e-4
  export(v, f, polar=True, radians=True)
  w = vector3d.load(f, columnNames=["polar angle", "azimuth angle"], radians=True)
  # six significant digits of a radian are coarser than of a degree
  assert np.max(angle(v, w)) / degree < 1e-3
  weight = np.arange(1, 6) / 15
  v.export(f, {"weight": weight})
  w, S = loadVector3d(f, columnNames=["x", "y", "z", "weight"])
  assert np.max(np.abs(S["weight"] - weight)) < 1e-6
  # lengths survive the Cartesian table
  export(2 * v, f)
  assert np.allclose(np.linalg.norm(vector3d.load(f).data, axis=-1), 2, atol=1e-5)

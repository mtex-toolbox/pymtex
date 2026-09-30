# orientation lists as text files (MATLAB @quaternion/export, export_VPSC)
import numpy as np

from mtex import crystalFrame, degree, export, orientation
from mtex.io.exportorientation import _export_VPSC


def test_the_euler_table_and_the_vpsc_file(tmp_path):
  cs = crystalFrame("mmm", [4.7, 10.2, 6.0], mineral="Fo")
  o = orientation.byEuler(np.array([10, 20]) * degree, np.array([30, 40]) * degree, np.array([50, 60]) * degree, cs)
  f = tmp_path / "o.txt"
  export(o, f, "Bunge")
  lines = f.read_text().splitlines()
  assert lines[0].split() == ["phi1", "Phi", "phi2"] and [float(x) for x in lines[1].split()] == [10, 30, 50]
  # more columns from a dictionary, radians and quaternions on request
  export(o, f, {"area": [1.5, 2.5]}, "Bunge")
  lines = f.read_text().splitlines()
  assert lines[0].split()[-1] == "area" and float(lines[2].split()[-1]) == 2.5
  export(o, f, radians=True)
  assert np.isclose(float(f.read_text().splitlines()[1].split()[0]), 10 * degree, atol=1e-4)
  export(o, f, quaternion=True)
  lines = f.read_text().splitlines()
  assert lines[0].split() == ["a", "b", "c", "d"] and np.allclose([float(x) for x in lines[1].split()], o.data[0], atol=1e-4)
  # the VPSC texture: four header lines, the convention letter, weights normalised
  _export_VPSC(o, f, "Bunge", weights=[1, 3])
  lines = f.read_text().splitlines()
  assert lines[0].startswith("texture") and lines[3] == "B 2"
  d = np.loadtxt(f, skiprows=4)
  assert d.shape == (2, 4) and np.isclose(d[:, 3].sum(), 1) and np.isclose(d[1, 3], 0.75) and np.allclose(d[:, 0], [10, 20])

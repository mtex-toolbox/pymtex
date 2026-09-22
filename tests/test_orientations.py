# the orientation table reader (MATLAB orientation/load.m)
import numpy as np

from mtex import crystalFrame, degree, orientation
from mtex.io.orientations import loadOrientation

cs = crystalFrame('m-3m')


def test_euler_columns_with_a_property_and_positions(tmp_path):
  ori = orientation.byEuler(np.array([10, 20, 30]) * degree, np.array([40, 50, 60]) * degree, np.array([70, 80, 5]) * degree, cs)
  E = np.stack([ori.phi1, ori.Phi, ori.phi2], axis=-1) / degree
  f = tmp_path / 'ori.txt'
  np.savetxt(f, np.column_stack([np.arange(3), E[:, 1], E[:, 0], E[:, 2]]))
  o, S = loadOrientation(f, cs, columnNames=['Grain Id', 'phi1', 'Phi', 'phi2'], columns=[0, 2, 1, 3])
  assert np.max(o.angle(ori)) < 1e-6
  assert np.allclose(S['grainid'], np.arange(3))


def test_radians_passive_and_quaternions(tmp_path):
  ori = orientation.rand(5, cs, rng=np.random.default_rng(1))
  f = tmp_path / 'rad.txt'
  np.savetxt(f, np.stack([ori.phi1, ori.Phi, ori.phi2], axis=-1))
  assert np.max(loadOrientation(f, cs, columnNames=['phi1', 'Phi', 'phi2'], radians=True)[0].angle(ori)) < 1e-6
  inv = loadOrientation(f, cs, columnNames=['phi1', 'Phi', 'phi2'], radians=True, passive=True)[0]
  assert np.allclose(np.abs(np.sum(inv.data * ori.inv().data, axis=-1)), 1)
  f = tmp_path / 'q.txt'
  np.savetxt(f, ori.data, header='Quat real, Quat i, Quat j, Quat k')
  q = orientation.load(f, cs, columnNames=['Quat real', 'Quat i', 'Quat j', 'Quat k'])
  assert np.max(q.angle(ori)) < 1e-6

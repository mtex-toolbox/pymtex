import os

# the tests see the defaults, never the settings file of whoever runs them
os.environ["MTEX_SETTINGS"] = ""

import numpy as np
import pytest

import mtex


# skip a test whose classes are not built yet, so the semantics file is complete before the code
def need(*names):
  missing = [n for n in names if not hasattr(mtex, n)]
  if missing:
    pytest.skip(f"not built yet: {', '.join(missing)}")


@pytest.fixture(autouse=True)
def fresh_session():
  reset = getattr(mtex, "reset", None)
  if reset is not None:
    reset()
  yield


# a DREAM.3D 6 file of 5 x 4 x 3 voxels along x, y, z of two phases, cubic and hexagonal, with
# Euler angles, feature ids and a confidence index; an xnovo file of the same grid as 'xnovo'
@pytest.fixture
def volumeFile(tmp_path):
  import h5py
  rng = np.random.default_rng(5)
  shape = (3, 4, 5)

  def write(kind="dream3d"):
    path = str(tmp_path / ("volume.dream3d" if kind == "dream3d" else "volume.h5"))
    with h5py.File(path, "w") as f:
      if kind == "dream3d":
        g = f.create_group("DataContainers/ImageDataContainer")
        g.create_dataset("CellData/EulerAngles", data=(rng.random(shape + (3,)) * [6, 3, 6]).astype(np.float32))
        g.create_dataset("CellData/Phases", data=rng.integers(1, 3, shape + (1,)).astype(np.int32))
        g.create_dataset("CellData/FeatureIds", data=rng.integers(1, 7, shape + (1,)).astype(np.int32))
        g.create_dataset("CellData/Confidence Index", data=rng.random(shape + (1,)).astype(np.float32))
        g.create_dataset("CellData/IPFColors", data=rng.integers(0, 255, shape + (3,)).astype(np.uint8))
        g.create_dataset("CellEnsembleData/CrystalStructures", data=np.array([[999], [1], [0]], dtype=np.uint32))
        for name, v in (("DIMENSIONS", np.array([5, 4, 3])), ("SPACING", np.float32([0.5, 0.25, 2.0])), ("ORIGIN", np.float32([0, 0, 0]))):
          g.create_dataset(f"_SIMPL_GEOMETRY/{name}", data=v)
      else:
        g = f.create_group("LabDCT")
        g.create_dataset("Data/PhaseId", data=rng.integers(0, 2, shape).astype(np.uint8))
        g.create_dataset("Data/Rodrigues", data=rng.normal(size=shape + (3,)).astype(np.float32))
        g.create_dataset("Data/GrainId", data=rng.integers(0, 7, shape).astype(np.int32))
        g.create_dataset("Center", data=np.zeros(3))
        g.create_dataset("Spacing", data=np.array([0.005, 0.005, 0.01]))
        p = f.create_group("PhaseInfo/Phase01")
        p.create_dataset("SpaceGroup", data=np.array([225], dtype=np.int32))
        p.create_dataset("UnitCell", data=np.array([4.05, 4.05, 4.05, 90, 90, 90.0]))
        p.create_dataset("Name", data=np.array([b"Aluminum"], dtype=object), dtype=h5py.string_dtype())
    return path

  return write


def close(a, b, tol=1e-12):
  return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0)


IN100 = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")


# the IN100 volume of ../master/data and its grains at 10 degrees, as MATLAB's pages segment it,
# built once per run; the tests only read them
@pytest.fixture(scope="session")
def in100():
  if not os.path.exists(IN100):
    pytest.skip("the IN100 volume of ../master/data")
  from mtex import EBSD3, calcGrains, degree
  ebsd = EBSD3.load(IN100)
  return ebsd, calcGrains(ebsd, angle=10 * degree)

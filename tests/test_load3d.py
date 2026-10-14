# the import of volumes: xnovo and a stack of maps, against MATLAB master (bench/matlab/probe_ebsd3.m)
import glob
import os

import numpy as np
import pytest

from mtex import EBSD3, rotation, setMTEXpref

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data")
XNOVO = os.path.join(DATA, "EBSD3", "SimulatedMultiPhase.h5")
# a DREAM3D-NX file with blanks in its group names and without a surface mesh
IN100_NX = os.path.join(DATA, "..", "..", "data", "dream3d", "SmallIN100_Final.dream3d")
# a DREAM.3D 6 file of a hexagonal and a cubic phase, with DREAM.3D's IPF colours with respect to y
ZIRCALOY = os.path.join(DATA, "..", "..", "data", "dream3d", "zenodo_3785085", "Zircaloy4_7Nb_slab.dream3d")


@pytest.mark.skipif(not os.path.exists(XNOVO), reason="the xnovo volume of ../master/data")
def test_the_xnovo_volume():
  x = EBSD3.load(XNOVO)
  assert x.shape == (50, 50, 50) and x.scanUnit == "mm" and len(x.CSList) == 10
  assert [f.pointGroup for f in x.CSList.frames[1:]] == ["m-3m", "m-3m", "6/mmm", "4/mmm", "-3m1", "-3m1", "mmm", "12/m1", "-1"]
  assert np.allclose(x.pos.data[0, 0, 0], [-0.245] * 3) and np.allclose(x.pos.data[19, 29, 39], [-0.055, 0.045, 0.145])
  assert x.phaseId[19, 29, 39] == 7 and np.allclose(x.rotations.data[19, 29, 39], [0.92317598, -0.36615592, 0.03882264, 0.11031204])
  # Corundum's rhombohedral cell read with the angles of the hexagonal one, as MATLAB
  assert np.allclose(x.CSList.frames[5].axes.data.reshape(3, 3)[1], [-2.564, 4.441, 0], atol=1e-3)


@pytest.mark.skipif(not glob.glob(os.path.join(DATA, "EBSD", "3dData", "*.ANG")), reason="the stack of ../master/data")
def test_a_stack_of_maps():
  files = sorted(glob.glob(os.path.join(DATA, "EBSD", "3dData", "*.ANG")))
  e = EBSD3.load(files, z=0.1 * np.arange(len(files)))
  assert e.lattice.shape == (35, 40, len(files)) and np.allclose(e.lattice.spacing, [0.4, 0.4, 0.1])
  assert e.gridify().isPadding.sum() == 0
  with pytest.raises(ValueError):
    EBSD3.load(files[:2], z=[0.0])


# the layouts a DREAM.3D file comes in: where the voxels, the phases' structures and the grid are
LAYOUTS = {
  "DREAM.3D 6": ("DataContainers/ImageDataContainer", "CellData", "CellEnsembleData", "datasets"),
  "DREAM3D-NX": ("DataStructure/DataContainer", "CellData", "CellEnsembleData", "attributes"),
  "names with spaces": ("DataStructure/Image Geometry", "Cell Data", "Cell Ensemble Data", "attributes"),
  "names of a pipeline": ("DataStructure/fw-ar-IF1", "Scan Data", "Phase Data", "attributes"),
}
# a volume of 4 x 3 x 2 voxels along x, y, z
SHAPE, SPACING, ORIGIN = (4, 3, 2), (0.5, 0.25, 2.0), (-1.0, 3.0, 0.5)


# the arrays of the volume as functions of the voxel (i, j, k), each an (x, y, z, components) array
def voxelArrays():
  rng = np.random.default_rng(3)
  n = SHAPE + (1,)
  return {"EulerAngles": (rng.random(SHAPE + (3,)) * [2 * np.pi, np.pi, 2 * np.pi]).astype(np.float32),
          "Phases": rng.integers(1, 3, n).astype(np.int32), "Mask": (rng.random(n) > 0.2).astype(np.uint8),
          "FeatureIds": rng.integers(1, 5, n).astype(np.int32), "Confidence Index": rng.random(n).astype(np.float32),
          "IPFColors": rng.integers(0, 255, SHAPE + (3,)).astype(np.uint8)}


# a DREAM.3D file of the layout holding the arrays, stored (z, y, x, components) as DREAM.3D does
def writeDream3d(path, layout, arrays, structures=(999, 1, 0), materials=None, lattices=None, **stored):
  import h5py
  root, cells, ensemble, grid = LAYOUTS[layout]
  with h5py.File(path, "w") as f:
    g = f.create_group(root)
    for name, a in arrays.items():
      g.create_dataset(f"{cells}/{name}", data=np.ascontiguousarray(np.swapaxes(a, 0, 2)), **stored)
    if structures is not None:
      g.create_dataset(f"{ensemble}/CrystalStructures", data=np.array(structures, dtype=np.uint32).reshape(-1, 1))
    if materials is not None:
      g.create_dataset(f"{ensemble}/MaterialName", data=np.array(materials, dtype=object), dtype=h5py.string_dtype())
    if lattices is not None:
      g.create_dataset(f"{ensemble}/LatticeConstants", data=np.array(lattices, dtype=np.float32))
    for name, v in (("DIMENSIONS", np.array(SHAPE, dtype=np.int64)), ("SPACING", np.float32(SPACING)), ("ORIGIN", np.float32(ORIGIN))):
      if grid == "datasets":
        g.create_dataset(f"_SIMPL_GEOMETRY/{name}", data=v)
      else:
        g.attrs["_" + name] = v
  return str(path)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_dream3d_readsTheVoxelsOfEveryLayout(tmp_path, layout):
  a = voxelArrays()
  e = EBSD3.load(writeDream3d(tmp_path / "v.dream3d", layout, a))
  eul = a["EulerAngles"].astype(float)
  assert e.shape == SHAPE and np.allclose(e.lattice.spacing, SPACING) and np.allclose(e.pos.data[0, 0, 0], ORIGIN)
  assert np.allclose(e.pos.data[3, 2, 1], np.array(ORIGIN) + np.array([3, 2, 1]) * SPACING)
  assert np.allclose(e.rotations.data, rotation.byEuler(eul[..., 0], eul[..., 1], eul[..., 2], "ZXZ").data, atol=1e-15)
  assert np.array_equal(e.phaseId, np.where(a["Mask"][..., 0] == 0, 0, a["Phases"][..., 0]))
  assert not e.CSList.isIndexed(0) and [f.pointGroup for f in e.CSList.frames[1:]] == ["432", "622"]
  # the scalar arrays are the properties, the feature ids the grain ids; the colours are not read
  assert sorted(e.prop) == ["ConfidenceIndex", "grainId"]
  assert np.array_equal(e.prop["grainId"], a["FeatureIds"][..., 0]) and np.array_equal(e.prop["ConfidenceIndex"], a["Confidence Index"][..., 0])
  # in the types of the file
  assert e.prop["grainId"].dtype == np.int32 and e.prop["ConfidenceIndex"].dtype == np.float32


@pytest.mark.parametrize("stored", [{}, {"chunks": True, "compression": "gzip"}], ids=["contiguous", "gzip"])
def test_dream3d_readOverTheCoresIsTheReadOfH5pyAndNumPy(tmp_path, monkeypatch, stored):
  import functools
  import mtex.io.load3d as load3d
  fname = writeDream3d(tmp_path / "v.dream3d", "DREAM.3D 6", voxelArrays(), **stored)
  # the threads read however small the arrays
  monkeypatch.setattr(load3d, "readArray", functools.partial(load3d.readArray, least=0))
  fast = EBSD3.load(fname)
  setMTEXpref("kernels", "numpy"), setMTEXpref("threads", 1)
  try:
    slow = EBSD3.load(fname)
  finally:
    setMTEXpref("kernels", "numba"), setMTEXpref("threads", None)
  assert np.allclose(fast.rotations.data, slow.rotations.data, atol=1e-15) and np.array_equal(fast.pos.data, slow.pos.data)
  assert np.array_equal(fast.phaseId, slow.phaseId) and all(np.array_equal(fast.prop[k], slow.prop[k]) for k in slow.prop)


def test_dream3d_withoutEulerAnglesTakesTheQuaternions(tmp_path):
  # DREAM.3D's (x, y, z, w) is the passive rotation
  a = voxelArrays()
  eul = a.pop("EulerAngles").astype(float)
  q = rotation.byEuler(eul[..., 0], eul[..., 1], eul[..., 2], "ZXZ").data
  a["Quats"] = np.concatenate([-q[..., 1:], q[..., :1]], axis=-1).astype(np.float32)
  e = EBSD3.load(writeDream3d(tmp_path / "v.dream3d", "DREAM3D-NX", a))
  assert np.allclose(e.rotations.data, q, atol=1e-6) and "Quats" not in e.prop


def test_dream3d_ofFeatureIdsAloneIsOnePhaseWithoutRotations(tmp_path):
  ids = voxelArrays()["FeatureIds"]
  e = EBSD3.load(writeDream3d(tmp_path / "v.dream3d", "DREAM.3D 6", {"FeatureIds": ids}, structures=None))
  assert np.all(np.isnan(e.rotations.data)) and np.all(e.phaseId == 1) and e.CSList.frames[1].pointGroup == "1"
  assert np.array_equal(e.grainId, ids[..., 0])


def test_dream3d_takesTheSymmetryItIsGiven(tmp_path):
  from mtex import crystalFrame
  cs = ["notIndexed", crystalFrame("m-3m", mineral="Ni"), crystalFrame("6/mmm", mineral="Ti")]
  e = EBSD3.load(writeDream3d(tmp_path / "v.dream3d", "DREAM3D-NX", voxelArrays()), CS=cs)
  assert [f.mineral for f in e.CSList.frames[1:]] == ["Ni", "Ti"]


def test_load_findsADream3dVolumeInAnH5File(tmp_path):
  assert EBSD3.load(writeDream3d(tmp_path / "v.h5", "names of a pipeline", voxelArrays())).shape == SHAPE


def test_load_refusesAFileWithoutVoxels(tmp_path):
  import h5py
  with h5py.File(tmp_path / "v.dream3d", "w") as f:
    f.create_dataset("DataStructure/Geometry/SharedVertexList", data=np.zeros((3, 3)))
  with pytest.raises(ValueError, match="voxel data"):
    EBSD3.load(str(tmp_path / "v.dream3d"))


@pytest.mark.parametrize("name, prop", [("FeatureIds", "grainId"), ("Confidence Index", "ConfidenceIndex"), ("X Position", "XPosition"),
                                        ("Average neighbour misorientations", "AverageNeighbourMisorientations"), ("2Theta-max", "x2Theta_max")])
def test_propertyName_isMATLABsValidName(name, prop):
  from mtex.io.load3d import _propertyName
  assert _propertyName(name) == prop


@pytest.mark.skipif(not os.path.exists(IN100_NX), reason="the DREAM3D-NX volume of ../data/dream3d")
def test_dream3d_readsADream3dNXFileAsMATLAB():
  # bench/matlab/probe_dream3d.m on MATLAB develop; the quaternions up to their sign
  e = EBSD3.load(IN100_NX)
  assert e.shape == (189, 201, 117) and np.allclose(e.lattice.spacing, 0.25) and np.bincount(e.phaseId.reshape(-1)).tolist() == [1072734, 3371979]
  assert np.allclose(e.pos.data[65, 64, 39], [-31, 16, -19.5]) and e.CSList.frames[1].pointGroup == "432"
  assert e.CSList.frames[1].mineral == "Nickel"
  assert np.allclose(e.rotations.data[94, 100, 58], [-0.30812373, 0.44180860, 0.67022787, 0.51054827], atol=1e-8)
  assert np.allclose(e.rotations.data[65, 64, 39], [0.74085404, 0.54717957, 0.15549391, -0.35714345], atol=1e-8)
  sums = {"ConfidenceIndex": 1315316.569, "Fit": -104365549.8, "ImageQuality": 684004956.2, "ParentIds": 3796976474, "SEMSignal": 7261014293,
          "XPosition": 101950104.8, "YPosition": 110691801.8, "grainId": 3946028298}
  assert sorted(e.prop) == sorted(sums) and all(np.isclose(e.prop[k].sum(dtype=float), v, rtol=1e-9) for k, v in sums.items())


@pytest.mark.parametrize("layout", LAYOUTS)
def test_dream3d_namesThePhasesByTheirMaterialsOnTheirCells(tmp_path, layout):
  # the names without the blanks DREAM.3D leaves, the cells in degrees; a name the file leaves
  # blank is unknown, a cell of zeros the group's own
  a = voxelArrays()
  cells = [[0] * 6, [3.52, 3.52, 3.52, 90, 90, 90], [3.231, 3.231, 5.148, 90, 90, 120]]
  e = EBSD3.load(writeDream3d(tmp_path / "v.dream3d", layout, a, materials=["Invalid Phase", "Nickel ", "Zr"], lattices=cells))
  assert [f.mineral for f in e.CSList.frames[1:]] == ["Nickel", "Zr"]
  assert [np.linalg.norm(v) for v in e.CSList.frames[2].axes.data.reshape(3, 3)] == pytest.approx([3.231, 3.231, 5.148], rel=1e-6)
  blank = EBSD3.load(writeDream3d(tmp_path / "b.dream3d", layout, a, materials=["", " ", "Zr"], lattices=[[0] * 6] * 3))
  assert [f.mineral for f in blank.CSList.frames[1:]] == ["unknown", "Zr"]
  assert [np.linalg.norm(v) for v in blank.CSList.frames[2].axes.data.reshape(3, 3)] == pytest.approx([1, 1, 1])


@pytest.mark.skipif(not os.path.exists(ZIRCALOY), reason="the Zircaloy slab of zenodo 3785085")
def test_dream3d_hexagonalPhaseIsAlignedAsDREAM3DColoursIt():
  # the voxels DREAM.3D colours green and blue, the corners its key labels <2-1-10> and <10-10>,
  # are nearest those directions in the frame the reader builds
  import h5py
  from mtex import Miller, degree, orientation, yvector
  from mtex.io.load3d import _dream3dEnsemble
  cells = "DataContainers/ImageDataContainer/CellData/"
  with h5py.File(ZIRCALOY, "r") as f:
    cs = _dream3dEnsemble(f, "DataContainers/ImageDataContainer/CellEnsembleData/CrystalStructures")[1]
    eul = f[cells + "EulerAngles"][60].reshape(-1, 3).astype(float)
    col = f[cells + "IPFColor"][60].reshape(-1, 3) / 255
    hexagonal = f[cells + "Phases"][60].reshape(-1) == 1
  assert cs.pointGroup == "622"
  h = orientation(rotation.byEuler(eul[:, 0], eul[:, 1], eul[:, 2]), cs).inv() * yvector
  for k, near, far in ((1, (2, -1, -1, 0), (1, 0, -1, 0)), (2, (1, 0, -1, 0), (2, -1, -1, 0))):
    pure = np.flatnonzero(hexagonal & (col[:, k] > 0.8) & (np.delete(col, k, axis=1).max(axis=1) < 0.4))
    assert len(pure) > 50
    d = [np.median(np.asarray(h[pure].angle(Miller(*idx, cs, "uvw"), antipodal=True))) / degree for idx in (near, far)]
    assert d[0] < 10 < 25 < d[1], (k, d)


def test_dream3dPhases_reads_the_structures_after_the_first():
  from mtex.io.load3d import _dream3dPhases
  phases = _dream3dPhases(np.array([[999], [1], [7], [999]], dtype=np.uint32))
  assert phases[0] == "notIndexed" and phases[1].pointGroup == "432" and phases[2].pointGroup == "4" and phases[3] == "notIndexed"

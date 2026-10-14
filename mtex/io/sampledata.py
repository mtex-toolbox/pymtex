# the sample data of the documentation (MATLAB tools/mtexdata.m, Julia sampledata.jl)
#
#   mtexdata()                    the table of the data sets
#   ebsd = mtexdata('forsterite') one of them, downloaded on first use and read with its recipe
#   mtexdatafile('forsterite')    the path of its file, without the recipe
#
# A recipe is what the MATLAB page does to the file: the phases it names, the Euler
# correction it states, the crop or the turn it applies. The files come from a pinned tag
# of mtex-toolbox/mtex-data, the data repository of all ports, whose manifest.json is
# sampledata.json here, and are checked against their digests; they are kept under the
# preference 'mtexDataPath', the environment variable MTEX_DATA, or ~/.mtex/data, and a
# local copy of MTEX's data folder serves as well.
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..geometry.frames import crystalFrame, specimenFrame
from ..geometry.quaternion import rotation
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .load import loadEBSD
from .orientations import loadOrientation
from .vectors import loadVector3d
from ..geometry.display import aligned

if TYPE_CHECKING:
  from ..functions.polefigure import PoleFigure
  from ..functions.s2funclass import S2Fun
  from ..functions.so3funclass import SO3Fun
  from ..maps.ebsd import EBSD
  from ..maps.grains import grain2d
  from ..maps3d.ebsd3 import EBSD3
  from ..tensors.tensor import tensor
  from .vocabulary import (SampleDataName, SampleEBSD3Name, SampleEBSDName, SampleFileName, SampleGrainsName,
                           SamplePoleFigureName, SampleS2FunName, SampleSO3FunName, SampleTensorName)

with open(os.path.join(os.path.dirname(__file__), "sampledata.json")) as _f:
  MANIFEST = json.load(_f)

# the files of every data set of the data repository, by its name in lower case
SOURCES = {name.lower(): [dict(MANIFEST["files"][f], path=f) for f in d["files"]] for name, d in MANIFEST["datasets"].items()}

# the data sets with a recipe: name, type, description, the data set whose files they read
# ("" for none), the recipe
DATASETS = []


def _dataset(name, kind, description, artifact, recipe):
  DATASETS.append({"name": name, "type": kind, "description": description, "artifact": artifact,
                   "files": [s["path"] for s in SOURCES.get(artifact.lower(), [])], "recipe": recipe})


def mtexDataPath():
  """the directory the sample data are fetched into: the preference mtexDataPath, else the
  environment variable MTEX_DATA, else ~/.mtex/data"""
  root = getMTEXpref("mtexDataPath") or os.environ.get("MTEX_DATA") or os.path.join(os.path.expanduser("~"), ".mtex", "data")
  return os.path.abspath(os.path.expanduser(root))


def _sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


# where a source file is, under the data path or its EBSD or PoleFigure folder, fetched and
# checked where it is not there yet
def _resolve(source, root, download):
  parts = source["path"].split("/")
  full = os.path.join(root, *parts)
  short = os.path.join(root, *parts[1:]) if len(parts) > 1 else full
  for p in (full, short):
    if os.path.isfile(p):
      return p
  if not download:
    raise FileNotFoundError(f"the sample data file {source['path']} is not under {root}")
  os.makedirs(os.path.dirname(full), exist_ok=True)
  print(f"  downloading {source['url']}\n  to {full}")
  urllib.request.urlretrieve(source["url"], full)
  if _sha256(full) != source["sha256"]:
    os.remove(full)
    raise OSError(f"the download of {source['path']} does not match its digest")
  return full


# the data set of a recipe by its name, case aside; None for a name with none
def _lookup(name):
  key = str(name).lower()
  return next((d for d in DATASETS if d["name"].lower() == key), None)


def _find(name):
  d = _lookup(name)
  if d is None:
    raise ValueError(f"no sample data set called {name!r}; mtexdata() lists them")
  return d


def mtexdatafile(name: SampleFileName | Any, root=None, download=True):
  """the file of a sample data set, or its files, fetched on first use

  Besides the data sets mtexdata reads, every data set of the data repository has its
  files here, the Neper tessellation 'my100grains' for instance. root= is the folder the
  files are kept in, mtexDataPath() by default; download=False raises for a missing one
  instead of fetching it.

    fname = mtexdatafile('forsterite')
    tess, oris = mtexdatafile('my100grains')      # a data set of several files
    fname = mtexdatafile('forsterite', root=mtexDataPath(), download=False)
  """
  # a data set without a recipe here is named by the repository
  d = _lookup(name)
  artifact = str(name).lower() if d is None else d["artifact"].lower()
  if artifact not in SOURCES and artifact:
    raise ValueError(f"no sample data set called {name!r}; mtexdata() lists them")
  root = mtexDataPath() if root is None else os.path.abspath(os.path.expanduser(root))
  paths = [_resolve(s, root, download) for s in SOURCES.get(artifact, [])]
  return paths[0] if len(paths) == 1 else paths


# the class of what a data set is read as, for the editor
@overload
def mtexdata(name: None = None, root=None, download=True, verbose=True) -> SampleTable: ...


@overload
def mtexdata(name: SampleEBSDName, root=None, download=True, verbose=True) -> EBSD: ...


@overload
def mtexdata(name: SampleGrainsName, root=None, download=True, verbose=True) -> grain2d: ...


@overload
def mtexdata(name: SamplePoleFigureName, root=None, download=True, verbose=True) -> PoleFigure: ...


@overload
def mtexdata(name: SampleEBSD3Name, root=None, download=True, verbose=True) -> EBSD3: ...


@overload
def mtexdata(name: SampleSO3FunName, root=None, download=True, verbose=True) -> SO3Fun: ...


@overload
def mtexdata(name: SampleS2FunName, root=None, download=True, verbose=True) -> S2Fun: ...


@overload
def mtexdata(name: SampleTensorName, root=None, download=True, verbose=True) -> tensor: ...


@overload
def mtexdata(name: SampleDataName | Any, root=None, download=True, verbose=True) -> Any: ...

def mtexdata(name=None, root=None, download=True, verbose=True):
  """the sample data of the documentation, read from the data repository

    mtexdata()                                  # the table of the data sets
    ebsd = mtexdata('forsterite')               # a map, read with the recipe its page uses
    pf = mtexdata('dubna')                      # pole figures, with their directions and coefficients
    ebsd = mtexdata('forsterite', root=mtexDataPath(), download=False)  # from the local folder only

  The files are fetched from the data repository mtex-toolbox/mtex-data on first use and
  kept under getMTEXpref('mtexDataPath'), MTEX_DATA or ~/.mtex/data; root names another
  folder, a local copy of MTEX's data folder for instance. Every data set is read as its
  documentation page does, in the plotting convention of the session: a page that wants a
  particular alignment sets it itself. mtexdatafile gives the files without reading them.

  Input
    name - the name of the data set, case does not matter
  Output
    ebsd - EBSD, grain2d, PoleFigure, EBSD3, SO3Fun, S2Fun or tensor, by the data set; the
        table of the data sets without a name
  Keywords
    root - the folder the files are kept in, instead of mtexDataPath()
    download - fetch a missing file; False raises instead
    verbose - accepted for MATLAB's silent flag; reading prints nothing, a download its source
  See also
    mtexdatafile, loadEBSD, loadPoleFigure"""
  if name is None:
    return SampleTable()
  d = _find(name)
  files = mtexdatafile(name, root, download) if d["files"] else None
  # a binary pair is one map, named by its first file
  if isinstance(files, list) and d["type"] == "EBSD":
    files = files[0]
  return d["recipe"](files)


def _fileNames(files):
  if len(files) <= 2:
    return ", ".join(files)
  return f"{files[0]} and {len(files) - 1} more"


class SampleTable:
  """the data sets, as rows of name, type, description and files"""

  def __init__(self):
    self.rows = [(d["name"], d["type"], d["description"], _fileNames(d["files"])) for d in DATASETS]

  def __iter__(self):
    return iter(d["name"] for d in DATASETS)

  def __len__(self):
    return len(DATASETS)

  def __repr__(self):
    labels = ("name", "type", "description", "files")
    widths = [max(len(labels[j]), *(len(r[j]) for r in self.rows)) for j in range(4)]
    def line(r):
      return "  " + "  ".join(str(x).ljust(w) for x, w in zip(r, widths))
    return aligned("available sample data\n" + "\n".join([line(labels)] + [line(r) for r in self.rows]))


# ------------------------------------------------------------------ the recipes

X_TURN = rotation.byAxisAngle(vector3d(1, 0, 0), np.pi)


def _oxford(path):
  return loadEBSD(path, eulerCorrection=X_TURN)


def _small(path):
  e = _oxford(path)
  return e[e.inpolygon([33000, 4500, 3000, 3000])]


def _olivine(path):
  e = loadEBSD(path, eulerCorrection=rotation.id)
  e = e.rotate(rotation.byEuler(90 * degree, 180 * degree, 180 * degree), keepEuler=True)
  return e.rotate(rotation.byEuler(0.0, 0.0, 90 * degree))


def _epidote(path):
  # the phases 3 and 4 of the file, Pyrope and omphacite, are left out as the page does
  e = _oxford(path)
  ids = e.phaseId.copy()
  ids[np.isin(e.phase, [3, 4])] = 0
  return _withPhaseId(e, ids)


def _withPhaseId(e, ids):
  from ..maps.ebsd import EBSD
  rot = e.rotations.data.copy()
  rot[ids == 0] = np.nan
  return EBSD(e.pos, rotation(rot), ids, e.CSList, e.prop, scanUnit=e.scanUnit, meta=e.meta, phaseMap=e.phaseMap, id=e.id,
              lattice=e.lattice, isPadding=e.isPadding)


def _proper(e, names):
  # the proper groups of the named phases, as the page uses them for the variants
  for name in names:
    k = e.CSList.phaseId(name)
    e.CSList.frames[k] = e.CSList.frames[k].properGroup()
  return e


def _aachen(path):
  cs = ["notIndexed", crystalFrame("m-3m", mineral="Fe"), crystalFrame("m-3m", mineral="Mg")]
  return loadEBSD(path, cs, columnNames=["Index", "Phase", "x", "y", "Euler 1", "Euler 2", "Euler 3", "MAD", "BC", "BS", "Bands", "Error", "ReliabilityIndex"])


def _sharp(path):
  return loadEBSD(path, ["notIndexed", crystalFrame("-3m", [5, 5, 17], mineral="calcite")], columnNames=["Euler 1", "Euler 2", "Euler 3", "Phase", "x", "y"])


def _csl(path):
  # this export labels its only phase -1 and spells a failed angle in Italian; the sole
  # phase is iron, and a measurement without an angle is of no phase
  return loadEBSD(path, crystalFrame("m-3m", mineral="iron"), columnNames=["filephase", "x", "y", "Euler 1", "Euler 2", "Euler 3", "IQ", "CI", "Error"],
                  replaceExpr=[("Non un numero reale", "nan")])


def _mylonite(path):
  cs = [crystalFrame("-1", [8.169, 12.851, 7.1124], np.array([93.63, 116.4, 89.46]) * degree, mineral="Andesina"),
        crystalFrame("-3m", [4.913, 4.913, 5.504], mineral="Quartz"),
        crystalFrame("2/m11", [5.339, 9.249, 20.196], np.array([95.06, 90, 90]) * degree, mineral="Biotite"),
        crystalFrame("12/m1", [8.5632, 12.963, 7.2099], np.array([90, 116.07, 90]) * degree, mineral="Orthoclase")]
  return loadEBSD(path, cs, columnNames=["Phase", "x", "y", "Euler 1", "Euler 2", "Euler 3"])


def _aluminium(path):
  cs = crystalFrame("m-3m", [4.04958, 4.04958, 4.04958], mineral="Al")
  return loadEBSD(path, cs, columnNames=["Euler 1", "Euler 2", "Euler 3", "x", "y"], columns=[0, 1, 2, 3, 4], radians=True)


def _titanium(path):
  cs = crystalFrame("622", [3, 3, 4.7], "X||a", mineral="Titanium (Alpha)")
  return loadEBSD(path, cs, columnNames=["phi1", "Phi", "phi2", "phase", "ci", "iq", "sem_signal", "x", "y", "grainId"])


def _martensite(path):
  return _proper(_oxford(path), ["Iron bcc (old)", "Iron fcc"])


def _alphaBetaTitanium(path):
  return _proper(loadEBSD(path, eulerCorrection=rotation.id), ["Ti (alpha)", "Ti (BETA)"])


def _emsphinx(path):
  # an in situ deformed austenitic steel indexed with EMSphInx; the file names no phase,
  # its master patterns were alpha and gamma iron plus alpha cobalt; of its four scans the
  # first is the sample map, read without the list of the others
  e = loadEBSD(path, verbose=False)
  for k, name in enumerate(("Iron bcc", "Iron fcc", "Cobalt")):
    e.CSList.frames[k + 1].mineral = name
  return e


# the olivine texture history of a VPSC run as the page reads it: one ODF of halfwidth 10
# degrees per strain step; the slip activity is the second file of mtexdatafile('vpsc')
def _vpsc(files):
  from .loadodf import _loadODF_VPSC
  cs = crystalFrame("222", [4.762, 10.225, 5.994], mineral="olivine")
  return _loadODF_VPSC(files[0], halfwidth=10 * degree, CS=cs)


# the pole figure sets: the crystal frame of the page, the directions the file names do
# not spell, the structure coefficients of the superposed Dubna reflection
def _dubna(files):
  from .loadpolefigure import loadPoleFigure
  cs = crystalFrame("321", [4.916, 4.916, 5.4054], mineral="Quartz")
  return loadPoleFigure(files, cs, superposition=[1, 1, [0.52, 1.23], 1, 1, 1, 1])


def _odf(path):
  from .loadodf import loadODF
  return loadODF(path, "Bunge", "interp", CS=crystalFrame("m-3m"), SS=specimenFrame("1"), columnNames=["Euler 1", "Euler 2", "Euler 3", "weights"])


class imageSet(SimpleNamespace):
  """the images a sample map carries along, as MATLAB's struct: img.fsdT1, img.pixSzImg"""

  def __repr__(self):
    rows = [(k, " × ".join(str(n) for n in np.shape(v)) + " double" if np.ndim(v) else f"{float(v):g}") for k, v in vars(self).items()]
    w = max(len(k) for k, _ in rows)
    return aligned("\n".join(f"  {k:>{w}}: {v}" for k, v in rows))


# a volume, read by EBSD3.load
def _volume(path):
  from ..maps3d.ebsd3 import EBSD3
  return EBSD3.load(path)


# the WC-Co map of TrueEBSD with its four forescatter images, the centre half of the full
# field every fourth point, as MATLAB's .mat holds it, read from the HDF5 file of plain
# arrays bench/matlab/export_trueebsd.m writes: h5py reads MATLAB's arrays transposed
def _trueEbsdWCCoSmall(path):
  import h5py
  from ..geometry.frames import notIndexedFrame
  from ..maps.ebsd import EBSD
  with h5py.File(path, "r") as f:
    e = f["ebsd"]
    x, y = np.asarray(e["x"]).ravel(), np.asarray(e["y"]).ravel()
    q = np.asarray(e["rotation"]).reshape(4, -1).T
    phaseId = np.asarray(e["phaseId"]).ravel().astype(int) - 1
    prop = {k: np.asarray(v).ravel() for k, v in e["prop"].items()}
    scanUnit = e.attrs["scanUnit"]
    CSList = []
    for k in sorted(f["phases"], key=int):
      a = f["phases"][k].attrs
      if "pointGroup" not in a:
        CSList.append(notIndexedFrame())
        continue
      align = a["alignment"]
      align = [s.strip() for s in align.split(",")] if isinstance(align, str) and align else []
      # the lengths are the norms of MATLAB's axes, which differ in the last digits
      CSList.append(crystalFrame(a["pointGroup"], list(np.round(a["axes"], 5)), np.round(a["angles"], 4) * degree, *align, mineral=a["mineral"]))
    im = f["images"]
    imgs = {k: np.asarray(im[k]).T for k in ("fsdB3", "fsdT3", "fsdT1", "fsdT10")}
    imgs["pixSzImg"] = float(np.asarray(im["pixSzImg"]).ravel()[0])
  ebsd = EBSD(np.column_stack([x, y, np.zeros_like(x)]), rotation(q), phaseId, CSList, prop, scanUnit=scanUnit).gridify()
  ebsd.meta["trueEbsdImgs"] = imageSet(**imgs)
  return ebsd


def _dubnaODF(files):
  from ..analysis.pf2odf import calcODF
  return calcODF(_dubna(files))


def _geesthacht(path):
  from ..geometry.miller import Miller
  from .loadpolefigure import loadPoleFigure
  return loadPoleFigure(path, Miller([1, 1, 1, 1], [0, 0, 1, 1], [4, 4, 0, 0], crystalFrame("m-3m")))


def _ptx(files):
  from .loadpolefigure import loadPoleFigure
  return loadPoleFigure(files, crystalFrame("mmm"))


def _steel(files):
  from ..geometry.miller import Miller
  from .loadpolefigure import loadPoleFigure
  pf = loadPoleFigure(files, Miller([1, 2, 2], [1, 0, 1], [0, 0, 1], crystalFrame("m-3m")), interface="rw1")
  return pf.rotate(90 * degree)


# the zinc alloy sheet with the texture-free reference measured on the same grid; the
# lattice is zinc's, the page states none
def _zncuti(files):
  from .loadpolefigure import loadPoleFigure
  return loadPoleFigure(files, crystalFrame("6/mmm", [2.665, 2.665, 4.947], mineral="Zn"))


def _aachenexp(path):
  from .loadpolefigure import loadPoleFigure
  return loadPoleFigure(path)


def testGrainsMap():
  """artificial grain shapes, MTEX's testgrains: a map of one phase in which every shape is a
  grain of its own orientation, so that every shape parameter has something to react to; a
  disc, discs with a small and a large hole, an ellipse, a rectangle and a turned one, a
  clover, a disc with a slit, an L, a triangle, a hexagon and a disc with a bay, inside one
  background grain. The holes are not indexed, and the orientations step by seven degrees."""
  from ..maps.ebsd import EBSD
  cs = crystalFrame("m-3m", mineral="test")
  n1, n2 = 160, 120
  x, y = np.meshgrid(np.arange(n1, dtype=float), np.arange(n2, dtype=float))
  x, y = x.ravel(), y.ravel()
  def disc(cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

  def rect(cx, cy, w, h, t=0.0):
    u = np.cos(t) * (x - cx) + np.sin(t) * (y - cy)
    v = -np.sin(t) * (x - cx) + np.cos(t) * (y - cy)
    return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)

  def polygon(cx, cy, r, k, phi0=0.0):
    inside = np.ones(len(x), dtype=bool)
    for j in range(k):
      phi = phi0 + 2 * np.pi * j / k
      inside &= (x - cx) * np.cos(phi) + (y - cy) * np.sin(phi) <= r * np.cos(np.pi / k)
    return inside

  clover = np.zeros(len(x), dtype=bool)
  for phi in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3):
    clover |= disc(100 + 7 * np.cos(phi), 57 + 7 * np.sin(phi), 8)
  shapes = [disc(20, 20, 12), disc(55, 20, 14), disc(90, 20, 14), ((x - 130) / 20) ** 2 + ((y - 20) / 8) ** 2 <= 1,
            rect(20, 57, 30, 12), rect(60, 57, 30, 12, 30 * degree), clover,
            disc(140, 57, 14) & ~((np.abs(y - 57) <= 0.75) & (x >= 140)),
            rect(16, 97, 10, 30) | rect(24, 87, 26, 10), polygon(55, 97, 16, 3, np.pi / 2), polygon(90, 97, 14, 6),
            disc(135, 97, 14) & ~disc(147, 97, 9)]
  holes = disc(55, 20, 4) | disc(90, 20, 8)
  shape = np.zeros(len(x), dtype=int)
  for s, mask in reversed(list(enumerate(shapes, start=1))):
    shape[mask] = s
  rot = rotation.byAxisAngle(vector3d(0, 0, 1), 7 * degree * shape)
  pos = np.column_stack([x, y, np.zeros(len(x))])
  return EBSD(pos, rot, np.where(holes, 0, 1), [cs]).gridify()


def _testGrains(files):
  from ..maps.calcgrains import calcGrains
  return calcGrains(testGrainsMap(), 5 * degree)


_dataset("aachen", "EBSD", "two-phase iron and magnesium map", "aachen", _aachen)
_dataset("sharp", "EBSD", "calcite orientation map", "sharp", _sharp)
_dataset("small", "EBSD", "the Forsterite map cropped to a 3000 x 3000 um region", "forsterite", _small)
_dataset("csl", "EBSD", "iron map for the coincidence site lattice examples", "csl", _csl)
_dataset("mylonite", "EBSD", "multiphase mylonite map", "mylonite", _mylonite)
_dataset("epidote", "EBSD", "multiphase epidote map", "epidote", _epidote)
_dataset("forsterite", "EBSD", "multiphase Forsterite map", "forsterite", _oxford)
_dataset("olivine", "EBSD", "olivine optical map in the prepared reference frame", "olivine", _olivine)
_dataset("twins", "EBSD", "magnesium map for the twinning examples", "twins", _oxford)
_dataset("copper", "EBSD", "copper map, EDAX setting 1", "copper", lambda p: loadEBSD(p, setting=1))
_dataset("single", "EBSD", "single grain aluminium map", "single", _aluminium)
_dataset("alu", "EBSD", "polycrystalline aluminium map", "alu", _aluminium)
_dataset("titanium", "EBSD", "alpha titanium map with X||a", "titanium", _titanium)
_dataset("ferrite", "EBSD", "ferrite map, EDAX setting 1", "ferrite", lambda p: loadEBSD(p, setting=1))
_dataset("dc06", "EBSD", "ferritic steel after two percent uniaxial tension, EDAX setting 2", "dc06", lambda p: loadEBSD(p, setting=2))
_dataset("martensite", "EBSD", "martensite map with proper crystal groups", "martensite", _martensite)
_dataset("emsland", "EBSD", "Emsland plessite map", "emsland", _oxford)
_dataset("alphaBetaTitanium", "EBSD", "alpha and beta titanium map with proper crystal groups", "alphabetatitanium", _alphaBetaTitanium)
_dataset("EMSphinx", "EBSD", "EMSphInx steel map with named iron and cobalt phases", "emsphinx", _emsphinx)
_dataset("testgrains", "grain2d", "artificial grain shapes for the shape parameters", "", _testGrains)
_dataset("dubna", "PoleFigure", "quartz pole figures with superposition coefficients", "dubna", _dubna)
_dataset("geesthacht", "PoleFigure", "Geesthacht diffraction pole figures", "geesthacht", _geesthacht)
_dataset("steel", "PoleFigure", "steel pole figures turned by 90 degree", "steel", _steel)
_dataset("ptx", "PoleFigure", "three orthorhombic pole figures of a Siemens diffractometer", "ptx", _ptx)
_dataset("zncuti", "PoleFigure", "ZnCuTi sheet pole figures corrected by a texture-free reference", "zncuti", _zncuti)
_dataset("aachenexp", "PoleFigure", "Aachen experimental pole figures", "aachenexp", _aachenexp)
_dataset("dubnaODF", "SO3Fun", "the ODF reconstructed from the Dubna quartz pole figures", "dubna", _dubnaODF)
_dataset("eclogite", "EBSD", "sparse eclogite map whose positions collide on a lattice, kept as a list", "eclogite", lambda p: loadEBSD(p))
_dataset("vectors", "vector3d", "a thousand directions as polar and azimuth angles", "vectors",
         lambda p: loadVector3d(p, columnNames=["polar angle", "azimuth angle"])[0])
_dataset("dubnaValues", "orientation", "values of the Dubna ODF at 19848 orientations of a 5 degree grid", "dubnaValues",
         lambda p: loadOrientation(p, columnNames=["phi1", "Phi", "phi2", "values"]))
_dataset("smiley", "vector3d", "777 directions with the values of the smiley function", "smiley",
         lambda p: loadVector3d(p, columnNames=["x", "y", "z", "values"]))
_dataset("tongue", "orientation", "382 quartz orientations of the Tongue quartzite as Bunge Euler angles", "tongue",
         lambda p: loadOrientation(p, crystalFrame.load("quartz.cif"), columnNames=["phi1", "Phi", "phi2"])[0])
_dataset("vpsc", "VPSC", "the texture and slip activity history of a VPSC olivine run", "vpsc", _vpsc)
_dataset("trueEbsdWCCoSmall", "EBSD", "WC-Co map with four forescatter images of the same area, for TrueEBSD", "trueebsdwccosmall",
         _trueEbsdWCCoSmall)
_dataset("xnovo", "EBSD3", "a simulated multiphase xnovo volume, a cylinder of nine phases", "xnovo", _volume)
_dataset("SmallIN100", "EBSD3", "the IN100 nickel superalloy volume of DREAM.3D with its grain mesh", "smallin100", _volume)
_dataset("odf", "SO3Fun", "an ODF tabulated on a Bunge Euler grid, read as values", "odf", _odf)

# the room pressure stiffness of San Carlos olivine (Abramson et al. 1997) in its source setting
def _olivine1997(path):
  from ..tensors.tensor import stiffnessTensor
  return stiffnessTensor.load(path, crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="Olivine"))


# the piezoelectric modulus of right-handed quartz in pC/N, a Voigt table with the doubled shear columns
def _quartzPiezo(path):
  from ..tensors.tensor import tensor
  cs = crystalFrame("32", [4.916, 4.916, 5.4054], "X||a*", "Z||c", mineral="Quartz")
  return tensor.load(path, cs, rank=3, doubleConvention=True, name="piezoelectric strain", unit="pC/N")


# the simulated spherical Kikuchi master pattern of quartz: a MATLAB v7.3 object file whose
# one large array is the harmonic coefficients, read with h5py into the frame they are written
# in, the quartz cell of the file with the a axis along x
def _quartzPattern(path):
  import h5py
  from ..functions.s2fun import S2FunHarmonic
  with h5py.File(path, "r") as f:
    arrays = [np.array(v).ravel() for v in f["#refs#"].values() if isinstance(v, h5py.Dataset) and v.dtype.kind == "f"]
  fhat = max(arrays, key=len)
  cs = crystalFrame("321", [4.914, 4.914, 5.406], "X||a", "Z||c", mineral="Quartz")
  return S2FunHarmonic(fhat, frame=cs)


_dataset("quartzPiezo", "tensor", "the piezoelectric modulus of right-handed quartz (Frondel 1962)", "quartzpiezo",
         _quartzPiezo)
_dataset("olivine1997", "tensor", "the stiffness tensor of San Carlos olivine at room pressure", "olivine1997", _olivine1997)
_dataset("quartzPattern", "S2Fun", "a spherical Kikuchi master pattern of quartz", "quartzpattern", _quartzPattern)


__all__ = ["mtexdata", "mtexdatafile", "mtexDataPath"]

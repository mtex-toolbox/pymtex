import os
import subprocess
import sys

import numpy as np

from mtex import crystalFrame, degree, inspectEBSD, loadEBSD, notIndexedFrame, phaseList, rotation
from mtex.io.importscript import buildImportWizardScript, frameKey, frameLiteral, loadCall, phasesDiffer, shortestFloat

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shortest_float():
  assert shortestFloat(4.0399999618530273) == "4.04"
  assert shortestFloat(90.0) == "90" and shortestFloat(5.98) == "5.98" and shortestFloat(1 / 3) == "0.33333333"


def test_frame_literal():
  cs = crystalFrame("mmm", [4.756, 10.207, 5.98], mineral="Forsterite")
  assert frameLiteral(cs) == "crystalFrame('mmm', [4.756, 10.207, 5.98], [90, 90, 90], mineral='Forsterite')"
  cs = crystalFrame("6/mmm", [3.2, 3.2, 5.2], np.array([90, 90, 120]) * degree, "X||a", "Z||c*", mineral="Mg", color="red")
  assert frameLiteral(cs) == "crystalFrame('6/mmm', [3.2, 3.2, 5.2], [90, 90, 120], 'X||a', 'Z||c*', mineral='Mg', color='red')"
  assert frameLiteral(notIndexedFrame()) == "notIndexedFrame()"
  # the literal reads back as the frame
  ns = {"crystalFrame": crystalFrame, "notIndexedFrame": notIndexedFrame}
  back = eval(frameLiteral(cs), ns)
  assert back.pointGroup == "6/mmm" and np.allclose(back.abc, [3.2, 3.2, 5.2]) and back.color == "red"


def test_phases_differ():
  a = phaseList([crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")])
  b = phaseList([crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Al")])
  assert not phasesDiffer(a, b) and not phasesDiffer(a, [frameKey(f) for f in b.frames])
  c = phaseList([crystalFrame("m-3m", [4.05, 4.05, 4.05], mineral="Aluminium")])
  assert phasesDiffer(a, c) and phasesDiffer(a, None)


def test_script_carries_only_what_changed():
  path = os.path.join(FIXTURES, "Forsterite_cut.ctf")
  info = inspectEBSD(path)
  text = buildImportWizardScript(path, phases=info.phases, headerPhases=info.phases, mapConvention="y↓→x", eulerConvention="x←↓y",
                                 vendorCorrection=info.eulerCorrection)
  assert "csList" not in text and "eulerCorrection" not in text and "dataSet" not in text
  assert "ebsd = EBSD.load(fname)" in text and "plottingConvention.default('y↓→x')" in text
  assert "plot(ebsd['Forsterite'], ebsd['Forsterite'].orientations)" in text
  # another Euler frame, a data set of several and a renamed variable
  text = buildImportWizardScript(path, phases=info.phases, headerPhases=info.phases, mapConvention="y↓→x", eulerConvention="y↑→x",
                                 vendorCorrection=info.eulerCorrection, dataSet=1, dataSets=["a", "b"], variable="fo")
  assert "fo = EBSD.load(fname, dataSet=1, eulerCorrection=eulerCorrection)" in text
  assert "eulerCorrection = plottingConvention('y↓→x').rot * plottingConvention('y↑→x').rot.inv()" in text
  assert loadCall(path, 1, ["a", "b"], "fo") == f"fo = EBSD.load({os.path.abspath(path)!r}, dataSet=1)"


# executing what the script writes gives back the map the direct load gives
def test_script_round_trip():
  path = os.path.join(FIXTURES, "Forsterite_cut.ctf")
  info = inspectEBSD(path)
  text = buildImportWizardScript(path, phases=info.phases, headerPhases=info.phases, mapConvention="y↓→x", eulerConvention="y↑→x",
                                 vendorCorrection=info.eulerCorrection)
  ns = {}
  exec(text.replace("plot(ebsd", "(ebsd"), ns)
  direct = loadEBSD(path)
  ebsd = ns["ebsd"]
  assert ebsd.size == direct.size and np.array_equal(ebsd.phaseId, direct.phaseId)
  assert np.allclose(ebsd.pos.data, direct.pos.data)
  # the correction written is the map frame against the Euler frame: ij against y↑→x is the half turn about x
  assert float(ebsd.EulerCorrection.angle(rotation.byEuler(0.0, np.pi, 0.0, "Bunge"))) < 1e-12


# an edited cell is reproduced in a fresh session, where the script's frames are registered
# before the file's are read
def test_edited_phase_reproduced_in_fresh_session(tmp_path):
  path = os.path.join(FIXTURES, "Forsterite_cut.ctf")
  info = inspectEBSD(path)
  frames = list(info.phases.frames)
  frames[1] = crystalFrame("mmm", [4.8, 10.207, 5.98], mineral="Forsterite", intern=False)
  text = buildImportWizardScript(path, phases=phaseList(frames), headerPhases=info.phases, mapConvention="y↓→x")
  assert "crystalFrame('mmm', [4.8, 10.207, 5.98], [90, 90, 90], mineral='Forsterite')" in text
  script = tmp_path / "import_test.py"
  script.write_text(text.replace("plot(ebsd", "(ebsd") + "\nprint(ebsd.CSList.frames[1].abc[0])\n", encoding="utf-8")
  out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT)
  assert out.returncode == 0, out.stderr
  assert out.stdout.strip().endswith("4.8")


def test_a_volume_is_loaded_by_EBSD3():
  text = buildImportWizardScript("/data/volume.dream3d", volume=True, eulerConvention="y↑→x")
  assert "from mtex import EBSD3," in text and "# the volume\nebsd = EBSD3.load(fname)" in text and "eulerCorrection" not in text
  # the reader of a volume takes no correction, so the volume is turned once it is read
  text = buildImportWizardScript("/data/volume.dream3d", volume=True, mapConvention="y↑→x", eulerConvention="y↓→x")
  assert "ebsd = EBSD3.load(fname)\nebsd = ebsd.rotate(eulerCorrection, keepXY=True)" in text
  assert loadCall("/data/volume.dream3d", volume=True) == f"ebsd = EBSD3.load({os.path.abspath('/data/volume.dream3d')!r})"
  assert loadCall("/data/map.ctf") == f"ebsd = EBSD.load({os.path.abspath('/data/map.ctf')!r})"

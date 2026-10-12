# the Python that reproduces an import the wizard settled (MATLAB buildImportWizardScript,
# Julia importScript; docs/plans/wizard.md W1)
#
#   text = buildImportWizardScript('map.ctf', phases=ebsd.CSList, headerPhases=info.phases, ...)
#
# A pure function of what the wizard settled, so it is tested by executing what it writes.
# The crystallography is assigned after the load and only where it differs from what the
# file declares, the data set only where the file holds several, the Euler correction only
# where it differs from the vendor's; a script that changed nothing carries none of them.
from __future__ import annotations

import os
from datetime import date

import numpy as np

from ..geometry.frames import closestAlignment, crystalFrame
from ..geometry.plottingconvention import plottingConvention
from ..geometry.units import degree
from ..maps.phaselist import phaseList


def buildImportWizardScript(path, phases=None, headerPhases=None, dataSet=None, dataSets=(), mapConvention="y↑→x",
                            eulerConvention=None, vendorCorrection=None, variable="ebsd", drawn=None):
  """the script of an import: the file, the plotting convention, the correction, the phases, a plot

    buildImportWizardScript('map.ctf')                             the plain load
    buildImportWizardScript('map.oh5', dataSet=1, dataSets=sets)   a map of a project file
    buildImportWizardScript('map.ang', phases=csList, headerPhases=info.phases)
  """
  name = variable or "ebsd"
  lines = [f"# the import of {os.path.basename(path)}, written by the MTEX import wizard on {date.today().isoformat()}",
           "from mtex import EBSD, crystalFrame, degree, notIndexedFrame, plot, plottingConvention", ""]
  changed = phases is not None and phasesDiffer(phases, headerPhases)
  if changed:
    lines += ["# the crystal symmetry, lattice angles in degree", "csList = [" + ",\n          ".join(frameLiteral(f) for f in phases.frames) + "]", ""]
  lines += ["# the plotting convention of the map", f"plottingConvention.default({mapConvention!r})", ""]
  lines += ["# the file", f"fname = {os.path.abspath(path)!r}", ""]
  args = ["fname"]
  if dataSets is not None and len(dataSets) > 1 and dataSet is not None:
    args.append(f"dataSet={int(dataSet)}")
  if eulerConvention is not None:
    correction = plottingConvention(mapConvention).rot * plottingConvention(eulerConvention).rot.inv()
    if vendorCorrection is None or float(correction.angle(vendorCorrection)) > 1e-9:
      lines += ["# the frame the Euler angles are written in, against the frame of the map",
                f"eulerCorrection = plottingConvention({mapConvention!r}).rot * plottingConvention({eulerConvention!r}).rot.inv()", ""]
      args.append("eulerCorrection=eulerCorrection")
  lines += ["# the map", f"{name} = EBSD.load({', '.join(args)})"]
  if changed:
    lines.append(f"{name}.CSList = csList")
  lines.append("")
  mineral = plotPhase(phases, drawn)
  if mineral is not None:
    lines += ["# a look at the orientations of the largest phase", f"plot({name}[{mineral!r}], {name}[{mineral!r}].orientations)", ""]
  return "\n".join(lines)


def loadCall(path, dataSet=None, dataSets=(), variable="ebsd"):
  """the one line load, for the clipboard"""
  args = [repr(os.path.abspath(path))]
  if dataSets is not None and len(dataSets) > 1 and dataSet is not None:
    args.append(f"dataSet={int(dataSet)}")
  return f"{variable or 'ebsd'} = EBSD.load({', '.join(args)})"


# ------------------------------------------------------------------ the pieces

def phasesDiffer(phases, header):
  """whether the phases differ from what the file declared, in name, group, cell, setup or colour;
  the header side is a phase list or the value snapshot frameKeys took of one"""
  if header is None:
    return True
  keys = [frameKey(f) for f in header.frames] if isinstance(header, phaseList) else list(header)
  return [frameKey(f) for f in phases.frames] != keys


def frameKey(f):
  """the value of a frame as a tuple, what an edit changes"""
  if not f.isIndexed:
    return (False, f.name, _color(f))
  return (True, f.name, f.pointGroup, tuple(round(float(v), 6) for v in f.abc), tuple(round(float(v) / degree, 6) for v in f.abg),
          tuple(closestAlignment(f)), _color(f))


def plotPhase(phases, drawn):
  """the phase the closing plot shows: the largest drawn indexed phase by name, the first indexed one otherwise"""
  if phases is None:
    return None
  ids = [k for k, f in enumerate(phases.frames) if f.isIndexed and (drawn is None or drawn.get(k, True))]
  if not ids:
    return None
  if drawn is not None and any(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in drawn.values()):
    ids.sort(key=lambda k: -int(drawn.get(k, 0)))
  return phases.frames[ids[0]].name


def frameLiteral(f):
  """crystalFrame('mmm', [4.756, 10.207, 5.98], [90, 90, 90], 'X||a', 'Z||c*', mineral='Forsterite')"""
  if not f.isIndexed:
    return f"notIndexedFrame({f.name!r})" if f.name != "notIndexed" else "notIndexedFrame()"
  abc = "[" + ", ".join(shortestFloat(v) for v in f.abc) + "]"
  abg = "[" + ", ".join(shortestFloat(v / degree) for v in f.abg) + "]"
  args = [repr(f.pointGroup), abc, abg]
  # the setup is written where it is not the one the constructor would choose by itself
  setup = closestAlignment(f)
  if setup != closestAlignment(crystalFrame(f.pointGroup, f.abc, f.abg, intern=False, check=False)):
    args += [repr(s) for s in setup]
  args.append(f"mineral={f.name!r}")
  c = _color(f)
  if c is not None:
    args.append(f"color={c!r}")
  return "crystalFrame(" + ", ".join(args) + ")"


# the colour a frame carries of its own, as a name or an RGB tuple
def _color(f):
  c = f.color
  if c is None:
    return None
  if isinstance(c, str):
    return c
  a = np.asarray(c, dtype=float).reshape(-1)
  if np.any(np.isnan(a)):
    return None
  return tuple(float(shortestFloat(v)) for v in a[:3])


def shortestFloat(x):
  """the shortest decimal that reads back as the same single precision number, so a lattice
  parameter read off an HDF5 file prints as 4.04 and not as 4.0399999618530273"""
  x = float(x)
  if abs(x - round(x)) < 1e-9 * max(1.0, abs(x)) and abs(x) < 1e12:
    return str(int(round(x)))
  f32 = np.float32(x)
  for digits in range(1, 17):
    s = f"{x:.{digits}g}"
    if np.float32(float(s)) == f32:
      return np.format_float_positional(float(s), trim="-") if "e" in s else s
  return repr(x)


__all__ = ["buildImportWizardScript", "loadCall"]

# the Python that reproduces an import the wizard settled (MATLAB buildImportWizardScript,
# Julia importScript; docs/plans/wizard.md W1)
#
#   text = buildImportWizardScript('map.ctf', phases=ebsd.CSList, ...)
#
# A pure function of what the wizard settled, so it is tested by executing what it writes.
# The crystallography is always written and assigned after the load, so that the script states
# the phases it was set up with whatever the reader makes of the header later; the data set only
# where the file holds several, the Euler correction only where it differs from the vendor's.
from __future__ import annotations

import os
from datetime import date

import numpy as np

from ..geometry.frames import closestAlignment, crystalFrame
from ..geometry.plottingconvention import plottingConvention
from ..geometry.units import degree


def buildImportWizardScript(path, phases=None, dataSet=None, dataSets=(), mapConvention="y↑→x",
                            eulerConvention=None, vendorCorrection=None, variable="ebsd", drawn=None, volume=False):
  """the Python script of an import as the import wizard sets it up: the file, the plotting
  convention, the correction of the Euler angles, the phases and a first plot

    script = buildImportWizardScript('map.ctf')                               # the plain load
    script = buildImportWizardScript('map.h5oina', dataSet=2, dataSets=sets)  # a map of a project file
    script = buildImportWizardScript('map.ang', phases=ebsd.CSList)                       # the phases stated
    script = buildImportWizardScript('map.ctf', mapConvention='y↑→x', eulerConvention='y↓→x')  # a correction
    script = buildImportWizardScript('volume.dream3d', volume=True)          # a volume, read by EBSD3.load

  The script sets the plotting convention of the map, loads the file with EBSD.load or
  EBSD3.load, turns the Euler angles where their frame differs from the map's, replaces the
  phases with the ones it states, every one, and plots the orientations of the
  largest phase. It runs as it is and reproduces the import. The window's Script tab shows
  it; loadCall gives the one line load.

  Input
    path - the file
  Output
    script - str
  Keywords
    phases - phaseList, the phases as edited, written out and assigned after the load; lattice
        values as the shortest decimal that reads back as the file's single precision
    dataSet - the number of the map of a file with several
    dataSets - the maps of the file; a dataSet is written only when there are several
    mapConvention - the plotting convention of the map
    eulerConvention - the convention the Euler angles are written in; a correction is
        written where it differs from the map's
    vendorCorrection - the correction the reader applies anyway, which is then not written
    variable - the name of the variable the map is assigned to
    drawn - the phases the plot may show, by index, True or False or a pixel count
    volume - the file is a volume, read by EBSD3.load
  See also
    importWizard, loadCall, inspectEBSD, EBSD.load, EBSDImport"""
  name = variable or "ebsd"
  lines = [f"# the import of {os.path.basename(path)}, written by the MTEX import wizard on {date.today().isoformat()}",
           f"from mtex import {'EBSD3' if volume else 'EBSD'}, crystalFrame, degree, notIndexedFrame, plot, plottingConvention", ""]
  stated = phases is not None
  if stated:
    lines += ["# the crystal symmetry, lattice angles in degree", "csList = [" + ",\n          ".join(frameLiteral(f) for f in phases.frames) + "]", ""]
  lines += ["# the plotting convention of the map", f"plottingConvention.default({mapConvention!r})", ""]
  lines += ["# the file", f"fname = {os.path.abspath(path)!r}", ""]
  args = ["fname"]
  if dataSets is not None and len(dataSets) > 1 and dataSet is not None:
    args.append(f"dataSet={int(dataSet)}")
  turned = False
  if eulerConvention is not None:
    correction = plottingConvention(mapConvention).rot * plottingConvention(eulerConvention).rot.inv()
    # a volume is read as its file states it
    if (vendorCorrection is None and not volume) or float(correction.angle() if vendorCorrection is None else correction.angle(vendorCorrection)) > 1e-9:
      lines += ["# the frame the Euler angles are written in, against the frame of the map",
                f"eulerCorrection = plottingConvention({mapConvention!r}).rot * plottingConvention({eulerConvention!r}).rot.inv()", ""]
      # the reader of a volume takes no correction: its orientations are turned once it is read
      turned = volume
      if not volume:
        args.append("eulerCorrection=eulerCorrection")
  lines += ["# the volume" if volume else "# the map", f"{name} = {'EBSD3' if volume else 'EBSD'}.load({', '.join(args)})"]
  if turned:
    lines.append(f"{name} = {name}.rotate(eulerCorrection, keepXY=True)")
  if stated:
    lines.append(f"{name}.CSList = csList")
  lines.append("")
  mineral = plotPhase(phases, drawn)
  if mineral is not None:
    lines += ["# a look at the orientations of the largest phase", f"plot({name}[{mineral!r}], {name}[{mineral!r}].orientations)", ""]
  return "\n".join(lines)


def loadCall(path, dataSet=None, dataSets=(), variable="ebsd", volume=False):
  """the one line load of a file, for the clipboard

    line = loadCall('map.ctf')                                  # ebsd = EBSD.load('/.../map.ctf')
    line = loadCall('map.h5oina', dataSet=2, dataSets=[1, 2, 3])  # a map of a project file
    line = loadCall('volume.dream3d', variable='vol', volume=True)  # a volume

  The file is written with its absolute path. The import wizard copies it to the clipboard;
  buildImportWizardScript writes the whole script.

  Input
    path - the file
  Output
    line - str
  Keywords
    dataSet - the number of the map of a file with several
    dataSets - the maps of the file; a dataSet is written only when there are several
    variable - the name of the variable the map is assigned to
    volume - the file is a volume, read by EBSD3.load
  See also
    buildImportWizardScript, importWizard, EBSD.load"""
  args = [repr(os.path.abspath(path))]
  if dataSets is not None and len(dataSets) > 1 and dataSet is not None:
    args.append(f"dataSet={int(dataSet)}")
  return f"{variable or 'ebsd'} = {'EBSD3' if volume else 'EBSD'}.load({', '.join(args)})"


# ------------------------------------------------------------------ the pieces

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

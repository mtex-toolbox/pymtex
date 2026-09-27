# the crystal frame of a CIF file (MATLAB loadCIF, crystalFrame.load): the space group
# symbol, the cell and the mineral name read off the tags; the file is looked up as given,
# then among the files MTEX ships, mtexCifPath, then fetched from the Crystallography Open
# Database when the name is a COD number
#
#   cs = crystalFrame.load('Al-Aluminum.cif')
#   cs = crystalFrame.load('5000035.cif')
#   cs = crystalFrame.load(5000035)
from __future__ import annotations

import os
import re

from ..geometry.frames import crystalFrame
from ..geometry.units import degree

MINERAL_TAGS = ("_chemical_name_mineral", "_chemical_name_systematic", "_chemical_formula_structural", "_chemical_formula_sum")
GROUP_TAGS = ("_symmetry_space_group_name_H-M", "_space_group_name_H-M_alt", "_symmetry_point_group_name_H-M", "_symmetry_cell_setting")


def mtexCifPath():
  """the directory of the CIF files shipped with MTEX, where crystalFrame.load looks up a name"""
  return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geometry", "data", "cif")


def cifFiles():
  """the names of the CIF files shipped with MTEX, sorted"""
  return sorted(f for f in os.listdir(mtexCifPath()) if f.endswith(".cif"))


# the value of a tag: the first line starting with it, quotes stripped
def _token(lines, tag):
  for line in lines:
    if line.startswith(tag) and (len(line) == len(tag) or line[len(tag)].isspace()):
      return line[len(tag):].strip().strip("'\"").strip()
  return ""


def _number(text):
  m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
  return float(m.group()) if m else None


def _resolve(fname):
  if isinstance(fname, (int, float)):
    fname = f"{int(fname)}.cif"
  fname = str(fname)
  if not fname.lower().endswith(".cif"):
    fname += ".cif"
  if os.path.isfile(fname):
    return fname
  shipped = os.path.join(mtexCifPath(), os.path.basename(fname))
  if os.path.isfile(shipped):
    return shipped
  stem = os.path.splitext(os.path.basename(fname))[0]
  if stem.isdigit():
    import urllib.request
    url = f"http://www.crystallography.net/cif/{stem}.cif"
    print(f"CIF-File from Crystallography Open Database\n> download : {url}")
    with urllib.request.urlopen(url, timeout=30) as r:
      data = r.read()
    with open(shipped, "wb") as f:
      f.write(data)
    print(f"> copied to: {shipped}")
    return shipped
  raise FileNotFoundError(f"no CIF file {fname!r}; the shipped ones are " + ", ".join(cifFiles()))


def loadCIF(fname):
  """the crystal frame a CIF file describes: its space group, cell and mineral; a Bruker .phl
  phase list gives one frame per phase"""
  if str(fname).lower().endswith(".phl"):
    return loadPHL(fname)
  path = _resolve(fname)
  with open(path, encoding="utf-8", errors="replace") as f:
    lines = [l.strip() for l in f]
  mineral = next((_token(lines, t) for t in MINERAL_TAGS if _token(lines, t)), "")
  group = next((_token(lines, t) for t in GROUP_TAGS if _token(lines, t)), "")
  if not group:
    raise ValueError(f"{path} names no space group")
  axes = [_number(_token(lines, "_cell_length_" + a)) for a in "abc"]
  angles = [_number(_token(lines, "_cell_angle_" + a)) for a in ("alpha", "beta", "gamma")]
  if any(x is None for x in axes):
    axes = [1.0, 1.0, 1.0]
  if any(x is None for x in angles):
    angles = [90.0, 90.0, 90.0]
  return crystalFrame(group, axes, [a * degree for a in angles], mineral=mineral or None)


def loadPHL(fname):
  """a Bruker phase list: XML with one TEBSDExtPhaseEntry per phase, the space group by its
  number, the cell under Cell/Dim and Cell/Angles, the colour as r,g,b"""
  import xml.etree.ElementTree as ET
  path = fname if os.path.isfile(fname) else os.path.join(mtexCifPath(), os.path.basename(fname))
  if not os.path.isfile(path):
    raise FileNotFoundError(f"no phase list {fname!r}")
  root = ET.parse(path).getroot()
  frames = []
  for entry in root.iter("ClassInstance"):
    if entry.get("Type") != "TEBSDExtPhaseEntry":
      continue
    phase = entry.find("TEBSDPhaseEntry")
    number = int(phase.findtext("IT", "0"))
    dim = [float(x) for x in phase.findtext("Cell/Dim", "1,1,1").split(",")]
    angles = [float(x) * degree for x in phase.findtext("Cell/Angles", "90,90,90").split(",")]
    color = entry.findtext("Color")
    color = tuple(int(c) / 255 for c in color.split(",")) if color else None
    frames.append(crystalFrame(dim, angles, spaceId=number, mineral=entry.get("Name"), color=color))
  return frames


__all__ = ["loadCIF", "loadPHL", "mtexCifPath", "cifFiles"]

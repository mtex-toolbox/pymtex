# orientation lists as text files (MATLAB @quaternion/export, @orientation/export_VPSC)
#
#   export(ori, fname), export(ori, fname, 'Bunge'), export(ori, fname, S, 'Bunge')
#   export(ori, fname, quaternion=True), export(ori, fname, radians=True)
#   export_VPSC(ori, fname, 'Bunge', weights=w)
#
# A table with one row per orientation: the Euler angles in degrees under their names,
# or the four quaternion components, and one more column per entry of a dictionary of
# values; the VPSC texture file with its four header lines and the weights normalised to
# one.
from __future__ import annotations

import numpy as np

from ..geometry.units import degree

EULER_NAMES = {"Bunge": ("phi1", "Phi", "phi2"), "Matthies": ("alpha", "beta", "gamma"), "ZYZ": ("alpha", "beta", "gamma"),
               "Roe": ("Psi", "Theta", "Phi"), "Kocks": ("Psi", "Theta", "phi")}


# the Euler convention among positional strings, Bunge unless named
def _convention(args):
  for a in args:
    if isinstance(a, str) and a in EULER_NAMES:
      return a
  return "Bunge"


def _columns(ori, args, radians, quaternion):
  if quaternion:
    return ori.data.reshape(-1, 4), ("a", "b", "c", "d")
  conv = _convention(args)
  e = np.asarray(ori.Euler(conv)).reshape(3, -1).T
  return (e if radians else e / degree), EULER_NAMES[conv]


def exportOrientation(ori, fname, *args, radians=False, quaternion=False):
  """export(ori, fname, 'Bunge'): the Euler angles of the orientations in degrees as a table,
  with a dictionary among the arguments as further columns"""
  d, names = _columns(ori.flatten(), args, radians, quaternion)
  names = list(names)
  for a in args:
    if isinstance(a, dict):
      for key, value in a.items():
        names.append(str(key))
        d = np.concatenate([d, np.asarray(value, dtype=float).reshape(-1, 1)], axis=1)
  with open(fname, "w", encoding="utf-8") as f:
    f.write(" ".join(f"{n:>10}" for n in names) + "\n")
    for row in d:
      f.write(" ".join(f"{x:10.5g}" for x in row) + "\n")


# the letter VPSC uses for the convention
VPSC_LETTER = {"Bunge": "B", "Kocks": "K", "Roe": "R"}


def export_VPSC(ori, fname, *args, weights=None, radians=False):
  """export_VPSC(ori, fname, 'Bunge', weights=w): the VPSC texture file of the orientations,
  the weights normalised to sum to one"""
  o = ori.flatten()
  d, _ = _columns(o, args, radians, False)
  w = np.ones(o.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  w = w / np.sum(w)
  with open(fname, "w", encoding="utf-8") as f:
    f.write(f"texture exported by MTEX\n\n\n{VPSC_LETTER[_convention(args)]} {o.size}\n")
    for row, wk in zip(d, w):
      f.write(f"{row[0]:7.2f} {row[1]:7.2f} {row[2]:7.2f} {wk:11.7f}\n")


__all__ = ["exportOrientation", "export_VPSC"]

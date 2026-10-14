# orientation lists as text files (MATLAB @quaternion/export, @orientation/export_VPSC)
#
#   export(ori, fname), export(ori, fname, 'Bunge'), export(ori, fname, S, 'Bunge')
#   export(ori, fname, quaternion=True), export(ori, fname, radians=True)
#   export(ori, fname, 'Bunge', interface='VPSC', weights=w)
#
# A table with one row per orientation: the Euler angles in degrees under their names,
# or the four quaternion components, and one more column per entry of a dictionary of
# values; the VPSC texture file with its four header lines and the weights normalised to
# one.
from __future__ import annotations

import os

import numpy as np

from ..geometry.units import degree
from .table import writeTable

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


def exportOrientation(ori, fname, *args, radians=False, quaternion=False, interface=None, weights=None):
  """orientations written to a text file, as a table of Euler angles or quaternions with further
  columns, or as a VPSC texture file; export(ori, fname) calls it

    export(ori, fname)                                          # Bunge Euler angles in degree
    export(ori, fname, 'Matthies', radians=True)                # another convention, in radians
    export(ori, fname, quaternion=True)                         # the four quaternion components
    export(ori, fname, {'area': grains.area})                   # further columns
    export(ori, fname, 'Bunge', interface='VPSC', weights=w)    # the VPSC texture file

  The table has one row per orientation, the columns named in a header line: phi1, Phi, phi2
  for Bunge, alpha, beta, gamma for Matthies and ZYZ, Psi, Theta, Phi for Roe, Psi, Theta, phi
  for Kocks, or a, b, c, d; every entry of a dictionary among the arguments adds a column, as
  MATLAB's struct does. loadOrientation reads it back. The VPSC file, MATLAB's export_VPSC,
  has four header lines and the weights, normalised to sum to one, in the fourth column; it
  takes the conventions Bunge, Kocks and Roe.

  Input
    ori - orientation
    fname - the name of the file
    args - the Euler convention, 'Bunge', 'Matthies', 'ZYZ', 'Roe' or 'Kocks', and dictionaries
        of further columns, one value per orientation
  Keywords
    radians - the angles in radians instead of degree
    quaternion - the quaternion components instead of Euler angles
    interface - None for the table, 'VPSC' for the VPSC texture file
    weights - the weights of the orientations in the VPSC file, equal by default
  See also
    loadOrientation, export, quaternion.Euler"""
  if interface == "VPSC":
    return _export_VPSC(ori, fname, *args, weights=weights, radians=radians)
  if interface is not None:
    raise ValueError(f"the interface of an orientation export is None or 'VPSC', not {interface!r}")
  d, names = _columns(ori.flatten(), args, radians, quaternion)
  extra = [(str(key), value) for a in args if isinstance(a, dict) for key, value in a.items()]
  names = list(names) + [key for key, _ in extra]
  d = np.column_stack([d] + [np.asarray(value, dtype=float).reshape(-1, 1) for _, value in extra])
  writeTable(os.fspath(fname), names, d.T)


# the letter VPSC uses for the convention
VPSC_LETTER = {"Bunge": "B", "Kocks": "K", "Roe": "R"}


# _export_VPSC(ori, fname, 'Bunge', weights=w): the VPSC texture file of the orientations,
# the weights normalised to sum to one
def _export_VPSC(ori, fname, *args, weights=None, radians=False):
  o = ori.flatten()
  d, _ = _columns(o, args, radians, False)
  w = np.ones(o.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  w = w / np.sum(w)
  n = min(len(d), len(w))
  with open(fname, "w", encoding="utf-8") as f:
    f.write(f"texture exported by MTEX\n\n\n{VPSC_LETTER[_convention(args)]} {o.size}\n")
    np.savetxt(f, np.column_stack([d[:n, :3], w[:n]]), fmt="%7.2f %7.2f %7.2f %11.7f")


__all__ = ["exportOrientation"]

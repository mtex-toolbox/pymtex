# the point group names: the 32 classes in every setting, the Laue classes, the space groups
import re

# the named groups: the 32 crystallographic classes in every setting MTEX distinguishes, in
# MTEX's order, plus the icosahedral two; each with its Schoenflies name and the aliases
# the constructor accepts
POINT_GROUP_NAMES = [
  ("1", "C1"), ("-1", "Ci", "S2", "triclinic"),
  ("211", "C2"), ("m11", "Cs", "C1h"), ("2/m11", "C2h"),
  ("121", "C2", "2"), ("1m1", "Cs", "m"), ("12/m1", "C2h", "2/m"),
  ("112", "C2"), ("11m", "Cs"), ("112/m", "C2h", "monoclinic"),
  ("222", "D2", "22", "V"), ("2mm", "C2v"), ("m2m", "C2v"), ("mm2", "C2v"),
  ("mmm", "D2h", "2/m2/m2/m", "Vh", "orthotropic", "orthorhombic"),
  ("3", "C3"), ("-3", "C3i", "S6"),
  ("321", "D3", "32"), ("3m1", "C3v", "3m"), ("-3m1", "D3d", "-32/m", "-3m"),
  ("312", "D3"), ("31m", "C3v"), ("-31m", "D3d", "trigonal"),
  ("4", "C4"), ("-4", "S4"), ("4/m", "C4h"),
  ("422", "D4", "42"), ("4mm", "C4v"), ("-42m", "D2d", "Vd"), ("-4m2", "D2d"),
  ("4/mmm", "D4h", "4/m2/m2/m", "tetragonal"),
  ("6", "C6"), ("-6", "C3h"), ("6/m", "C6h"),
  ("622", "D6", "62"), ("6mm", "C6v"), ("-62m", "D3h"), ("-6m2", "D3h"),
  ("6/mmm", "D6h", "6/m2/m2/m", "hexagonal"),
  ("23", "T"), ("m-3", "Th", "m3", "2/m-3"),
  ("432", "O", "43"), ("-43m", "Td"), ("m-3m", "Oh", "m3m", "4/m-32/m", "cubic"),
  ("532", "I"), ("-5-32/m", "Ih", "icosahedral"),
]

# the Laue classes by MTEX's LaueId option: -1 2/m mmm 4/m 4/mmm m-3 m-3m -3 -3m 6/m 6/mmm
LAUE_CLASSES = ("-1", "12/m1", "mmm", "4/m", "4/mmm", "m-3", "m-3m", "-3", "-3m1", "6/m", "6/mmm")

# the last space group number of each point group, symmetry.m/spaceGroups
SPACE_GROUPS = [(1, "1"), (2, "-1"), (5, "2"), (9, "m"), (15, "2/m"), (24, "222"), (46, "mm2"),
  (74, "mmm"), (80, "4"), (82, "-4"), (88, "4/m"), (98, "422"), (110, "4mm"), (122, "-42m"),
  (142, "4/mmm"), (146, "3"), (148, "-3"), (155, "32"), (161, "3m"), (167, "-3m"), (173, "6"),
  (174, "-6"), (176, "6/m"), (182, "622"), (186, "6mm"), (190, "-6m2"), (194, "6/mmm"),
  (199, "23"), (206, "m-3"), (214, "432"), (220, "-43m"), (230, "m-3m")]


def findPointGroup(name):
  """the symbol a name stands for, findsymmetry: an exact name, then a space group symbol, then
  a name contained in the string, later entries first"""
  name = str(name)
  for row in reversed(POINT_GROUP_NAMES):
    if name in row:
      return row[0]
  try:
    symbol = hm2point(name)
  except ValueError:
    symbol = None
  if symbol is not None:
    return findPointGroup(symbol)
  for row in reversed(POINT_GROUP_NAMES):
    if any(n in name for n in row):
      return row[0]
  raise ValueError(f"unknown point group '{name}'")


def isPointGroupName(name):
  return any(str(name) in row for row in POINT_GROUP_NAMES)


def pointGroupId(symbol):
  """MTEX's numeric id of a symbol, 0 for an unnamed group"""
  for k, row in enumerate(POINT_GROUP_NAMES):
    if row[0] == symbol:
      return k + 1
  return 0


def hm2point(s):
  """a Hermann-Mauguin space group symbol reduced to its point group, hms2point"""
  s = str(s)
  if not s or s[0] not in "ABCFIPR":
    raise ValueError(f"'{s}' is not a space group symbol")
  s = s[1:]
  s = re.sub(r"_(\d)", r"\1", s)            # screw axes 6_3, 2_1 are rotations
  s = re.sub(r"[abcdn]", "m", s)            # glide planes are mirrors
  if sum(c.isdigit() or c == "m" for c in s) >= 4:   # 63, 62, 42, 32: the second digit is a screw
    m = re.search(r"\d\d", s)
    if m is not None and m.group()[0] > m.group()[1]:
      s = s[:m.start() + 1] + s[m.end():]
  if re.search(r"[m2-6]", s):
    s = s.replace("1", "")
  s = "".join(c for c in s if not c.isspace() and not c.isupper() and c not in ":.")
  return s if s else "1"


def monoclinicSetting(name, abg):
  """'2', 'm', '2/m' with lattice angles: the unique axis is the one whose angle is not 90 degree"""
  name = str(name)
  if name not in ("2", "m", "2/m"):
    return name
  import numpy as np
  setting = ["1", "1", "1"]
  setting[int(np.argmax(np.abs(np.asarray(abg, dtype=float) - np.pi / 2)))] = name
  return "".join(setting)

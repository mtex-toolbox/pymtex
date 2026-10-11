# the positional arguments of the public functions: a string names something ('uvw', 'ZYZ',
# 'axisAngle', a section, a method), a flag is a keyword. MATLAB's flags given as strings are
# refused with the keyword to write instead, 'upper' as upper=True and 'noSymmetry' as
# symmetry=False
from __future__ import annotations


def flagKeyword(flag):
  """the keyword that replaces a MATLAB flag string: 'upper' is upper=True, 'noSymmetry'
  symmetry=False"""
  if len(flag) > 2 and flag.startswith("no") and flag[2].isupper():
    return f"{flag[2].lower()}{flag[3:]}=False"
  return f"{flag}=True"


def refuseFlags(args, where, names=()):
  """raise a TypeError for a string among args that is not one of the names the function takes,
  naming the keyword that replaces it"""
  for a in args:
    if isinstance(a, str) and a not in names:
      raise TypeError(f"{where} takes its flags as keywords: write {flagKeyword(a)} instead of '{a}'")


__all__ = []

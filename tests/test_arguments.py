# the positional arguments of the public functions: names stay strings, flags are keywords
import pytest

from mtex.geometry.arguments import flagKeyword, refuseFlags


# a MATLAB flag string is refused with the keyword that replaces it, a name the function takes passes
def test_refuseFlags_namesTheKeywordAndLetsNamesPass():
  assert flagKeyword("upper") == "upper=True" and flagKeyword("noSymmetry") == "symmetry=False"
  refuseFlags(("left", 3), "rotate", names=("left", "right"))
  with pytest.raises(TypeError, match="contourf=True"):
    refuseFlags(("contourf",), "plotPF")

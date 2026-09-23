from mtex.geometry.display import numString, sizeString, table, xnum2str


def test_helpers():
  assert sizeString(()) == "1" and sizeString((2, 3)) == "2 x 3"
  assert numString(1.0) == "1" and numString(0.57735) == "0.577" and numString(1e-12) == "0" and numString(-2.5) == "-2.5"
  assert table(["x", "y"], [[1, 22], [0.5, 3]]) == "   x    y\n   1  0.5\n  22    3"


def test_xnum2str_follows_the_size_of_the_number():
  # MATLAB's xnum2str: whole numbers from ten, one decimal from one, trailing zeros dropped
  assert [xnum2str(x) for x in (27532.4, 12.5, 1.27, 1.0, 0.25, 0.0234, -0.0)] == ["27532", "13", "1.3", "1", "0.25", "0.023", "0"]
  assert xnum2str(2.5e7) == "2.5e+07" and xnum2str(float("nan")) == "NaN"

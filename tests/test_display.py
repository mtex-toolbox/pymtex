from mtex.geometry.display import numString, sizeString, table


def test_helpers():
  assert sizeString(()) == "1" and sizeString((2, 3)) == "2 x 3"
  assert numString(1.0) == "1" and numString(0.57735) == "0.577" and numString(1e-12) == "0" and numString(-2.5) == "-2.5"
  assert table(["x", "y"], [[1, 22], [0.5, 3]]) == "   x    y\n   1  0.5\n  22    3"

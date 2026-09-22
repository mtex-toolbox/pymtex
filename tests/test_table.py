import os

import numpy as np
import pytest

from mtex.io.table import coerceNumeric, headerDict, isNumericRow, leadingLines, linesFrom, readTable, validName


def test_readTable_reads_tabs_carriage_returns_blank_lines_and_comments():
  data = b"# a comment\n1\t2.5\r\n\n3 4e0\r\n# trailing\n"
  M = readTable(data, comments="#")
  assert M.shape == (2, 2)
  assert np.array_equal(M, [[1, 2.5], [3, 4]])


def test_readTable_starts_at_the_offset():
  data = b"names here\n1 2\n3 4\n"
  head, offset = leadingLines(data, isNumericRow)
  assert head == [b"names here"] and offset == 11
  assert np.array_equal(readTable(data, offset), [[1, 2], [3, 4]])


def test_leadingLines_stops_at_the_block_and_strips_carriage_returns():
  data = b"ab\r\ncd\n\n1 2\n3 4"
  head, offset = leadingLines(data, isNumericRow)
  assert head == [b"ab", b"cd", b""] and offset == 8
  head, offset = leadingLines(b"a\nb", lambda l: False)
  assert head == [b"a", b"b"] and offset == 3


def test_linesFrom_reads_a_count_of_lines():
  data = b"a\r\nb\nc\nd"
  lines, offset = linesFrom(data, 3, 2)
  assert lines == [b"b", b"c"] and offset == 7
  assert linesFrom(data, 7, 5) == ([b"d"], 8)


def test_decimal_comma_is_decided_once_for_the_block():
  M = readTable(b"1,5\t2\n3,25\t4\n")
  assert np.array_equal(M, [[1.5, 2], [3.25, 4]])


def test_a_delimiter_is_replaced_before_reading():
  M = readTable(b"1;2.5\n3;4\n", delimiter=";")
  assert np.array_equal(M, [[1, 2.5], [3, 4]])


def test_a_row_of_the_wrong_width_names_itself():
  with pytest.raises(ValueError, match="row 3"):
    readTable(b"1 2\n3 4\n5\n6 7\n")
  with pytest.raises(ValueError, match="row 2"):
    readTable(b"1 2\n3 x\n")
  with pytest.raises(ValueError, match="row 2"):
    readTable(b"1 2\n3 4e\n")


def test_no_data_rows_is_an_error():
  with pytest.raises(ValueError, match="no data rows"):
    readTable(b"# only\n", comments="#")


def test_one_row_is_a_two_dimensional_table():
  assert readTable(b"1 2 3\n").shape == (1, 3)


def test_headerDict_makes_valid_names_and_numbers():
  h = headerDict([("x-star", "0.45"), ("GRID", "HexGrid"), ("2D", "1"), ("", "x"), ("OPERATOR", "  ")])
  assert h == {"x_star": 0.45, "GRID": "HexGrid", "x2D": 1.0, "OPERATOR": ""}


def test_validName_and_coerceNumeric():
  assert validName("Euler angles refer") == "Euler_angles_refer"
  assert validName("#") == "x_"
  assert coerceNumeric("1e3") == 1000.0
  assert coerceNumeric("abc") == "abc"


def test_isNumericRow():
  assert isNumericRow(b"1 2.5\t-3e-2 ,4")
  assert isNumericRow(b" 1,5 2 ")
  assert not isNumericRow(b"Phase X Y")
  assert not isNumericRow(b"")
  assert not isNumericRow(b"1 2 x")


def test_the_two_backends_read_the_same_numbers():
  from mtex.io.kernels import parseBlockNumba, parseBlockNumpy
  block = b"# c\n1 2.5 -3e2\n+4.25E-1\t005\t.5\r\n\n 6. 700000000000000000000 1e-30\n0.000123456789012345 -0.0 9e22\n"
  a = parseBlockNumpy(block, False, b"#")
  b = parseBlockNumba(block, False, b"#")
  if b is not None:
    assert a.shape == b.shape == (4, 3)
    assert np.array_equal(a, b)
  # beyond fifteen digits the mantissa rounds once more, an ulp or two
  a = parseBlockNumpy(b"1234567890.12345678 0.1234567890123456789\n", False, None)
  b = parseBlockNumba(b"1234567890.12345678 0.1234567890123456789\n", False, None)
  assert b is None or np.allclose(a, b, rtol=1e-15, atol=0)
  with open(os.path.join(os.path.dirname(__file__), "fixtures", "ebsd", "Forsterite_cut.ctf"), "rb") as f:
    data = f.read()
  block = data.split(b"Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\r\n", 1)[1]
  a = parseBlockNumpy(block, False, None)
  b = parseBlockNumba(block, False, None)
  assert b is None or np.array_equal(a, b)
  a = parseBlockNumpy(block.replace(b".", b","), True, None)
  b = parseBlockNumba(block.replace(b".", b","), True, None)
  assert b is None or np.array_equal(a, b)

from mtex import crystalFrame, degree, getMTEXpref, orientation, setMTEXpref, specimenFrame, vector3d
from mtex.geometry.display import (aligned, display, fields, frameLabel, framesLabel, header, numString, shownRows,
                                   sizeString, table, takeRows, textTable, xnum2str)


def test_helpers():
  assert sizeString(()) == "1" and sizeString((5,)) == "5" and sizeString((2, 3)) == "2 × 3"
  assert numString(1.0) == "1" and numString(0.57735) == "0.577" and numString(1e-12) == "0" and numString(-2.5) == "-2.5"
  assert table(["x", "y"], [[1, 22], [0.5, 3]]) == "   x    y\n   1  0.5\n  22    3"


def test_xnum2str_follows_the_size_of_the_number():
  # MATLAB's xnum2str: whole numbers from ten, one decimal from one, trailing zeros dropped
  assert [xnum2str(x) for x in (27532.4, 12.5, 1.27, 1.0, 0.25, 0.0234, -0.0)] == ["27532", "13", "1.3", "1", "0.25", "0.023", "0"]
  assert xnum2str(2.5e7) == "2.5e+07" and xnum2str(float("nan")) == "NaN"


def test_fields_align_their_colons_and_leave_out_none():
  assert fields([("mineral", "Iron"), ("color", None), ("symmetry", "m3̅m")]) == ["  mineral : Iron", "  symmetry: m3̅m"]
  assert fields([("kernel", "x")], "    ") == ["    kernel: x"] and fields([("a", None)]) == []


def test_header_and_display():
  assert header("vector3d") == "vector3d" and header("vector3d", "", "y↑→x") == "vector3d (y↑→x)"
  assert display("S1Grid", [("points", 3), ("range", None)], "  table") == "S1Grid\n  points: 3\n  table"


def test_aligned_aligns_each_run_at_its_indent():
  text = "X\n  bandwidth: 3\n  mean: 1\n  a table\n    sub: 2\n    longer key: 3"
  assert aligned(text) == "X\n  bandwidth: 3\n  mean     : 1\n  a table\n    sub       : 2\n    longer key: 3"
  assert aligned(aligned(text)) == aligned(text)


def test_a_long_table_keeps_its_first_and_last_rows():
  assert shownRows(20) == list(range(20))
  rows = shownRows(100)
  assert rows == [0, 1, 2, 3, 4, None, 95, 96, 97, 98, 99]
  take, pos = takeRows(100)
  assert take == [0, 1, 2, 3, 4, 95, 96, 97, 98, 99] and pos == [0, 1, 2, 3, 4, None, 5, 6, 7, 8, 9]
  lines = table(["k"], [range(100)]).split("\n")
  assert lines[6].strip() == "⋮" and lines[-1].strip() == "99" and len(lines) == 12
  old = getMTEXpref("displayRows")
  try:
    setMTEXpref("displayRows", 8)
    assert shownRows(9) == [0, 1, None, 7, 8]
  finally:
    setMTEXpref("displayRows", old)


def test_text_tables_carry_no_trailing_spaces():
  assert textTable(["a", "bb"], [["1", "22"], ["", ""]]) == "   a  bb\n   1\n  22"


def test_frames_are_named_as_matlab_names_them():
  cs = crystalFrame("m-3m", mineral="Iron")
  assert frameLabel(cs) == "Iron" and frameLabel(crystalFrame("432")) == "432" and frameLabel(None) == ""
  assert frameLabel(specimenFrame.rolling) == specimenFrame.rolling.conventionChar()
  assert framesLabel(cs, specimenFrame.rolling) == "Iron → " + specimenFrame.rolling.conventionChar()


def test_the_displays_of_the_classes():
  cs = crystalFrame("6/mmm", [1.4, 1.4, 1.5], mineral="Titanium")
  # the angles only where the lattice does not imply them, the colons aligned
  assert repr(cs).split("\n")[:3] == ["crystalFrame (⊙c→a)", "  mineral        : Titanium", "  symmetry       : 6/mmm"]
  assert "alpha" not in repr(cs)
  o = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, crystalFrame("m-3m", mineral="Iron"))
  assert repr(o).split("\n")[0].startswith("orientation (Iron → ")
  lines = repr(vector3d.rand(100)).split("\n")
  assert lines[1] == "  size: 100" and "⋮" in lines[8] and len(lines) == 14

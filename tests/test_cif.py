# the CIF reader: the frames of the shipped files, the mineral, the cell, the lookup by name
import numpy as np
import pytest

from mtex import crystalFrame, degree
from mtex.io.cif import cifFiles, loadCIF


def test_shipped_files_read():
  cs = crystalFrame.load("Al-Aluminum.cif")
  assert cs.name == "Aluminum" and cs.sym.char() == "m3̅m"
  assert np.allclose(cs.abc, 4.04958)
  ti = loadCIF("Ti-Titanium-alpha.cif")
  assert ti.sym.char() == "6/mmm" and abs(ti.abg[2] - 120 * degree) < 1e-12
  assert "Al-Aluminum.cif" in cifFiles()


def test_name_without_extension_and_missing_file():
  assert crystalFrame.load("Al-Aluminum") is crystalFrame.load("Al-Aluminum.cif")
  with pytest.raises(FileNotFoundError):
    loadCIF("no-such-mineral.cif")

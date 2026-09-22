# the ODF writers: the generic grid, a given grid, the component file, the VPSC sample
import os

import numpy as np

from mtex import Miller, crystalFrame, degree, equispacedSO3Grid, export, fibreODF, orientation, uniformODF, unimodalODF, xvector


def _model():
  cs = crystalFrame("cubic")
  return cs, 0.5 * uniformODF(cs) + 0.2 * fibreODF(Miller(1, 0, 0, cs), xvector, halfwidth=10 * degree) \
    + 0.3 * unimodalODF(orientation.byAxisAngle(xvector, 45 * degree, cs), halfwidth=15 * degree)


def test_generic_grid_file(tmp_path):
  cs, odf = _model()
  fname = tmp_path / "odf.txt"
  export(odf, fname, "Bunge", resolution=10 * degree)
  lines = fname.read_text().split("\n")
  assert lines[0] == "% MTEX ODF" and lines[3].startswith("% phi1")
  rows = np.array([[float(x) for x in l.split()] for l in lines[4:] if l])
  assert rows.shape[1] == 4 and rows[0, 0] == 0 and rows[1, 0] == 10
  # the value column is the function at the node
  o = orientation.byEuler(rows[5, 0] * degree, rows[5, 1] * degree, rows[5, 2] * degree, cs)
  assert abs(rows[5, 3] - odf.eval(o)) < 1e-4


def test_given_grid_and_component_file(tmp_path):
  cs, odf = _model()
  S3G = equispacedSO3Grid(cs, resolution=20 * degree)
  export(odf, tmp_path / "grid.txt", S3G, "Bunge", "generic")
  n = sum(1 for l in (tmp_path / "grid.txt").read_text().split("\n") if l and not l.startswith("%"))
  assert n == S3G.size
  export(odf, tmp_path / "odf.mtex", interface="mtex")
  text = (tmp_path / "odf.mtex").read_text()
  assert "%% uniform component" in text and "%% fibre component" in text and "%% radial component" in text


def test_vpsc_sample(tmp_path):
  cs, odf = _model()
  export(odf, tmp_path / "vpsc.txt", "VPSC", points=200)
  lines = (tmp_path / "vpsc.txt").read_text().split("\n")
  assert lines[3] == "B 200" and len([l for l in lines[4:] if l]) == 200

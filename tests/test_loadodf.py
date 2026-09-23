# the ODF readers: a generic table as values or as masses, a VPSC file, the round trip
import numpy as np

from mtex import Miller, SO3Fun, crystalFrame, degree, export, fibreODF, mean, orientation, uniformODF, xvector
from mtex.io.loadodf import loadODF


def test_generic_table_round_trip(tmp_path):
  cs = crystalFrame("m-3m")
  odf = 0.6 * uniformODF(cs) + 0.4 * fibreODF(Miller(1, 0, 0, cs), xvector, halfwidth=15 * degree)
  fname = tmp_path / "odf.txt"
  export(odf, fname, "Bunge", resolution=15 * degree)
  # the masses at the nodes give a density of mean one
  dens = SO3Fun.load(fname, "Bunge", "density", CS=cs, halfwidth=15 * degree, columnNames=["Euler 1", "Euler 2", "Euler 3", "weights"])
  assert abs(mean(dens) - 1) < 1e-6
  # the values at the nodes are met by the fit
  fit = loadODF(fname, "Bunge", "interp", CS=cs, columnNames=["Euler 1", "Euler 2", "Euler 3", "weights"])
  o = orientation.byEuler(30 * degree, 45 * degree, 60 * degree, cs)
  assert abs(fit.eval(o) - odf.eval(o)) < 0.15 * max(1.0, odf.eval(o))


def test_degrees_are_told_by_the_range(tmp_path):
  cs = crystalFrame("432")
  fname = tmp_path / "small.txt"
  fname.write_text("0.1 0.2 0.3 1\n0.2 0.1 0.4 1\n")
  dens = loadODF(fname, CS=cs, halfwidth=10 * degree)
  assert abs(mean(dens) - 1) < 1e-6


def test_vpsc_blocks(tmp_path):
  cs = crystalFrame("432")
  fname = tmp_path / "tex.out"
  rows = "\n".join(f"{10 * k:7.2f} {20:7.2f} {30:7.2f} {0.25:11.7f}" for k in range(4))
  fname.write_text(f"texture exported\n\n\nB 4\n{rows}\n")
  odf = SO3Fun.load(fname, cs, halfwidth=10 * degree)
  assert abs(mean(odf) - 1) < 1e-6 and odf.CS is cs
  # a weight file has no strain and no ellipsoid, MATLAB's NaN
  assert np.isnan(odf.opt.strain) and np.all(np.isnan(odf.opt.strainEllipsoid)) and odf.opt.orientations.size == 4
  # a VPSC output block keeps its header and its further columns in opt
  rows = "\n".join(f"{10 * k:7.2f} {20:7.2f} {30:7.2f} {0.25:11.7f} {0.3:9.6f} {2.5:9.6f}" for k in range(4))
  fname.write_text(f"TEXTURE AT STRAIN =    0.2500\n   1.123   1.127   0.750  <-- axes\n -180.00   90.00 -180.00  <-- angles\nB 4\n{rows}\n")
  odf = SO3Fun.load(fname, cs, halfwidth=10 * degree)
  assert odf.opt.strain == 0.25 and np.allclose(odf.opt.strainEllipsoid, [1.123, 1.127, 0.75])
  assert np.allclose(odf.opt.strainEllipsoidAngles, [-180, 90, -180]) and odf.opt.data.shape == (4, 2)

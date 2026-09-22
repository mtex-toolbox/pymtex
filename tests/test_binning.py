import os

import numpy as np

from mtex import Miller, degree, loadEBSD
from mtex.functions.density import calcDensity, calcPDF
from mtex.wizard.binning import binOrientations, binnedDensity

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


def test_binning_merges_nothing_away():
  ebsd = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  o = ebsd["Forsterite"].orientations
  reps, w = binOrientations(o, 4 * degree)
  assert reps.size == w.size < o.size and w.sum() == o.size
  # a finer cell keeps more, a coarser fewer
  assert binOrientations(o, 2 * degree)[0].size > reps.size > binOrientations(o, 10 * degree)[0].size


def test_binned_density_peaks_where_the_full_one_does():
  ebsd = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  o = ebsd["Forsterite"].orientations
  binned, full = binnedDensity(o, 4 * degree), calcDensity(o, harmonic=True)
  h = Miller(1, 0, 0, o.CS)
  p1, p2 = calcPDF(binned, h, None), calcPDF(full, h, None)
  assert abs(float(p1.max()[0]) - float(p2.max()[0])) < 0.1 * float(p2.max()[0])
  # a short list is estimated as it is
  small = binnedDensity(o[:100])
  assert type(small).__name__ == "SO3FunHarmonic"

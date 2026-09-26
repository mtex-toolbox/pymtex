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


# the preview density of a phase from at most so many measurements, the same draw every time,
# and still peaking where the density of all of them does
def test_the_preview_density_subsamples():
  from mtex.wizard.binning import previewDensity
  ebsd = loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))
  k = next(i for i, f in enumerate(ebsd.CSList.frames) if f.name == "Forsterite")
  n = int(np.sum((ebsd.phaseId.reshape(-1) == k) & ebsd.isIndexed.reshape(-1)))
  full, sub = previewDensity(ebsd, k), previewDensity(ebsd, k, maxPoints=n // 3)
  again = previewDensity(ebsd, k, maxPoints=n // 3)
  h = Miller(1, 0, 0, ebsd.CSList.frames[k])
  p1, p2, p3 = (calcPDF(f, h, None) for f in (full, sub, again))
  assert abs(float(p2.max()[0]) - float(p1.max()[0])) < 0.25 * float(p1.max()[0])
  assert float(p2.max()[0]) == float(p3.max()[0])

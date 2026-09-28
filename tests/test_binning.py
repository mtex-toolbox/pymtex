import os

import numpy as np
import pytest

from mtex import Miller, degree, loadEBSD
from mtex.functions.density import calcDensity, calcPDF
from mtex.wizard.binning import binOrientations, binnedDensity

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ebsd")


@pytest.fixture(scope="module")
def ebsd():
  return loadEBSD(os.path.join(FIXTURES, "Forsterite_cut.ctf"))


@pytest.fixture(scope="module")
def o(ebsd):
  return ebsd["Forsterite"].orientations


def test_binOrientations_mergesNothingAway(o):
  reps, w = binOrientations(o, 4 * degree)
  assert reps.size == w.size < o.size and w.sum() == o.size


# a finer cell keeps more, a coarser fewer
def test_binOrientations_keepsMoreInAFinerCell(o):
  reps, _ = binOrientations(o, 4 * degree)
  assert binOrientations(o, 2 * degree)[0].size > reps.size > binOrientations(o, 10 * degree)[0].size


def test_binnedDensity_peaksWhereTheFullOneDoes(o):
  binned, full = binnedDensity(o, 4 * degree), calcDensity(o, halfwidth=10 * degree, harmonic=True)
  h = Miller(1, 0, 0, o.CS)
  p1, p2 = calcPDF(binned, h, None), calcPDF(full, h, None)
  assert abs(float(p1.max()[0]) - float(p2.max()[0])) < 0.1 * float(p2.max()[0])


# a short list is estimated as it is
def test_binnedDensity_ofAShortListIsNotBinned(o):
  small = binnedDensity(o[:100])
  assert type(small).__name__ == "SO3FunHarmonic"


# the preview density of a phase from all its measurements, and from at most a third of them
@pytest.fixture(scope="module")
def previews(ebsd):
  from mtex.wizard.binning import previewDensity
  k = next(i for i, f in enumerate(ebsd.CSList.frames) if f.name == "Forsterite")
  n = int(np.sum((ebsd.phaseId.reshape(-1) == k) & ebsd.isIndexed.reshape(-1)))
  full, sub = previewDensity(ebsd, k), previewDensity(ebsd, k, maxPoints=n // 3)
  again = previewDensity(ebsd, k, maxPoints=n // 3)
  h = Miller(1, 0, 0, ebsd.CSList.frames[k])
  return tuple(calcPDF(f, h, None) for f in (full, sub, again))


# the preview density from at most so many measurements still peaks where the density of all
# of them does
def test_previewDensity_subsampledPeaksWhereTheFullOneDoes(previews):
  p1, p2, _ = previews
  assert abs(float(p2.max()[0]) - float(p1.max()[0])) < 0.25 * float(p1.max()[0])


# the subsample is the same draw every time
def test_previewDensity_subsamplesTheSameDrawEveryTime(previews):
  _, p2, p3 = previews
  assert float(p2.max()[0]) == float(p3.max()[0])

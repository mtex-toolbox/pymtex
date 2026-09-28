# the Patala misorientation colour key, against MATLAB's PatalaColorKey
import numpy as np
import pytest

from mtex import PatalaColorKey, crystalFrame, orientation, rotation

# two misorientations per Laue group and MATLAB's colours of them (bench: 2000 per group agree to 1e-14)
PINNED = {
  "m-3m": ([[0.610006183976, -0.763857546797, 0.12496471778, -0.169699508022], [-0.216148281674, -0.102951572327, -0.964580169307, -0.110751934812]],
           [[0.801557780425, 0.752557465526, 0.595887897226], [0.971808396069, 0.064085439159, 0.736207214168]]),
  "m-3": ([[-0.337244225633, -0.44623047735, -0.12949635751, -0.818764549033], [0.306355527062, 0.222606941245, 0.015034588955, 0.925400670998]],
          [[0.309523144169, 0.10880010891, 0.307462133237], [0.577614780749, 0.546277090107, 0.412149705192]]),
  "mmm": ([[-0.624957019665, 0.312908766158, 0.020987972335, -0.714896029259], [0.25016226183, 0.250841843083, 0.264012612927, -0.897103423653]],
          [[0.02870229561, 0.162114977795, 0.563381710668], [0.855647506768, 0.852336983669, 0.852038160774]]),
  "4/mmm": ([[-0.517223220336, 0.624171707483, 0.555112147593, 0.186387562667], [-0.815928256194, -0.16683003178, -0.257819020662, -0.489855258036]],
            [[0.103134313761, 0.764230763582, 0.880045097845], [0.877027306974, 0.655425110289, 0.370303319144]]),
  "6/mmm": ([[0.674915688906, -0.056020888415, 0.09417836324, 0.729712894793], [-0.566908678746, -0.257045667304, -0.074450511366, -0.779101531406]],
            [[0.994183806226, 0.012521110069, 0.171743786383], [0.842957997147, 0.800148868443, 0.778375213269]]),
}


@pytest.mark.parametrize("pg", list(PINNED))
def test_PatalaColorKey_coloursAreMATLABs(pg):
  q, rgb = PINNED[pg]
  cs = crystalFrame(pg)
  q = np.array(q)
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  out = PatalaColorKey(cs).orientation2color(orientation(rotation(q), cs, cs))
  assert out.shape == (2, 3) and np.allclose(out, rgb, atol=1e-9)


# two hundred random misorientations of a Laue group
def _misorientations(pg):
  cs = crystalFrame(pg)
  q = np.random.default_rng(1).normal(size=(200, 4))
  return cs, orientation(rotation(q / np.linalg.norm(q, axis=1, keepdims=True)), cs, cs)


@pytest.mark.parametrize("pg", list(PINNED))
def test_PatalaColorKey_equivalentMisorientationsGetOneColour(pg):
  cs, mori = _misorientations(pg)
  key = PatalaColorKey(cs)
  rgb = key.orientation2color(mori)
  G = cs.sym.properGroup().rotations()
  g1, g2 = G[3 % len(G)], G[len(G) - 1]
  # a symmetric equivalent on either side and the grain exchange give the same colour
  eq = orientation(rotation(g2) * rotation(mori.data) * rotation(g1), cs, cs)
  assert np.allclose(key.orientation2color(eq), rgb, atol=1e-9)
  assert np.allclose(key.orientation2color(orientation(rotation(mori.data).inv(), cs, cs)), rgb, atol=1e-9)


@pytest.mark.parametrize("pg", list(PINNED))
def test_PatalaColorKey_coloursLieInTheColourCube(pg):
  cs, mori = _misorientations(pg)
  rgb = PatalaColorKey(cs).orientation2color(mori)
  assert np.all((rgb >= 0) & (rgb <= 1))


def test_PatalaColorKey_refusesAllButTheFiveLaueGroups():
  with pytest.raises(ValueError):
    PatalaColorKey(crystalFrame("-3m"))


def test_PatalaColorKey_isAntipodal():
  assert PatalaColorKey(crystalFrame("m-3m")).antipodal

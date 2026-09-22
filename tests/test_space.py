# the geometry of plots in space (docs/plans/plotting.md S5): the sphere mesh, the grid, the
# axes of a frame and the fundamental region in axis angle space, the numbers of
# ../mtexJulia/test/space.jl S5 and S5b
import numpy as np
import pytest

from mtex import Miller, crystalFrame, degree, fundamentalRegion, orientation, rotation, specimenFrame, vector3d
from mtex.plotting.space import (axisAnglePoints, frameAxes, maxAngles, regionLines, regionSurface, regionWalls, smallCircle,
                                 sphereGridLines, sphereMesh)

rng = np.random.default_rng(5)


def test_S5_the_sphere_in_space():
  V, F = sphereMesh(10 * degree)
  assert np.allclose(np.linalg.norm(V, axis=1), 1)
  # every triangle is wound outwards
  n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
  assert np.all(np.sum(n * (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]), axis=1) > 0)
  # a closed surface: every edge in two faces, and Euler's characteristic
  edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
  u, c = np.unique(edges, axis=0, return_counts=True)
  assert np.all(c == 2) and len(V) - len(u) + len(F) == 2
  assert len(sphereMesh(5 * degree)[0]) > len(V)
  # the grid: five small circles and twelve meridians at thirty degrees
  ls = sphereGridLines(30 * degree)
  assert len(ls) == 5 + 12 and all(np.allclose(np.linalg.norm(l, axis=1), 1) for l in ls)
  assert np.allclose(ls[0][:, 2], np.cos(30 * degree))
  assert np.isclose(ls[-1][0, 2], 1) and np.isclose(ls[-1][-1, 2], -1)
  c = smallCircle([1, 1, 1], 20 * degree)
  assert np.allclose(c @ (np.ones(3) / np.sqrt(3)), np.cos(20 * degree)) and np.allclose(c[0], c[-1])
  # the axes a frame names
  assert [n for _, n in frameAxes(specimenFrame("1"))] == ["X", "Y", "Z"]
  ax = frameAxes(crystalFrame("6/mmm"))
  assert [n for _, n in ax] == ["a", "b", "c"] and abs(ax[0][0] @ ax[1][0] - np.cos(120 * degree)) < 1e-9
  assert [n for _, n in frameAxes(None)] == ["x", "y", "z"]


def test_S5b_the_fundamental_region_in_axis_angle_space():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  oR = fundamentalRegion(cs, ss)
  w = maxAngles(oR, [[0, 0, 1], [1, 1, 1], [1, 1, 0]])
  assert np.isclose(w[0], np.pi / 4) and np.isclose(w[1], np.pi / 3) and np.isclose(w[2], 2 * np.arctan(np.tan(np.pi / 8) * np.sqrt(2)))
  # about a random axis the rotation just inside is inside and just beyond is not; against
  # a specimen without symmetry every axis is allowed, against 222 only those of the sector
  oR1 = fundamentalRegion(cs, specimenFrame("1"))
  for v in vector3d.rand(10, rng=rng):
    om = maxAngles(oR1, v.data)[0]
    assert 0 < om <= 63 * degree
    assert oR1.checkInside(rotation.byAxisAngle(v, om - 1e-2)) and not oR1.checkInside(rotation.byAxisAngle(v, om + 1e-2))
  w = maxAngles(oR, vector3d.rand(50, rng=rng).data)
  assert np.any(w == 0) and np.all(w <= 63 * degree)
  assert np.isclose(maxAngles(fundamentalRegion(crystalFrame("1"), specimenFrame("1")), [[1, 2, 3]])[0], np.pi)
  # the surface: no NaN where the region has no sector, every point just inside
  P, F = regionSurface(oR1, 10 * degree)
  assert not np.any(np.isnan(P)) and F.shape[1] == 3
  inside = oR1.checkInside(rotation.byAxisAngle(vector3d(P[1:-1]), np.linalg.norm(P[1:-1], axis=1) * (1 - 1e-6)))
  assert np.all(inside)
  assert regionWalls(oR1) == [] and all(not np.any(np.isnan(l)) for l in regionLines(oR1))
  # a region cut down to an axis sector has NaN where the axis lies outside, and walls
  mR = fundamentalRegion(cs, cs)
  assert mR.sector is not None
  Pm, _ = regionSurface(mR, 10 * degree)
  assert np.any(np.isnan(Pm[:, 0])) and np.any(~np.isnan(Pm[:, 0]))
  walls = regionWalls(mR)
  assert walls and all(np.allclose(w[0], 0) for w in walls)
  # orientations as points: the norm is the angle, and the points lie inside the region
  os_ = orientation(rotation.rand(40, rng=rng), cs, ss)
  pts = axisAnglePoints(os_)
  inside = os_.project2FundamentalRegion()
  assert pts.shape == (40, 3) and np.allclose(np.linalg.norm(pts, axis=1), inside.angle(noSymmetry=True))
  assert np.all(np.linalg.norm(pts, axis=1) <= maxAngles(oR, pts) + 1e-9)
  assert np.allclose(axisAnglePoints(orientation(rotation.id, cs, ss)), 0)


def test_S5_crystal_shapes_on_the_screen():
  from mtex import EBSD, calcGrains, crystalShape, phaseList, zvector
  from mtex.plotting.space import shapeOutlines, shapesOnMap, visibleFaces
  fr = crystalFrame("m-3m", mineral="cubes")
  cube = crystalShape.cube(fr)
  # the front faces are those a viewer sees
  assert visibleFaces(cube, zvector).sum() == 1 and visibleFaces(cube, vector3d(1, 1, 1)).sum() == 3
  assert visibleFaces(cube, [1, 1, 0]).sum() == 2 and visibleFaces(cube, [-1, 0, 0]).sum() == 1
  L = orientation.byEuler(np.array([0, 45]) * degree, 0, 0, fr, specimenFrame("1")) * cube
  assert visibleFaces(L, zvector).shape == (2, 6) and np.all(visibleFaces(L, zvector).sum(axis=1) == 1)
  # the shapes into a map: one per grain of the phase, with the scale of the grain
  nx, ny = 8, 4
  i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
  i, j = i.reshape(-1, order="F"), j.reshape(-1, order="F")
  pos = np.stack([i, j, np.zeros(nx * ny)], axis=-1).astype(float)
  rot = rotation.byAxisAngle(zvector, np.where(i < 4, 0.0, 30 * degree))
  g = calcGrains(EBSD(pos, rot, np.full(nx * ny, 2, dtype=np.int8), phaseList(crystalFrame("1"), fr)), angle=5 * degree)
  s, ids = shapesOnMap(g, cube)
  assert g.size == 2 and len(s) == 2 and list(ids) == [0, 1] and s.frame is g.frame
  polys, shape, face = shapeOutlines(s, g.plane)
  assert len(polys) == 2 and list(shape) == [0, 1]
  for p, k in zip(polys, shape):
    # seen along the map normal a cube shows one face, a square of the side of the cube,
    # 1 / sqrt(3) of the shape's diameter, times the grain's scale, about the centroid
    side = np.linalg.norm(p[1] - p[0])
    assert len(p) == 4 and np.isclose(side, np.sqrt(g.area[k]) / np.sqrt(3))
    assert np.allclose(np.linalg.norm(p - g.plane.to2d(g.centroid.data[k]), axis=1), side / np.sqrt(2))
  # a layout that flips the plane is seen from the other side
  flipped = shapeOutlines(s, g.plane, [[1, 0], [0, -1]])
  assert len(flipped[0]) == 2 and not np.array_equal(flipped[2], face) or np.all(np.abs(flipped[0][0][:, 1]) == np.abs(polys[0][:, 1]))
  # a shape of another lattice draws nothing, the scale scales
  other = crystalShape(Miller([[1, 0, 0], [0, 0, 1]], crystalFrame("6/mmm", [1, 1, 1.6])))
  assert shapesOnMap(g, other)[0] is None
  assert np.allclose(shapesOnMap(g, cube, scale=0.5)[0].diameter, 0.5 * s.diameter)

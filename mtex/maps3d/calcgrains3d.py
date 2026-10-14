# grain reconstruction in a volume (MATLAB EBSD3square/calcGrains, ../master)
#
#   grains = calcGrains(ebsd3, angle=10*degree)          the grains, ebsd3.grainId written
#   grains = calcGrains(ebsd3, criterion=gbcSoft(...), minPixel=5)
#
# The voxels are the cells, so there is no decomposition: the neighbour pairs are the three
# axis shifts of the lattice, the criteria of the 2D maps judge them, and the grains are the
# connected components of the pairs they join. Every face between two grains, and every face
# of the volume's surface, is a unit square of the lattice written as two triangles over the
# corners of the voxels. The voxels are taken in MATLAB's column-major order throughout, so
# that the pairs, the grain numbers and the faces come in MATLAB's order; the padding of a
# gridded list belongs to no grain and its faces are outer faces, as in 2D.
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ..geometry.kernels import qconj, qmul
from ..maps.calcgrains import _grainTable, undersized
from ..maps.calcgrains import calcGrains as _calcGrains2d
from ..maps.decomposition import _gather
from ..maps.criteria import DEFAULT_ANGLE, gbcAngle, grainBoundaryCriterion
from ..maps.segment import components
from .ebsd3 import EBSD3
from .grain3boundary import grain3Boundary
from .grain3d import grain3d


def calcGrains(ebsd, angle=DEFAULT_ANGLE, threshold=None, criterion=None, minPixel=1, alpha=3.1, removeQuadruplePoints=False,
               mcl=None, fmc=None, verbose=False, backend=None, grainId=False):
  """the grains of a map, the pixels joined across every boundary the criterion keeps; MATLAB's
  grain reconstruction from 2d and 3d EBSD data

    grains = calcGrains(ebsd)                                  # grain boundaries above 15 degree
    grains = calcGrains(ebsd, angle=10*degree, minPixel=5, alpha=3.7)  # another angle, no grains below 5 pixels
    grains = calcGrains(ebsd, angle=[10*degree, 2*degree])     # high and low angle boundaries
    grains = calcGrains(ebsd, criterion=gbcAngle(10*degree, byPhase={'Enstatite': 5*degree}))  # a threshold per phase
    grains = calcGrains(ebsd, criterion=gbcSoft(10*degree, 2*degree), mcl=(1.4, 4))  # a soft threshold and Markov clustering
    grains = calcGrains(ebsd, fmc=1.5, minPixel=10)            # fast multiscale clustering
    grains = calcGrains(ebsd, grainId=True)                    # the grains stored with the map

  Two neighbouring pixels belong to one grain when the misorientation between them stays
  below the threshold angle, and a grain is a connected set of such pixels. With a pair
  [high, low] the boundaries above high separate the grains, and those between low and high
  are kept as low angle boundaries inside them, grains.innerBoundary: the subgrain boundaries.
  A threshold per phase, a soft threshold or any other rule is a grainBoundaryCriterion.

  The map is divided into cells around its pixels, a Voronoi decomposition bounded by alpha:
  a not indexed region narrower than 2 alpha pixel spacings is filled by the grains around
  it, a wider one becomes a notIndexed grain.

  The map is changed: it keeps the grain ids, the pixels the minPixel cull drops and the
  notIndexed pixels the grains absorb, so MATLAB's second output is the map passed; run on
  ebsd.copy() to keep the map as it was.

  Input
    ebsd - EBSD, a map of the plane, or EBSD3, a map of volumes
  Output
    grains - grain2d, or grain3d for a map of volumes; ebsd.grainId numbers the grain of
        every pixel
  Keywords
    angle - the misorientation angle above which neighbouring pixels belong to different
        grains; a pair [high, low] also keeps the low angle boundaries between
        low and high as grains.innerBoundary
    threshold - the same as angle, which it replaces when given
    criterion - a grainBoundaryCriterion instead of the angle: gbcAngle with thresholds per
        phase, gbcSoft for a soft threshold, gbcEqual for a property that must agree,
        gbcCustom for a rule of your own, gbcFMC with every parameter of fast multiscale
        clustering
    minPixel - the smallest number of pixels of a grain; the pixels of an indexed grain
        that would be smaller are marked notIndexed and the grains reconstructed again
    alpha - the bound of the decomposition in pixel spacings: a not indexed region
        narrower than 2 alpha is filled (map of the plane)
    removeQuadruplePoints - split every point where four pixels of a square grid meet into
        two triple points, joining the diagonal pair the criterion connects more (map of
        the plane)
    mcl - (p, maxIter), Markov clustering of the links between the pixels before the grains
        are formed, the inflation power p (1.4 in MATLAB) and the number of iterations (4);
        with a soft criterion it merges grains a hard threshold would split (map of the plane)
    fmc - fast multiscale clustering with the sharpness cmaha, larger giving more grains;
        gbcFMC as the criterion takes its other parameters (map of the plane)
    verbose - report the cluster hierarchy of fast multiscale clustering
    backend - 'numba' or 'numpy', the loops of the decomposition and the segmentation; by
        default the preference 'kernels' (map of the plane)
    grainId - join the pixels of one grain id stored with the map, ebsd.prop['grainId'] or
        the voxels' ebsd.grainId, instead of a criterion on the orientations
  References
    F. Bachmann, R. Hielscher, H. Schaeben, Grain detection from 2d and 3d EBSD data -
        Specification of the MTEX algorithm, Ultramicroscopy 111 (2011) 1720-1733
    C. McMahon, B. Soe, A. Loeb, A. Vemulkar, M. Ferry, L. Bassman, Boundary identification
        in EBSD data with a generalization of fast multiscale clustering, Ultramicroscopy
        133 (2013) 16-25
  See also
    GrainReconstruction, GrainReconstructionAdvanced, GrainReconstructionMCL, gbcAngle,
    grain2d.merge, smoothBoundary"""
  # calcGrains(ebsd, 'grainId'): MATLAB's flag
  if isinstance(angle, str) and angle == "grainId":
    angle, grainId = DEFAULT_ANGLE, True
  if not isinstance(ebsd, EBSD3):
    return _calcGrains2d(ebsd, angle, threshold, criterion, minPixel, alpha, removeQuadruplePoints, mcl, fmc, verbose, backend,
                         grainId)
  planar = {"alpha": alpha != 3.1, "removeQuadruplePoints": removeQuadruplePoints, "mcl": mcl is not None,
            "fmc": fmc is not None, "verbose": verbose, "backend": backend is not None}
  if any(planar.values()):
    raise TypeError(f"calcGrains of a map of volumes takes no {', '.join(k for k, v in planar.items() if v)}")
  return _calcGrains3d(ebsd, angle, threshold, criterion, minPixel, grainId)


def _calcGrains3d(ebsd, angle=DEFAULT_ANGLE, threshold=None, criterion=None, minPixel=1, grainId=False):
  if isinstance(angle, grainBoundaryCriterion):
    criterion, angle = angle, None
  crit = criterion if criterion is not None else gbcAngle(angle if threshold is None else threshold)
  # the grains a file stored: voxels of one stored id belong together, an id of 0 to no grain
  stored = None
  if grainId:
    if getattr(ebsd, "grainId", None) is None:
      raise ValueError("calcGrains(ebsd, 'grainId') needs the grain ids stored with the voxels")
    stored = np.asarray(ebsd.grainId).reshape(ebsd.shape).copy()
    ebsd.phaseId[stored == 0] = 0
  g, newId = ebsd.gridify(return_index=True)
  shape = g.shape
  n = int(np.prod(shape))
  # the voxels in MATLAB's order: e.flat[m] is MATLAB's voxel m + 1
  toC = np.arange(n).reshape(shape).ravel(order="F")
  e = g._take(toC, (n,))
  pad = e.isPadding.reshape(-1)
  L = np.arange(n).reshape(shape, order="F")
  Dl, Dr, dim = _pairs(L)
  if stored is not None:
    sid = np.zeros(n, dtype=np.int64)
    sid[newId] = stored.reshape(-1)
    sid = sid[toC]

  # the components of the voxels the criterion, or the stored ids, join
  def segment():
    if stored is None:
      connect = np.asarray(crit.evaluate(e, Dl, Dr), dtype=float)
    else:
      ph = e.phaseId.reshape(-1)
      connect = ((sid[Dl] == sid[Dr]) & (ph[Dl] == ph[Dr])).astype(float)
    join = (connect > 0) & ~pad[Dl] & ~pad[Dr]
    return _byLastVoxel(components(n, Dl[join], Dr[join]), ~pad)

  grainId, nGrains = segment()
  if minPixel > 1:
    # the cull changes what is joined, so the segmentation runs once more
    under = undersized(grainId, nGrains, e.isIndexed.reshape(-1), minPixel)
    if under.any():
      e.phaseId.reshape(-1)[under] = 0
      g.phaseId.reshape(-1)[toC[under]] = 0
      grainId, nGrains = segment()
  grains = _grains(e, g, grainId, nGrains, Dl, Dr, dim, L)
  # the grain id and the culled phases written into the map that was passed
  gid = np.empty(n, dtype=np.int64)
  gid[toC] = grainId
  if ebsd.isGridded:
    ebsd.grainId = gid.reshape(shape)
  else:
    ebsd.phaseId.reshape(-1)[:] = g.phaseId.reshape(-1)[newId]
    ebsd.grainId = gid[newId].reshape(ebsd.shape)
  return grains


# the grains numbered from 1 in the order of their last voxel, as MTEX's connectedComponents
# numbers the trees of the elimination tree by their roots; the padding gets no grain
def _byLastVoxel(raw, hasCell):
  grainId = np.zeros(len(raw), dtype=np.int64)
  idx = np.flatnonzero(hasCell)
  r = raw[idx]
  labels, lastRev = np.unique(r[::-1], return_index=True)
  last = len(r) - 1 - lastRev
  rank = np.empty(len(labels), dtype=np.int64)
  rank[np.argsort(last)] = np.arange(1, len(labels) + 1)
  grainId[idx] = rank[np.searchsorted(labels, r)]
  return grainId, len(labels)


# the neighbour pairs of an (n1, n2, n3) array of MATLAB indices along each axis in turn, and
# the axis of every pair
def _pairs(L):
  Dl, Dr, dim = [], [], []
  for d in range(3):
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[d], hi[d] = slice(0, -1), slice(1, None)
    Dl.append(L[tuple(lo)].ravel(order="F"))
    Dr.append(L[tuple(hi)].ravel(order="F"))
    dim.append(np.full(Dl[-1].size, d, dtype=np.int8))
  return np.concatenate(Dl), np.concatenate(Dr), np.concatenate(dim)


# the grain table and the boundary: the faces between grains, then the faces of the surface
# on the low side of every axis, then on the high side, as MATLAB orders them
def _grains(e, g, grainId, nGrains, Dl, Dr, dim, L):
  shape = L.shape
  lat = g.lattice
  isFace = grainId[Dl] != grainId[Dr]
  a, b, d = Dl[isFace], Dr[isFace], dim[isFace]
  # a face against the padding is a face of the surface of the voxel that is not padding
  padA = grainId[a] == 0
  a, b, up = np.where(padA, b, a), np.where(padA, -1, b), ~padA
  b = np.where(_gather(grainId, b, 0) == 0, -1, b)
  aa, bb, dd, uu = [a], [b], [d], [up]
  for side, up in ((0, False), (-1, True)):
    for k in range(3):
      sl = [slice(None)] * 3
      sl[k] = side
      v = L[tuple(sl)].ravel(order="F")
      v = v[grainId[v] > 0]
      aa.append(v)
      bb.append(np.full(len(v), -1))
      dd.append(np.full(len(v), k, dtype=np.int8))
      uu.append(np.full(len(v), up))
  a, b, d, up = np.concatenate(aa), np.concatenate(bb), np.concatenate(dd), np.concatenate(uu)
  # the four corners of every face, turned so that the normal points from a to b
  s = np.stack(np.unravel_index(a, shape, order="F"), axis=1)
  s[np.arange(len(s)), d] += up
  u, w = (d + 1) % 3, (d + 2) % 3
  eu = np.eye(3, dtype=np.int64)[u]
  ew = np.eye(3, dtype=np.int64)[w]
  Q = np.stack([s, s + eu, s + eu + ew, s + ew], axis=1)
  flip = ~up if np.linalg.det(lat.D) > 0 else up
  Q[flip] = Q[flip][:, ::-1]
  cshape = tuple(m + 1 for m in shape)
  Qlin = np.ravel_multi_index(Q.reshape(-1, 3).T, cshape, order="F").reshape(-1, 4)
  T = np.stack([Qlin[:, [0, 1, 2]], Qlin[:, [0, 2, 3]]], axis=1).reshape(-1, 3)
  cId, T = np.unique(T, return_inverse=True)
  T = T.reshape(-1, 3)
  c = np.stack(np.unravel_index(cId, cshape, order="F"), axis=1)
  V = lat.origin + (c - 0.5) @ lat.D
  a, b = np.repeat(a, 2), np.repeat(b, 2)
  inner = b >= 0
  bs = np.maximum(b, 0)
  m = len(a)
  f = np.arange(m)
  I_GF = csr_matrix((np.concatenate([np.ones(m), -np.ones(int(inner.sum()))]),
                     (np.concatenate([grainId[a] - 1, grainId[b[inner]] - 1]), np.concatenate([f, f[inner]]))), shape=(nGrains, m))
  ph = e.phaseId.reshape(-1)
  rot = e.rotations.data.reshape(-1, 4)
  ids = e.id.reshape(-1)
  mori = np.full((m, 4), np.nan)
  mori[inner] = qmul(qconj(rot[b[inner]]), rot[a[inner]])
  gB = grain3Boundary(V, T, np.stack([grainId[a], np.where(inner, grainId[bs], 0)], axis=1),
                      np.stack([ids[a], np.where(inner, ids[bs], -1)], axis=1),
                      np.stack([ph[a], np.where(inner, ph[bs], -1)], axis=1), mori, e.CSList, e.frame, e.scanUnit)
  numPixel = np.bincount(grainId, minlength=nGrains + 1)[1:]
  phaseId, meanRot, gos = _grainTable(grainId, ph, rot, nGrains, numPixel, e.CSList)
  return grain3d(np.arange(1, nGrains + 1), phaseId, e.CSList, numPixel, meanRot, gos, I_GF, gB, e.frame, phaseMap=e.phaseMap)


__all__ = ["calcGrains"]

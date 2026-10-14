# align a sequence of images and EBSD maps of one specimen area (MATLAB EBSDAnalysis/@trueEbsd;
# Tong and Britton, Ultramicroscopy 221, 2021; Tong et al., arXiv 2605.00703)
#
#   job = trueEbsd(imgList, T)     most distorted first, the ground truth last, one unfitted
#                                  spatial transform per hop, spatialTransformId where nothing moves
#   job.setOptions(numROI=16)      and setOptions(n, registerOn='raw') for map n alone
#   job.pixelSizeMatch()           every map on one grid
#   job.calcDistortion('fitErr')   every hop measured and fitted, stage by stage
#   job.undistort()                every map resampled onto the reference, the images a property of the map
#   job.resizedList, job.shifts[n][m], job.fitError[n], job.T[n], job.undistortedList, job.opt[n]
#
# No single distortion relates an EBSD map to an electron image, but a chain of intermediate
# images does, each pair separated by one simple enough to fit, and the hops compose. Hop n
# separates imgList[n] from imgList[n + 1]. A multi stage hop is written with +, never *, which
# absorbs an unfitted prototype as the identity. The methods alter the job, as those of
# parentGrainReconstructor do. The automatic settings, the retry and their constants are
# MATLAB's, measured there on the WC-Co sequence.
from __future__ import annotations

import warnings

import numpy as np

from ..geometry.arguments import refuseFlags
from ..geometry.spatialtransform import (spatialTransform, spatialTransformComposite, spatialTransformDrift, spatialTransformId,
                                        spatialTransformList)
from ..geometry.vector3d import vector3d
from .mapimage import mapImage, mapImageList, edgeMap
from .registration import _round, pairShifts, xcfShift
from ..geometry.display import aligned

OPTIONS = ("roiSize", "numROI", "registerOn", "edgeWidth", "highContrast", "pixelTime")


def resampleNearest(ebsd, pos, at=None):
  """the map read at positions by its nearest measurement, within the reach of its unit cell,
  as MATLAB's EBSD/interp: a position no measurement reaches is not indexed, with a NaN
  rotation and NaN properties; at, the positions the values are placed at, by default pos"""
  from scipy.spatial import cKDTree
  from ..geometry.quaternion import rotation
  from .ebsd import EBSD
  Q = np.asarray(pos.data if isinstance(pos, vector3d) else pos, dtype=float).reshape(-1, 3)
  P = ebsd.pos.data.reshape(-1, 3)
  pl = ebsd.lattice.plane if ebsd.lattice is not None else None
  P2, Q2 = (pl.to2d(P), pl.to2d(Q)) if pl is not None else (P[:, :2], Q[:, :2])
  known = np.isfinite(Q2).all(axis=1)
  dist, idx = np.full(len(Q2), np.inf), np.zeros(len(Q2), dtype=np.int64)
  dist[known], idx[known] = cKDTree(P2).query(Q2[known])
  r = np.max(np.linalg.norm(np.asarray(ebsd.unitCell.data), axis=-1)) if ebsd.lattice is not None else np.inf
  ok = dist <= r * (1 + 1e-10)
  k = idx[ok]
  n = len(Q)
  rot = np.full((n, 4), np.nan, dtype=ebsd.dtype)
  rot[ok] = ebsd.rotations.data.reshape(-1, 4)[k]
  ph = np.zeros(n, dtype=np.int8)
  ph[ok] = ebsd.phaseId.reshape(-1)[k]
  prop = {}
  for key, val in ebsd.prop.items():
    flat = val.reshape((ebsd.size,) + val.shape[ebsd.ndim:])
    out = np.full((n,) + flat.shape[1:], np.nan)
    out[ok] = flat[k]
    prop[key] = out
  if at is not None:
    Q = np.asarray(at.data if isinstance(at, vector3d) else at, dtype=float).reshape(-1, 3)
  return EBSD(Q, rotation(rot), ph, ebsd.CSList, prop, scanUnit=ebsd.scanUnit, frame=ebsd.frame, meta=ebsd.meta)


# the image of mg read where T sends each pixel of its grid, by the nearest pixel, so that no
# intensity is invented; for a map moved by T hand in inv(T)
def _resample(mg, img, T):
  m = mg.copy()
  m.img = img
  return m.interp(T.eval(mg.pos), method="nearest")


def _pxStr(v):
  return "·" if v is None or not np.isfinite(v[0]) else f"{v[0]:.2f} ({v[1]:+.2f},{v[2]:+.2f})"


class trueEbsd:
  """the alignment of a sequence of images and EBSD maps of one area, most distorted first and the
  ground truth last, one unfitted spatial transform per hop; the methods alter the job

    job = trueEbsd(imgList, T)
    job.pixelSizeMatch()
    job.calcDistortion('fitErr')
    job.undistort()"""

  def __init__(self, imgList=None, T=None):
    self.imgList = mapImageList()
    self._T = spatialTransformList()
    self.resizedList = mapImageList()
    self.undistortedList = mapImageList()
    self.shifts = []
    self.fitError = []
    self.opt = []
    if imgList is None:
      return
    imgList = mapImageList(m.copy() for m in ([imgList] if isinstance(imgList, mapImage) else imgList))
    if not all(isinstance(m, mapImage) for m in imgList):
      raise TypeError("a trueEbsd job is built from a list of mapImage; an EBSD map joins it as mapImage(ebsd.bc, ebsd)")
    _shareOneFrame(imgList)
    _shareOneUnit(imgList)
    # the name an aligned image is written under
    for k, m in enumerate(imgList):
      if not m.name:
        m.name = f"img{k}"
    self.imgList = imgList
    self.opt = [{"roiSize": "auto", "numROI": 24, "registerOn": "edge", "edgeWidth": "auto", "highContrast": "auto", "pixelTime": 0}
                for _ in imgList]
    if T is not None:
      self.T = T

  @property
  def T(self):
    """one unfitted transform per hop"""
    return self._T

  @T.setter
  def T(self, T):
    """sets the hops, one spatial transform per pair of neighbouring maps, spatialTransformId where
    nothing separates a pair"""
    T = [] if T is None else ([T] if isinstance(T, spatialTransform) else list(T))
    if not all(isinstance(t, spatialTransform) for t in T):
      raise TypeError("the hops of a trueEbsd job are spatial transforms")
    n = len(self.imgList)
    if n and T and len(T) != n - 1:
      raise ValueError(f"a {n} map sequence has {n - 1} hops, so T needs {n - 1} entries, got {len(T)}; use "
                       "spatialTransformId where nothing separates a pair - the reference has no entry of its own")
    self._T = spatialTransformList(T)

  # ------------------------------------------------------------------ the settings

  def setOptions(self, n=None, **kwargs):
    """job.setOptions(roiSize=5, numROI=24), job.setOptions(3, registerOn='raw')

    The registration settings of every map, or of map n (or a list of them) alone: roiSize, the
    tile width as a length in the scan unit, one or one per fit stage, 0 for a map correlated
    against nothing; numROI, the tiles across; registerOn, 'edge' or 'raw'; edgeWidth, the
    neighbour distance of the edge transform, a length; highContrast, True, False or 'auto';
    pixelTime, carried. What is left at 'auto' is measured when calcDistortion needs it.
    """
    idx = range(len(self.imgList)) if n is None else np.atleast_1d(n).astype(int)
    for i in idx:
      if not 0 <= i < len(self.imgList):
        raise IndexError(f"this job has {len(self.imgList)} maps, asked for map {i}")
    for name, value in kwargs.items():
      hit = next((o for o in OPTIONS if o.lower() == name.lower()), None)
      if hit is None:
        raise ValueError(f"there is no setting called {name!r}; there is " + ", ".join(OPTIONS))
      value = _checkValue(hit, value)
      for i in idx:
        self.opt[i][hit] = value

  # ------------------------------------------------------------------ one common grid

  def pixelSizeMatch(self, pixelsize=0, offsetMatch="centre", extentMatch="largest"):
    """every image and map of the job resampled onto one grid; MATLAB's pixelSizeMatch of trueEbsd

      job.pixelSizeMatch()                                   # the smallest pixel size present
      job.pixelSizeMatch(0.05)                               # another pixel size
      job.pixelSizeMatch(0.05, offsetMatch='topLeft', extentMatch='smallest')  # corners aligned, cropped

    The images are interpolated linearly, NaN outside them, the EBSD map takes its nearest
    measurement. The result is job.resizedList, which calcDistortion works on; the job is
    changed in place.

    Input
      job - trueEbsd
      pixelsize - the pixel size of the common grid; 0 for the smallest of job.imgList
    Keywords
      offsetMatch - how fields of view of different size are laid over one another: 'centre'
          aligns their centres, 'topLeft' their first corners; choose the one that leaves the
          smaller rigid shift between the images
      extentMatch - 'largest' extends every image to the largest field of view, repeating its
          edge pixels; 'smallest' crops all to the smallest
    See also
      trueEbsd.calcDistortion, trueEbsd.undistort, mapImage, EBSDTrueEbsd"""
    imgs = self.imgList
    if not pixelsize > 0:
      pixelsize = min(m.dx for m in imgs)
      print(f"using default pixel size of {pixelsize:g} um, minimum from imgList")
    if offsetMatch not in ("centre", "topLeft"):
      raise ValueError("invalid offsetMatch option: use either 'centre' or 'topLeft'")
    if extentMatch not in ("largest", "smallest"):
      raise ValueError("invalid extentMatch option: use either 'largest' or 'smallest'")
    # the positions of the pixels along the rows and the columns of every image
    v = []
    for m in imgs:
      pxn = m.dx
      vn = []
      for n in m.gridSize():
        g = pxn * np.arange(n)
        g = g - g.max() / 2 if offsetMatch == "centre" else g + pxn / 2
        vn.append(g)
      v.append(vn)
    lo = np.array([[vn[d].min() - m.dx / 2 for d in range(2)] for m, vn in zip(imgs, v)])
    hi = np.array([[vn[d].max() + m.dx / 2 for d in range(2)] for m, vn in zip(imgs, v)])
    vc = []
    for d in range(2):
      span = hi[:, d].max() - lo[:, d].min() if extentMatch == "largest" else hi[:, d].min() - lo[:, d].max()
      npix = int(np.floor(np.finfo(float).eps * 10000 + span / pixelsize))
      off = -(npix + 1) / 2 * pixelsize if offsetMatch == "centre" else -0.5 * pixelsize
      vc.append(pixelsize * np.arange(1, npix + 1) + off)
    G1, G2 = np.meshgrid(vc[0], vc[1], indexing="ij")
    out = mapImageList()
    for m, vn in zip(imgs, v):
      imgNew = _gridInterp(vn, m.img, vc)
      gL = m.layout
      b1, b2 = gL.row.data.reshape(3), gL.col.data.reshape(3)
      if m.ebsd is not None:
        from .gridify import gridify
        e = m.ebsd
        src = e.pos.data[0, 0] + (G1 - vn[0][0])[..., None] * b1 + (G2 - vn[1][0])[..., None] * b2
        pos = (G2[..., None] * b2 + G1[..., None] * b1).reshape(-1, 3)
        eNew = gridify(resampleNearest(e, src.reshape(-1, 3), at=pos), gL)
        new = mapImage(imgNew, eNew, name=m.name)
      else:
        new = mapImage(imgNew, name=m.name)
        new.frame = m.frame
        new._d1, new._d2 = pixelsize * b1, pixelsize * b2
        new._origin = vc[1][0] * b2 + vc[0][0] * b1
      new.scanUnit = m.scanUnit
      out.append(new)
    self.resizedList = out

  # ------------------------------------------------------------------ the fit

  def calcDistortion(self, *flags, fitErr=False, verbose=True, retryMax=100):
    """the distortion between every consecutive pair of maps of the job fitted; MATLAB's
    calcDistortion of trueEbsd

      job.calcDistortion()                                   # every hop fitted
      job.calcDistortion(fitErr=True)                        # and what each hop leaves over measured
      job.calcDistortion(fitErr=True, verbose=False, retryMax=1)  # silent, one pass per hop

    job.T[n] is the unfitted spatialTransform between map n and map n + 1. For every pair the
    images are cut into tiles, the tiles cross correlated for a local shift each, and the stages
    of the transform fitted one after another to those shifts, each to what the one before left
    over. The shifts land in job.shifts[n], and job.T[n] is replaced by the fitted transform,
    still of its declared class. The job is changed in place; run pixelSizeMatch first.

    The pairs are visited from the first map of high contrast forward to the reference, then
    backward from it to map 0, so that one map of every pair has usable contrast; where the
    other has none, its partner's image carried into its frame is what the next hop correlates.

    fitErr measures the displacement left after each hop, job.fitError[n], the number that says
    whether the registration worked; it costs one more correlation per hop. With it a hop that
    leaves more than 2 pixels is redone with the tiles doubled, until it converges, the tiles no
    longer fit the image or retryMax attempts are made.

    Input
      job - trueEbsd after pixelSizeMatch
    Keywords
      fitErr - measure what is left after each hop and retry the hops that leave too much
      verbose - print the table of the hops, MATLAB's silent the opposite
      retryMax - the most correlation attempts per hop, the first included
    See also
      trueEbsd.pixelSizeMatch, trueEbsd.undistort, spatialTransform, xcfShift, pairShifts,
      EBSDTrueEbsd"""
    refuseFlags(flags, "calcDistortion")
    retryMax = max(int(retryMax), 1)
    nImg = len(self.resizedList)
    if nImg < 2:
      raise ValueError("calcDistortion needs at least two maps")
    if any(m.isempty for m in self.resizedList):
      raise ValueError("job.resizedList is empty - run pixelSizeMatch first")
    if len(self.T) != nImg - 1:
      raise ValueError(f"{nImg} maps means {nImg - 1} hops, but job.T has {len(self.T)} entries; set one unfitted "
                       "spatial transform per hop, spatialTransformId where nothing separates a pair")
    self.opt, regImg, notes = _autoTune(self)
    self.shifts = [[] for _ in range(nImg - 1)]
    if fitErr:
      self.fitError = [pairShifts() for _ in range(nImg - 1)]
    nStart = next((k for k, o in enumerate(self.opt) if o["highContrast"] is True), 0)
    colw = (12, 13, 8, 18, 18)
    # the map the printed walk stands at
    at = ""

    def say(s):
      if verbose:
        print(s)

    def rowStr(c):
      return " ".join(f"{x:<{w}}" if k < 2 else f"{x:>{w}}" for k, (x, w) in enumerate(zip(c, colw))).rstrip()

    def sayRow(*c):
      say(f"  │ {rowStr(c)}".rstrip())

    def nameOf(n):
      return self.resizedList[n].name

    def sayFrom(n):
      nonlocal at
      if at == nameOf(n):
        return
      if at:
        say("")
      say(f"  ● {nameOf(n)}")
      at = nameOf(n)

    def runHop(n, forward, carryIn):
      nonlocal at
      test, ref = self.resizedList[n], self.resizedList[n + 1]
      xcf = _roiSettings(self.opt[n], test.dx)
      imTest0, imRef0 = regImg[n], regImg[n + 1]
      if carryIn is not None:
        if forward:
          imTest0 = carryIn
        else:
          imRef0 = carryIn
      proto = self.T[n]
      stages = proto.stageList()
      nStage = len(stages)
      isTrue = nStage == 0
      carryOut = None
      sayFrom(n)
      for _ in range(retryMax):
        imTest = imTest0
        self.shifts[n] = [pairShifts() for _ in range(max(nStage, 1))]
        fitted = []
        dist = proto.shortChar()
        if isTrue:
          sayRow(dist, "·", "·", _pxStr(np.zeros(3)), "")
          dist = ""
        else:
          for m in range(nStage):
            k = min(len(xcf), m + 1) - 1
            self.shifts[n][m] = _measureShift(ref, imRef0, test, imTest, xcf[k])
            fitted.append(_fitTo(stages[m], self.shifts[n][m]))
            sayRow(dist, fitted[m].shortChar(), f"{xcf[k]['ROISize']} px", _pxStr(self.shifts[n][m].meanShift(test.dx)), "")
            dist = ""
            imTest = _resample(test, imTest, fitted[m].inv())
          self._T[n] = _refitted(proto, fitted)
        if forward and self.opt[n + 1]["highContrast"] is not True:
          carryOut = imTest
        elif not forward and self.opt[n]["highContrast"] is not True:
          carryOut = _resample(ref, imRef0, self._T[n])
        if not fitErr:
          break
        self.fitError[n] = _measureShift(ref, imRef0, test, imTest, xcf[-1])
        resid = self.fitError[n].meanShift(test.dx)
        sayRow(dist, "↳ difference" if isTrue else "↳ residual", "", "", _pxStr(resid))
        if isTrue or resid[0] <= 2:
          break
        grown = True
        for k in range(min(len(xcf), nStage)):
          newSize = 2 * xcf[k]["ROISize"]
          r, c = test.gridSize()
          if newSize >= min(c - xcf[k]["NumROI"], r - xcf[k]["NumROI"]):
            say(f"  │ ⚠ residual over 2 px, but a {newSize} px ROI does not fit the image — keeping this fit")
            grown = False
            break
          xcf[k]["ROISize"] = newSize
        if not grown:
          break
        say(f"  │ ↻ residual over 2 px, retrying with a {xcf[0]['ROISize']} px ROI")
        self.opt[n]["roiSize"] = np.array([x["ROISize"] for x in xcf], dtype=float) * test.dx
      say(f"  ▼\n  ● {nameOf(n + 1)}")
      at = nameOf(n + 1)
      return carryOut

    if verbose:
      _sayAuto(notes)
      say(f"\n ◆ distortion across {nImg} maps, {nImg - 1} hops\n")
      say(f"    {rowStr(('distortion', 'stage', 'ROI', 'shift, px', 'residual, px'))}")
      say(f"    {rowStr(tuple('─' * w for w in colw))}")
    carry = None
    for n in range(nStart, nImg - 1):
      carry = runHop(n, True, carry)
    carry = None
    for n in range(nStart - 1, -1, -1):
      carry = runHop(n, False, carry)
    say("")

  # ------------------------------------------------------------------ the correction

  def undistort(self):
    """job.undistort(): every map resampled onto the reference grid through the hops between
    it and the reference, its own first, by the nearest pixel; every aligned image then a
    property of every map that carries an EBSD map, under its name"""
    from .gridify import gridify
    nImg = len(self.resizedList)
    if len(self.shifts) + 1 != nImg:
      raise ValueError(f"job has {nImg} maps but {len(self.shifts)} hops of shifts - run calcDistortion first, or check that it completed")
    out = mapImageList(m.copy() for m in self.resizedList)
    T = spatialTransformId()
    for n in range(nImg - 2, -1, -1):
      T = T * self.T[n]
      mg = self.resizedList[n]
      src = T.inv().eval(mg.pos)
      out[n].img = mg.interp(src, method="nearest")
      if mg.ebsd is not None:
        out[n].ebsd = gridify(resampleNearest(mg.ebsd, src, at=mg.ebsd.pos.data.reshape(-1, 3)), mg.layout)
    for n in range(nImg):
      if out[n].ebsd is None:
        continue
      e = out[n].ebsd
      for m in range(nImg):
        fn = out[m].name
        if fn in e.prop:
          warnings.warn(f"map {n} already carries a property {fn!r} - the aligned image of map {m} is overwriting it; "
                        "give that map a distinct name to keep both")
        e.prop[fn] = out[m].img
    self.undistortedList = out

  # ------------------------------------------------------------------ the display

  def _stage(self):
    if not self.imgList:
      return "empty"
    if self.undistortedList:
      return "undistorted"
    if self.shifts:
      return "shifts calculated"
    if self.resizedList:
      return "pixel size matched"
    return "as imported"

  def __repr__(self):
    head = f"trueEbsd ({self._stage()})"
    n = len(self.imgList)
    if n == 0:
      return aligned(head + "\n  no maps here!")
    lst = self.undistortedList or self.resizedList or self.imgList
    rows = []
    for k in range(n):
      m = lst[k]
      rows.append((str(k), m.name, " × ".join(str(s) for s in m.img.shape), self.T[k].shortChar() if k < len(self.T) else "",
                   self._hopStr(k, "shifts"), self._hopStr(k, "fitError")))
    labels = ("", "name", "image", "distortion", "shift, px", "residual, px")
    w = [max(len(labels[j]), *(len(r[j]) for r in rows)) for j in range(len(labels))]
    def line(r):
      return "  " + "  ".join(x.ljust(q) for x, q in zip(r, w)).rstrip()
    out = [head, "", line(labels)] + [line(r) for r in rows]
    if self.resizedList:
      d = self.resizedList[0]
      out += ["", f" common grid: {' × '.join(str(s) for s in d.img.shape)} at {d.dx:.4g} µm"]
    return aligned("\n".join(out))

  def _hopStr(self, k, what):
    if k >= len(self.imgList) - 1:
      return "-"
    if what == "shifts":
      if k >= len(self.shifts) or not self.shifts[k]:
        return "-"
      ps = self.shifts[k][-1]
    else:
      if k >= len(self.fitError):
        return "-"
      ps = self.fitError[k]
    if ps.isempty:
      return "-"
    return _pxStr(ps.meanShift(self.resizedList[k].dx))


# the image values at the common grid, linear on the grid of positions vn, NaN outside:
# MATLAB's griddedInterpolant(..., 'linear', 'none')
def _gridInterp(vn, img, vc):
  def weights(g, q):
    i = np.clip(np.searchsorted(g, q, side="right") - 1, 0, max(len(g) - 2, 0))
    j = np.minimum(i + 1, len(g) - 1)
    t = np.where(j > i, (q - g[i]) / np.where(j > i, g[j] - g[i], 1), 0.0)
    out = (q < g[0] - 1e-12 * max(1, abs(g[0]))) | (q > g[-1] + 1e-12 * max(1, abs(g[-1])))
    return i, j, t, out
  i0, i1, s, o1 = weights(vn[0], vc[0])
  j0, j1, t, o2 = weights(vn[1], vc[1])
  A = img.reshape(img.shape[:2] + (-1,))
  s, t = s[:, None, None], t[None, :, None]
  v = (1 - s) * ((1 - t) * A[i0][:, j0] + t * A[i0][:, j1]) + s * ((1 - t) * A[i1][:, j0] + t * A[i1][:, j1])
  v[o1] = np.nan
  v[:, o2] = np.nan
  return v.reshape((len(vc[0]), len(vc[1])) + img.shape[2:])


def _checkValue(name, value):
  if name in ("roiSize", "edgeWidth"):
    if isinstance(value, str):
      if value.lower() != "auto":
        raise ValueError(f"{name} is a length, or 'auto'; got {value!r}")
      return "auto"
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if (value < 0).any():
      raise ValueError(f"{name} is a length in the map's own unit, so it cannot be negative")
    return value if value.size > 1 else float(value[0])
  if name == "numROI":
    if not (np.isscalar(value) and value >= 1):
      raise ValueError("numROI is how many tiles fit across the image, one number - the count down follows from the aspect ratio")
    return int(value)
  if name == "registerOn":
    value = str(value).lower()
    if value not in ("edge", "raw"):
      raise ValueError(f"registerOn is 'edge' or 'raw'; got {value!r}")
    return value
  if name == "highContrast":
    if isinstance(value, str):
      if value.lower() != "auto":
        raise ValueError(f"highContrast is True, False or 'auto'; got {value!r}")
      return "auto"
    return bool(value)
  return value


# one job is one sample area: every entry stored along the same directions and measured in
# one unit, which is checked here and never guessed
def _shareOneFrame(imgList):
  if len(imgList) < 2:
    return
  ref = imgList[0].layout
  for k in range(1, len(imgList)):
    gL = imgList[k].layout
    if not gL == ref:
      raise ValueError(f"images 0 and {k} are stored along different directions, so they cannot be compared pixel by pixel\n"
                       f"  image 0: rows run {ref.row}, columns run {ref.col}\n  image {k}: rows run {gL.row}, columns run {gL.col}\n"
                       "bring them into one frame first, with transformReferenceFrame")


def _shareOneUnit(imgList):
  u = [m.scanUnit for m in imgList]
  bad = next((k for k in range(len(u)) if u[k] != u[0]), None)
  if bad is not None:
    raise ValueError(f"image 0 is in {u[0]} and image {bad} is in {u[bad]}, so no length given to the job means the same "
                     "thing to both; put the sequence in one unit before building it")


# the length valued settings as one pixel valued setting per fit stage, an even tile
def _roiSettings(opt, px):
  roi = np.maximum(_round(np.atleast_1d(opt["roiSize"]) / px), 0)
  roi = 2 * _round(roi / 2)
  return [{"ROISize": int(r), "NumROI": opt["numROI"]} for r in roi]


# the shifts between two images on one grid, u pointing from where a tile of the test image
# is to where it is in the reference
def _measureShift(mgRef, imRef, mgTest, imTest, roi):
  a, b = mgRef.copy(), mgTest.copy()
  a.img, b.img = imRef, imTest
  u, peak, pos = xcfShift(a, b, ROISize=roi["ROISize"], numROI=roi["NumROI"])
  return pairShifts(pos, -u, peak, roi["ROISize"])


# the transform of the prototype's class fitted to the shifts, weighted by the peaks; a drift
# continues its end segments beyond the first and the last row of tiles
def _fitTo(proto, ps):
  kw = dict(proto.fitOptions())
  if isinstance(proto, spatialTransformDrift):
    kw["extrapolate"] = True
  return type(proto).fit(ps.pos, ps.pos + ps.u, weights=ps.peak, **kw)


# the fitted stages in the class the hop was declared as
def _refitted(proto, fitted):
  if isinstance(proto, spatialTransformComposite):
    out = type(proto).__new__(type(proto))
    out.__dict__.update(proto.__dict__)
    out.stages = list(fitted)
    return out
  return fitted[0]


# ------------------------------------------------------------------ the automatic settings

# what was left at 'auto', measured: the edge width of every image where the variance of its
# edge transform saturates, the tile width of every hop from the shift a coarse correlation
# measures and the scale of the image's features. registerOn is stated, not measured
def _autoTune(job):
  opt = [dict(o) for o in job.opt]
  imgs = job.resizedList
  nImg = len(imgs)
  notes = {"edgePx": np.full(nImg, np.nan), "roiPx": np.full(nImg - 1, np.nan), "shiftPx": np.full(nImg - 1, np.nan),
           "featPx": np.full(nImg - 1, np.nan)}
  for n in range(nImg):
    if isinstance(opt[n]["edgeWidth"], str):
      w = _bestEdgeWidth(imgs[n])
      opt[n]["edgeWidth"] = w * imgs[n].dx
      notes["edgePx"][n] = w
  edgeOf, rawOf = [], []
  for n in range(nImg):
    padPx = max(1, int(_round(float(np.max(opt[n]["edgeWidth"])) / imgs[n].dx)))
    edgeOf.append(edgeMap(imgs[n], padPx))
    raw = imgs[n].img
    rawOf.append(raw.mean(axis=2) if raw.ndim == 3 else raw)
  for n in range(nImg - 1):
    if isinstance(job.T[n], spatialTransformId):
      continue
    if opt[n]["registerOn"] != opt[n + 1]["registerOn"]:
      raise ValueError(f"hop {n} correlates map {n} against map {n + 1}, but one is set to {opt[n]['registerOn']!r} and the "
                       f"other to {opt[n + 1]['registerOn']!r}; an edge transform and raw values have nothing to match on")
  regImg = [edgeOf[n] if opt[n]["registerOn"] == "edge" else rawOf[n] for n in range(nImg)]
  for n in range(nImg - 1):
    if not isinstance(opt[n]["roiSize"], str):
      continue
    sizePx, shiftPx, featPx = _bestRoiSize(regImg[n], regImg[n + 1], opt[n]["numROI"], imgs[n].gridSize())
    opt[n]["roiSize"] = sizePx * imgs[n].dx
    notes["roiPx"][n], notes["shiftPx"][n], notes["featPx"][n] = sizePx, shiftPx, featPx
  if isinstance(opt[-1]["roiSize"], str):
    opt[-1]["roiSize"] = 0.0
  for n in range(nImg):
    if isinstance(opt[n]["highContrast"], str):
      opt[n]["highContrast"] = True
  return opt, regImg, notes


def _sayAuto(a):
  if not (np.isfinite(a["edgePx"]).any() or np.isfinite(a["roiPx"]).any()):
    return

  def vec(v, dec):
    return " ".join("·" if not np.isfinite(x) else f"{x:.{dec}f}" for x in v)
  print("\n ◆ measured settings, override with setOptions")
  if np.isfinite(a["edgePx"]).any():
    print(f"   ▸ edgeWidth  per map  {vec(a['edgePx'], 0)} px")
  if np.isfinite(a["roiPx"]).any():
    print(f"   ▸ roiSize    per hop  {vec(a['roiPx'], 0)} px   ← shifts {vec(a['shiftPx'], 1)} px, features {vec(a['featPx'], 0)} px")


# the neighbour distance past which the variance of the edge transform stops rising, on the
# central 512 pixels at full resolution: the narrowest width within a tenth of the plateau
def _bestEdgeWidth(mg):
  def centred(n, want):
    if n <= want:
      return np.arange(n)
    first = (n - want) // 2
    return np.arange(first, first + want)
  r, c = mg.gridSize()
  probe = mg.subGrid(centred(r, 512), centred(c, 512))
  v = np.zeros(7)
  for k in range(1, 8):
    e = edgeMap(probe, k).reshape(-1)
    e = e[np.isfinite(e)]
    v[k - 1] = np.var(e, ddof=1) if e.size > 1 else 0.0
  w = np.flatnonzero(v >= 0.9 * v.max())
  return int(w[0]) + 1 if w.size else 1


# four times the shift a coarse correlation of few large tiles measures, floored by eight
# times the feature scale, since the window leaves only the central quarter of a tile
# contributing; a power of two, and no larger than the image can hold a grid of
def _bestRoiSize(a, b, numROI, sz):
  roi = 2 ** np.floor(np.log2(min(a.shape) / 3))
  roi = max(min(roi, 256), 32)
  u, peak, _ = xcfShift(a, b, ROISize=roi, numROI=max(4, int(_round(numROI / 4))))
  ok = np.isfinite(peak)
  shiftPx = float(np.median(np.asarray(u.norm()).reshape(-1)[ok])) if ok.any() else 0.0
  featPx = _featureScale(a)
  wanted = max(4 * shiftPx, 4 * 2 * featPx)
  sizePx = 2 ** np.ceil(np.log2(max(wanted, 16)))
  sizePx = min(sizePx, 2 ** np.floor(np.log2(min(sz) / 2)))
  return int(sizePx), shiftPx, featPx


# the half width of the autocorrelation peak along the middle row, the scale of the image's
# structure in pixels
def _featureScale(img):
  from scipy import fft
  img = img - np.nanmean(img)
  img = np.where(np.isfinite(img), img, 0.0)
  ac = np.fft.fftshift(np.real(fft.ifft2(np.abs(fft.fft2(img)) ** 2)))
  ac = ac / ac.max()
  mid = ac[int(_round(ac.shape[0] / 2))]
  c = int(_round(len(mid) / 2))
  k = np.flatnonzero(mid[c:] < 0.5)
  return int(k[0]) + 1 if k.size else 1


__all__ = ["trueEbsd"]

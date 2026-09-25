# the parts of plot(ebsd, 'orientation') timed one by one, for docs/research/benchmark-mapplot.md:
# the inverse pole figure colours, the raster of the map, and the drawing on the Agg canvas
# and on the screen
#
#   .venv/bin/python bench/mapplot_parts.py                the colours, the raster and the Agg drawing
#   .venv/bin/python bench/mapplot_parts.py --screen QtAgg  the drawing on the screen through one backend
#
# Every number is the best of three after a warm-up run; the maps are those of bench/mapplot.py.
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "/home/hielscher/mtex/frameSymmetry/data/EBSD"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FILES = [("Forsterite", os.path.join(D, "Forsterite.ctf")), ("Forsterite 2x2", os.path.join(out, "forsterite_2x2.ctf")),
         ("Forsterite 4x4", os.path.join(out, "forsterite_4x4.ctf"))]


def timed(fun, repeat=3):
  fun()
  best, r = np.inf, None
  for _ in range(repeat):
    t = time.perf_counter()
    r = fun()
    best = min(best, time.perf_counter() - t)
  return best, r


def row(name, n, **parts):
  print(f"{name},{n}," + ",".join(f"{k}={v * 1000:.1f}ms" for k, v in parts.items()), flush=True)


# ------------------------------------------------------------------ the colours

def colourParts(ebsd, name):
  from mtex.geometry.kernels import qrot
  from mtex.plotting.kernels import cubeMapIndex
  from mtex.plotting.orientationkeys import _conj, ipfColors, keyColorGrid
  n = ebsd.size
  q = ebsd.rotations.data.reshape(-1, 4)
  cs = ebsd.CSList[1]
  at = np.flatnonzero(ebsd.phaseId.reshape(-1) == 1)
  d = np.array([0.0, 0.0, 1.0])
  grid = keyColorGrid(cs)
  tRot, dirs = timed(lambda: qrot(_conj(q[at]), np.tile(d, (len(at), 1))))
  tLook, ind = timed(lambda: cubeMapIndex(dirs, grid.n))
  tGather, _ = timed(lambda: grid.colors[np.asarray(ind).reshape(-1)])
  tGrid, _ = timed(lambda: ipfColors(ebsd))
  tKernel, rgb = timed(lambda: ipfKernelColors(q, at, grid))
  ref = grid.colors[np.asarray(ind).reshape(-1)]
  same = np.mean(np.all(np.abs(rgb - ref) < 1e-12, axis=1))
  row(name, n, rotate=tRot, lookup=tLook, gather=tGather, ipfColors=tGrid, numba=tKernel)
  print(f"  the numba kernel agrees on {100 * same:.2f} % of the points", flush=True)


# the grid route as the one compiled loop of plotting/kernels.py: the crystal direction of the
# specimen direction, its node of the cube map, the colour
def ipfKernelColors(q, at, grid):
  from mtex.plotting.kernels import ipfGridColors
  out = np.empty((len(q), 3))
  ipfGridColors(q, at, np.array([0.0, 0.0, 1.0]), grid.n, grid.colors, out)
  return out[at]


# ------------------------------------------------------------------ the raster

def rasterParts(ebsd, name):
  from mtex.plotting.maps import latticeTable, mapRaster
  from mtex.plotting.orientationkeys import ipfColors
  n = ebsd.size
  tTable, lin = timed(lambda: latticeTable(ebsd))
  tRaster, r = timed(lambda: mapRaster(ebsd))
  (xlo, xhi), (ylo, yhi), index = r
  tDirect, direct = timed(lambda: alignedRaster(ebsd))
  same = direct.shape == index.shape and np.array_equal(direct, index)
  values = ipfColors(ebsd).reshape(-1, 3)
  covered = index >= 0

  def floatImage():
    img = np.zeros(index.shape + (4,))
    img[covered, :3] = values[index[covered]]
    img[covered, 3] = 1.0
    return img

  def byteImage():
    rgb = (np.clip(values, 0, 1) * 255 + 0.5).astype(np.uint8)
    img = np.zeros(index.shape + (4,), dtype=np.uint8)
    img[..., :3] = rgb[np.where(covered, index, 0)]
    img[..., 3] = np.where(covered, 255, 0)
    return img

  tFloat, _ = timed(floatImage)
  tByte, _ = timed(byteImage)
  row(name, n, latticeTable=tTable, mapRaster=tRaster, directIndex=tDirect, floatImage=tFloat, byteImage=tByte)
  print(f"  picture {index.shape[1]} x {index.shape[0]} pixels, the direct index {'agrees' if same else 'DIFFERS'}", flush=True)


# a lattice aligned with the screen has its picture in the lattice table itself: the
# pixel (row, column) is the site (i, j) up to the direction of the axes
def alignedRaster(ebsd):
  from mtex.plotting.maps import latticeTable
  l = ebsd.lattice
  lin = latticeTable(ebsd)
  A = l.A
  # which lattice index runs along the screen x and which along y, and in which direction
  if abs(A[0, 0]) >= abs(A[0, 1]):
    img = lin.T if True else lin
    sx, sy = np.sign(A[0, 0]), np.sign(A[1, 1])
  else:
    img = lin
    sx, sy = np.sign(A[0, 1]), np.sign(A[1, 0])
  if sx < 0:
    img = img[:, ::-1]
  if sy < 0:
    img = img[::-1, :]
  return np.ascontiguousarray(img)


# ------------------------------------------------------------------ the drawing

def drawParts(ebsd, name, backend):
  import matplotlib
  matplotlib.use(backend)
  import matplotlib.pyplot as plt
  from mtex import gcm, plot
  from mtex.plotting.maps import mapRaster
  from mtex.plotting.mtexfigure import mtexFigure
  from mtex.plotting.orientationkeys import ipfColors
  n = ebsd.size
  values = ipfColors(ebsd).reshape(-1, 3)
  (xlo, xhi), (ylo, yhi), index = mapRaster(ebsd)
  covered = index >= 0
  img = np.zeros(index.shape + (4,))
  img[covered, :3] = values[index[covered]]
  img[covered, 3] = 1.0
  img8 = (img * 255 + 0.5).astype(np.uint8)
  screen = backend not in ("Agg", "agg")

  def figure():
    plt.close("all")
    fig = mtexFigure(1)
    return fig, fig.nextAxis()

  tFigure, (fig, ax) = timed(figure)
  tImshow, h = timed(lambda: ax.imshow(img, extent=(xlo, xhi, ylo, yhi), origin="lower", interpolation="nearest"))
  tDraw, _ = timed(lambda: fig.fig.canvas.draw())
  h.remove()
  h8 = ax.imshow(img8, extent=(xlo, xhi, ylo, yhi), origin="lower", interpolation="nearest")
  tDraw8, _ = timed(lambda: fig.fig.canvas.draw())
  parts = dict(figure=tFigure, imshow=tImshow, draw=tDraw, draw_uint8=tDraw8)
  if screen:
    def show():
      fig.fig.show()
      fig.fig.canvas.draw()
      fig.fig.canvas.flush_events()
    tShow, _ = timed(show)
    x0, x1 = ax.get_xlim()

    def zoom():
      ax.set_xlim(x0 + 0.3 * (x1 - x0), x1 - 0.3 * (x1 - x0))
      fig.fig.canvas.draw()
      fig.fig.canvas.flush_events()
      ax.set_xlim(x0, x1)
      fig.fig.canvas.draw()
      fig.fig.canvas.flush_events()
    tZoom, _ = timed(zoom)
    parts.update(show=tShow, zoom_pair=tZoom)
  else:
    png = os.path.join(out, "parts.png")
    tSave, _ = timed(lambda: fig.savefig(png, dpi=150))
    tSave1, _ = timed(lambda: fig.savefig(png, dpi=150, pil_kwargs={"compress_level": 1}))
    tWhole, _ = timed(lambda: (plt.close("all"), plot(ebsd, "orientation"), gcm().fig.canvas.draw()))
    parts.update(savefig=tSave, savefig_level1=tSave1, whole_plot_and_draw=tWhole)
  plt.close("all")
  row(f"{name} [{backend}]", n, **parts)


def main():
  from mtex import loadEBSD
  if "--screen" in sys.argv:
    backend = sys.argv[sys.argv.index("--screen") + 1]
    for name, f in FILES:
      drawParts(loadEBSD(f), name, backend)
    return
  maps = [(name, loadEBSD(f)) for name, f in FILES]
  print("== the colours of the largest phase (rotate, look up, gather) and of the map")
  for name, e in maps:
    colourParts(e, name)
  print("== the raster")
  for name, e in maps:
    rasterParts(e, name)
  print("== the drawing on the Agg canvas")
  for name, e in maps:
    drawParts(e, name, "Agg")
  for backend in ("QtAgg", "TkAgg"):
    print(f"== the drawing on the screen through {backend}")
    r = subprocess.run([sys.executable, __file__, "--screen", backend], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-600:])


if __name__ == "__main__":
  main()

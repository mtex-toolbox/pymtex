# the inverse pole figure colouring and the plot of large EBSD maps, against the Julia port
# and MATLAB, for docs/research/benchmark-mapplot.md
#
#   .venv/bin/python bench/mapplot.py                      writes the tiled maps the other two read
#   (cd ../mtexJulia && julia -t auto --project=dev ../pymtex/bench/mapplot.jl)
#   bench/matlab/bench_mapplot.sh
#
# The Forsterite map, the Emsland map and the Forsterite map tiled two by two and four by
# four; the colours through the grid of the key and exactly, the picture of the map, and
# the map drawn and written to a PNG; best of three after a warm-up run.
import os
import sys
import time

import matplotlib
import matplotlib.pyplot
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtex import EBSD, calcGrains, degree, export, gcm, ipfColors, ipfHSVKey, loadEBSD, plot  # noqa: E402
from mtex.plotting.maps import mapRaster  # noqa: E402

D = "/home/hielscher/mtex/frameSymmetry/data/EBSD"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
DPI = 150


def timed(fun, repeat=3):
  fun()
  best, r = np.inf, None
  for _ in range(repeat):
    t = time.perf_counter()
    r = fun()
    best = min(best, time.perf_counter() - t)
  return best, r


# the map tiled k by k, every copy shifted by the extent of the map
def tiled(ebsd, k):
  ext = ebsd.extent()
  w, h = ext[1] - ext[0] + ebsd.lattice.dxy, ext[3] - ext[2] + ebsd.lattice.dxy
  return EBSD.cat(*[ebsd.shift([i * w, j * h, 0.0]) for j in range(k) for i in range(k)])


def render(ebsd, name, **kw):
  matplotlib.pyplot.close("all")
  plot(ebsd, "orientation", **kw)
  return gcm()


def main():
  print(f"python {sys.version.split()[0]}, numpy {np.__version__}, matplotlib {matplotlib.__version__}")
  print("map,n,colors grid,colors exact,key exact,raster,plot,save,boundaries")
  fo = loadEBSD(os.path.join(D, "Forsterite.ctf"))
  maps = [("Forsterite", fo), ("Emsland", loadEBSD(os.path.join(D, "Emsland_plessite_500x_15.ctf")))]
  for k in (2, 4):
    m = tiled(fo, k)
    assert m.lattice is not None, "the tiled map lost its lattice"
    export(m, os.path.join(out, f"forsterite_{k}x{k}.ctf"))
    maps.append((f"Forsterite {k}x{k}", m))
  rows = []
  for name, ebsd in maps:
    n = ebsd.size
    tGrid, _ = timed(lambda: ipfColors(ebsd))
    tExact, _ = timed(lambda: ipfColors(ebsd, exact=True), repeat=1 if n > 2_000_000 else 3)
    # the key itself on the orientations of the largest phase
    big = max(ebsd.CSList.indexedPhaseId, key=lambda i: np.count_nonzero(ebsd.phaseId == i))
    ori = ebsd[ebsd.phaseId == big].orientations
    key = ipfHSVKey(ori.CS)
    tKey, _ = timed(lambda: key.orientation2color(ori), repeat=1 if n > 2_000_000 else 3)
    tRaster, _ = timed(lambda: mapRaster(ebsd))
    tPlot, fig = timed(lambda: render(ebsd, name), repeat=1 if n > 2_000_000 else 3)
    png = os.path.join(out, f"mapplot_{name.replace(' ', '_')}.png")
    tSave, _ = timed(lambda: fig.savefig(png, dpi=DPI), repeat=1 if n > 2_000_000 else 3)
    tB = np.nan
    if n < 500_000:
      grains = calcGrains(ebsd, angle=10 * degree)
      def overlay():
        fig = render(ebsd, name)
        plot(grains.boundary, lineWidth=0.5)
        fig.savefig(os.path.join(out, f"mapplot_{name.replace(' ', '_')}_grains.png"), dpi=DPI)
      tB, _ = timed(overlay)
    row = (name, n, tGrid, tExact, tKey, tRaster, tPlot, tSave, tB)
    print(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row), flush=True)
    rows.append(row)
  with open(os.path.join(out, "mapplot_python.csv"), "w") as fh:
    for row in rows:
      fh.write(",".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
  main()

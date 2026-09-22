# the map plot on maps of sixteen and thirty five million points, tiled in memory from the
# Forsterite map, for docs/research/benchmark-mapplot.md: the colours, the whole plot on the
# Agg canvas, the file, and the window with a zoom on the screen; and the memory it takes
#
#   .venv/bin/python bench/mapplot_large.py
#   .venv/bin/python bench/mapplot_large.py --screen QtAgg
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "/home/hielscher/mtex/frameSymmetry/data/EBSD"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TILES = (8, 12)


def rss():
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def timed(fun, repeat=2):
  fun()
  best, r = np.inf, None
  for _ in range(repeat):
    t = time.perf_counter()
    r = fun()
    best = min(best, time.perf_counter() - t)
  return best, r


def tiled(ebsd, k):
  from mtex import EBSD
  ext = ebsd.extent()
  w, h = ext[1] - ext[0] + ebsd.lattice.dxy, ext[3] - ext[2] + ebsd.lattice.dxy
  return EBSD.cat(*[ebsd.shift([i * w, j * h, 0.0]) for j in range(k) for i in range(k)])


def main():
  import matplotlib
  screen = "--screen" in sys.argv
  matplotlib.use(sys.argv[sys.argv.index("--screen") + 1] if screen else "Agg")
  import matplotlib.pyplot as plt
  from mtex import gcm, ipfColors, loadEBSD, plot
  fo = loadEBSD(os.path.join(D, "Forsterite.ctf"))
  for k in TILES:
    t = time.perf_counter()
    ebsd = tiled(fo, k)
    tBuild = time.perf_counter() - t
    n = ebsd.size
    m0 = rss()
    if screen:
      def show():
        plt.close("all")
        plot(ebsd, "orientation")
        fig = gcm()
        fig.fig.show()
        fig.fig.canvas.draw()
        fig.fig.canvas.flush_events()
        return fig
      tShow, fig = timed(show, 1)
      ax = fig.fig.axes[0]
      x0, x1 = ax.get_xlim()

      def zoom():
        ax.set_xlim(x0 + 0.3 * (x1 - x0), x1 - 0.3 * (x1 - x0))
        fig.fig.canvas.draw()
        fig.fig.canvas.flush_events()
        ax.set_xlim(x0, x1)
        fig.fig.canvas.draw()
        fig.fig.canvas.flush_events()
      tZoom, _ = timed(zoom)
      print(f"{n},{k}x{k},build={tBuild:.2f}s,plot+show={tShow:.2f}s,zoom_pair={tZoom:.2f}s,rss={rss():.1f}GB", flush=True)
      plt.close("all")
      continue
    tColors, _ = timed(lambda: ipfColors(ebsd))

    def whole():
      plt.close("all")
      plot(ebsd, "orientation")
      gcm().fig.canvas.draw()
      return gcm()
    tPlot, fig = timed(whole)
    tSave, _ = timed(lambda: fig.savefig(os.path.join(out, f"mapplot_{k}x{k}.png"), dpi=150), 1)
    print(f"{n},{k}x{k},build={tBuild:.2f}s,colors={tColors:.2f}s,plot+draw={tPlot:.2f}s,save={tSave:.2f}s,rss_before={m0:.1f}GB,rss={rss():.1f}GB", flush=True)
    plt.close("all")
    del ebsd
  if not screen:
    r = subprocess.run([sys.executable, __file__, "--screen", "QtAgg"], capture_output=True, text=True)
    print("== on the screen through QtAgg"); print(r.stdout.strip() or r.stderr.strip()[-800:])


if __name__ == "__main__":
  main()

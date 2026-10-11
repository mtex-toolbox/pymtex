# the crosscheck operations of bench/*_ops.py timed in several MTEX versions and in this port, on
# this machine with each one's default threading: the MATLAB half of each run in a MATLAB of its
# own on that version's MTEX, an operation a version cannot run left out
#
#   .venv/bin/python bench/mtex_versions.py python [module ...]    the inputs and this port's times
#   .venv/bin/python bench/mtex_versions.py matlab VERSION [module ...]
#   .venv/bin/python bench/mtex_versions.py table                  docs/research/mtex-versions.md
#   .venv/bin/python bench/mtex_versions.py plot                   docs/research/mtex-versions.png
#
# The releases are their GitHub zips with the mex files of their tag, unpacked under
# MTEX_VERSIONS (one directory per version holding mtex-<version>); develop is ../mtexDevelop.
# The times land in bench/out/versions/<version>/<module>_<op>.txt.
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "versions")
MODULES = ("geometry", "functions", "tensors", "maps", "io", "analysis", "embedding", "plotting", "maps3d")
RELEASES = ("5.11.2", "6.1.1", "7.0.0", "7.1.0")
BASE = "6.1.1"
VERSIONS = RELEASES + ("develop", "pymtex")
COLOURS = {"5.11.2": "#8c564b", "7.0.0": "#2ca02c", "7.1.0": "#1f77b4", "develop": "#9467bd", "pymtex": "#d62728"}


def mtexRoot(version):
  if version == "develop":
    return os.path.join(os.path.dirname(ROOT), "mtexDevelop")
  return os.path.join(os.environ["MTEX_VERSIONS"], version, f"mtex-{version}")


def ops(module, *cmd, env=None):
  r = subprocess.run([sys.executable, os.path.join(HERE, f"{module}_ops.py"), *cmd], env=env)
  return r.returncode == 0


# the times a run left in the module's directory, copied under the version
def collect(module, port, version):
  dst = os.path.join(OUT, version)
  os.makedirs(dst, exist_ok=True)
  for f in glob.glob(os.path.join(HERE, "out", module, f"{port}_*.txt")):
    shutil.copy(f, os.path.join(dst, f"{module}_{os.path.basename(f)[len(port) + 1:]}"))
    if port == "matlab":
      os.remove(f)


def python(modules):
  for m in modules:
    for f in glob.glob(os.path.join(HERE, "out", m, "python_*.txt")):
      os.remove(f)
    if "\"export\":" in open(os.path.join(HERE, f"{m}_ops.py")).read():
      ops(m, "export")
    ops(m, "run")
    collect(m, "python", "pymtex")


def matlab(version, modules):
  env = dict(os.environ, MTEX_BATCH_ROOT=mtexRoot(version))
  for m in modules:
    for f in glob.glob(os.path.join(HERE, "out", m, "matlab_*.txt")):
      os.remove(f)
    ops(m, "matlab", env=env)
    collect(m, "matlab", version)


def times():
  out = {}
  for v in VERSIONS:
    for f in glob.glob(os.path.join(OUT, v, "*.txt")):
      out.setdefault(os.path.basename(f)[:-4], {})[v] = float(open(f).read())
  return out


def table():
  t = times()
  lines = ["| operation | " + " | ".join(VERSIONS) + " |", "|---|" + "---:|" * len(VERSIONS)]
  for op in sorted(t):
    lines.append(f"| `{op}` | " + " | ".join(f"{1e3 * t[op][v]:.1f}" if v in t[op] else "-" for v in VERSIONS) + " |")
  return "\n".join(lines)


def plot(path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np
  t = times()
  fig, ax = plt.subplots(figsize=(7, 7))
  base = [op for op in t if BASE in t[op]]
  lo = min(min(t[op].values()) for op in base) / 2
  hi = max(max(t[op].values()) for op in base) * 2
  ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=0)
  for f in (10, 0.1):
    ax.plot([lo, hi], [f * lo, f * hi], color="0.85", lw=0.8, ls="--", zorder=0)
  for v in VERSIONS:
    if v == BASE:
      continue
    pts = np.array([(t[op][BASE], t[op][v]) for op in base if v in t[op]])
    if len(pts):
      ax.scatter(pts[:, 0], pts[:, 1], s=22, color=COLOURS[v], label=f"{v} ({len(pts)})", alpha=0.85, edgecolors="none")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim(lo, hi)
  ax.set_ylim(lo, hi)
  ax.set_aspect("equal")
  ax.set_xlabel(f"time in MTEX {BASE} [s]")
  ax.set_ylabel("time in the other version [s]")
  ax.set_title(f"the crosscheck operations against MTEX {BASE}: below the diagonal is faster")
  ax.legend(loc="upper left")
  fig.tight_layout()
  fig.savefig(path, dpi=150)


if __name__ == "__main__":
  cmd = sys.argv[1]
  if cmd == "python":
    python(sys.argv[2:] or MODULES)
  elif cmd == "matlab":
    matlab(sys.argv[2], sys.argv[3:] or MODULES)
  elif cmd == "table":
    print(table())
  elif cmd == "plot":
    plot(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "research", "mtex-versions.png"))

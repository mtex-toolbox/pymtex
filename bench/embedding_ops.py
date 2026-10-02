# the isometric embedding of the orientation space against MATLAB: the images of many
# orientations, the projection of points off the manifold back onto it, and the mean
#
#   .venv/bin/python bench/embedding_ops.py export              the input
#   .venv/bin/python bench/embedding_ops.py run [op ...]        this port
#   .venv/bin/python bench/embedding_ops.py matlab [op ...]     bench/matlab/bench_embedding_ops.m through the bridge
#   .venv/bin/python bench/embedding_ops.py julia [op ...]      bench/embedding_ops.jl, the Julia port
#   .venv/bin/python bench/embedding_ops.py compare
#
# The input in bench/out/embedding: 1e5 random orientations O, the same turned by 5 degree about
# random axes P, and 1e5 orientations C spread by 15 degree about one centre. The ports choose
# their own orthonormal basis of the embedding, so nothing compares coordinates: the images by
# their distance to the image of O(1) over the norm of the identity's image, the projections of
# the midpoints (e(O) + e(P)) / 2 and the projected mean of e(C) by the misorientation angle,
# each in m-3m and 6/mmm.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "embedding")
SYM = {"m3m": "m-3m", "6mmm": "6/mmm"}
ALGORITHM = [
  {"name": "orientation embedding",
   "entry": {"matlab": ["@embedding.embedding", "@embedding.project", "@embedding.mean"],
             "python": ["mtex.analysis.embedding.embedding.__init__", "mtex.analysis.embedding.embedding.project",
                        "mtex.analysis.embedding.embedding.mean"],
             "julia": ["embedding", "project", "mean"]},
   "cases": {"embed_m3m": "embedding(ori), 1e5 orientations, m-3m: the distance of every image to the first",
             "project_m3m": "orientation((e(O) + e(P)) / 2), 1e4 midpoints, m-3m",
             "mean_m3m": "orientation(mean(embedding(C))), 1e5 orientations about one centre, m-3m",
             "embed_6mmm": "embedding(ori), 1e5 orientations, 6/mmm: the distance of every image to the first",
             "project_6mmm": "orientation((e(O) + e(P)) / 2), 1e4 midpoints, 6/mmm",
             "mean_6mmm": "orientation(mean(embedding(C))), 1e5 orientations about one centre, 6/mmm"},
   # the projection is an ascent stopped at a gradient of 1e-10, MATLAB's at its own tolerance
   "tolerance": {"embed_m3m": 1e-12, "project_m3m": 1e-6, "mean_m3m": 1e-6,
                 "embed_6mmm": 1e-12, "project_6mmm": 1e-6, "mean_6mmm": 1e-6}},
]
N, NP = 10**5, 10**4
COLS = {f"{op}_{s}": k for s in SYM for op, k in (("embed", 1), ("project", 4), ("mean", 4))}


def export():
  from mtex import degree, rotation, vector3d
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  unit = lambda n, k: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rng.standard_normal((n, k)))
  O = rotation(unit(N, 4))
  P = O * rotation.byAxisAngle(vector3d(unit(N, 3)), 5 * degree)
  C = rotation.byEuler(30 * degree, 40 * degree, 50 * degree) * vector3d(rng.standard_normal((N, 3)) * 15 * degree).exp()
  for name, r in (("O", O), ("P", P), ("C", C)):
    np.asarray(r.data, dtype="<f8").reshape(-1, 4).tofile(os.path.join(OUT, f"{name}.bin"))


def load(name):
  return np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, 4)


def run(ops):
  from mtex import crystalFrame, embedding, orientation, rotation
  quat = lambda o: np.asarray(o.data, float).reshape(-1, 4)
  table = {}
  for s, sym in SYM.items():
    cs = crystalFrame(sym)
    O, P, C = (orientation(rotation(load(x)), cs) for x in "OPC")

    def embed(O=O, CS=cs):
      e = embedding(O)
      return (e - embedding(O[0])).norm() / embedding(orientation.id(CS)).norm()

    table |= {
      f"embed_{s}": (embed, lambda d: np.asarray(d, float).reshape(-1, 1)),
      f"project_{s}": (lambda O=O, P=P: ((embedding(O[:NP]) + embedding(P[:NP])) / 2).orientation(), quat),
      f"mean_{s}": (lambda C=C: embedding(C).mean().orientation(), quat),
    }
  crosscheck.run(OUT, ALGORITHM, table, ops, reps={op: 1 if op.startswith("project") else 3 for op in COLS})


# distances relative to the largest; orientations by their misorientation angle in the case's group
def deviation(op, a, b):
  if a.shape != b.shape:
    return float("nan")
  if a.shape[1] == 4:
    from mtex import angle, crystalFrame, orientation, rotation
    cs = crystalFrame(SYM[op.split("_")[1]])
    return float(np.max(np.asarray(angle(orientation(rotation(a), cs), orientation(rotation(b), cs)))))
  return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: crosscheck.matlab("bench_embedding_ops", repr(OUT), crosscheck.cellOf(names), str(N)),
   "julia": lambda: crosscheck.julia("embedding_ops.jl", ALGORITHM, names),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()

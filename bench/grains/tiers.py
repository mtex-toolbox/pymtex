# the instances of the tiers: a specification each, its seed the CRC of its name, so that an
# instance stays the same when others are added
#
#   T0  12 instances of 64 x 64 pixels, the cross-port check
#   T1  69 instances of 128^2 or 256^2 pixels, one factor at a time around a clean and a noisy
#       annealed scenario, two seeds each, and five combined scenarios
from __future__ import annotations

import copy
import zlib


def spec(name, generator="G1", kind="square", n=64, lowAngle=0.0, twin=0.0, **kw):
  st = {k: kw.pop(k) for k in ("D", "rmin", "rmax", "aspect", "tmin", "tmax") if k in kw}
  if generator in ("G3", "G5L"):
    st.pop("D", None)
  ori = dict(lowAngle=lowAngle, twin=twin) | {k: kw.pop(k) for k in ("gradient", "correlated") if k in kw}
  return dict(name=name, seed=zlib.crc32(name.encode()), structure=dict(generator=generator, **st), grid=dict(kind=kind, nx=n, ny=n),
              phases=kw.pop("phases", ["Iron"]), orientation=ori, measure=kw)


SPOT = dict(spot=0.5, minShare=0.7)

T0 = [spec("T0_01_voronoi", D=10), spec("T0_02_hex", kind="hex", D=10), spec("T0_03_small", D=5),
      spec("T0_04_spot", D=10, **SPOT), spec("T0_05_noise", D=10, noise=1.5, wild=0.01),
      spec("T0_06_notIndexed", D=10, notIndexed=0.05), spec("T0_07_clusters", D=10, clusters=0.05),
      spec("T0_08_lowAngle", D=10, lowAngle=0.3, twin=0.1, noise=0.5),
      spec("T0_09_laguerre", generator="G2", D=10), spec("T0_10_laguerreHex", generator="G2", kind="hex", D=10, noise=1.5, **SPOT),
      spec("T0_11_discs", generator="G3", rmin=2, rmax=12), spec("T0_12_discsSpot", generator="G3", rmin=2, rmax=12, noise=0.5, **SPOT)]


def _t1():
  base = dict(clean=dict(D=20), noisy=dict(D=20, noise=1.5, wild=0.01, notIndexed=0.02, **SPOT))
  # every factor replaces or adds keywords of the base
  factors = dict(base={}, G2=dict(generator="G2"), G3=dict(generator="G3", rmin=2, rmax=30), hex=dict(kind="hex"),
                 D5=dict(D=5), D10=dict(D=10), D50=dict(D=50), sampling=None, sigma05=dict(noise=0.5), sigmaX=None,
                 wild=dict(wild=0.02), notIndexed=dict(notIndexed=0.05), clusters=dict(clusters=0.05),
                 lowAngle=dict(lowAngle=0.3, twin=0.1), gradient=dict(gradient=(1, 5)), stretched=dict(generator="G5", aspect=4))
  out = []
  for b, kw in base.items():
    for f, change in factors.items():
      k = copy.deepcopy(kw)
      if f == "sampling":
        k = k | SPOT if b == "clean" else {x: v for x, v in k.items() if x not in SPOT}
      elif f == "sigmaX":
        k = k | dict(noise=1.5) if b == "clean" else {x: v for x, v in k.items() if x != "sigma"}
      else:
        k |= change
      n = 128 if k.get("D", 20) <= 10 else 256
      for s in (1, 2):
        out.append(spec(f"T1_{b}_{f}_s{s}", n=n, **k))
  combined = dict(laguerreHexNoisy=dict(generator="G2", kind="hex", D=10, noise=1.5, wild=0.01, **SPOT),
                  discsNoisy=dict(generator="G3", rmin=2, rmax=30, noise=1.5, notIndexed=0.02, **SPOT),
                  hexLowAngle=dict(kind="hex", D=10, noise=0.5, lowAngle=0.3, twin=0.1, **SPOT),
                  laguerreSmall=dict(generator="G2", D=5, noise=1.5, wild=0.01, notIndexed=0.02, **SPOT),
                  lamellae=dict(generator="G5L", tmin=1, tmax=10, twin=1.0, noise=0.5, **SPOT))
  for c, k in combined.items():
    out.append(spec(f"T1_{c}", n=128 if k.get("D", 20) <= 10 else 256, **k))
  return out


T1 = _t1()
TIERS = dict(T0=T0, T1=T1)

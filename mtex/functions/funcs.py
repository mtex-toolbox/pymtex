# the verbs of the function layer as functions: conv(f, g) and f.conv(g) are one call; each
# dispatches to the method of its first argument, kernel or harmonic function alike
from __future__ import annotations

import numpy as np

from ..geometry.funcs import _method
from .characteristics import max_, min_
from .so3vectorfield import antiderivative, component, curl, div
from .density import calcDensity, interpolate
from .funops import mapFun

conv = _method("conv")
radon = _method("radon")
grad = _method("grad")
laplace = _method("laplace")
invRadon = _method("invRadon")
halfwidth = _method("halfwidth")
calcFourier = _method("calcFourier")
truncate = _method("truncate")
withExchange = _method("withExchange")
even = _method("even")
sqrt = _method("sqrt", np.sqrt)
diff = _method("diff")
max = max_
min = min_

__all__ = ["conv", "radon", "grad", "laplace", "invRadon", "mapFun", "halfwidth", "calcFourier",
           "truncate", "withExchange", "even", "sqrt", "diff", "calcDensity", "interpolate", "max", "min", "curl", "div", "antiderivative", "component"]

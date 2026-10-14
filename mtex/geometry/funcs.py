# the verbs as functions: angle(o1, o2) and o1.angle(o2) are one call, and a plain array
# falls through to NumPy
from __future__ import annotations

import builtins

import numpy as np

from ._array import MtexArray, cat
from .arguments import refuseFlags


# the purpose and calling form of the public functions that forward to an object method
_METHOD_HELP = {
  'Euler': ('express rotations as Euler angles in radians', 'alpha, beta, gamma = Euler(rot)'),
  'KAM': ('measure the mean misorientation angle from each map measurement to its neighbours', 'omega = KAM(ebsd)'),
  'Laue': ('add inversion symmetry to a point group or reference frame', 'laue = Laue(cs)'),
  'LaueName': ('return the Hermann-Mauguin symbol of the corresponding Laue group', 'name = LaueName(cs)'),
  'Rodrigues': ('express rotations as axes scaled by the tangent of half the rotation angle', 'r = Rodrigues(rot)'),
  'alignment': ('describe how the Cartesian axes align with the axes of a crystal lattice', 'axes = alignment(cs)'),
  'angle': ('measure angles between geometric objects, or the phase of numeric values', 'omega = angle(v, w)'),
  'aspectRatio': ('measure the ratio of the longest to shortest fitted half axis of each grain', 'ratio = aspectRatio(grains)'),
  'axis': ('return the axis of a rotation or misorientation', 'a = axis(rot)'),
  'calcDistortion': ('fit the spatial distortion between the maps of a registration job', 'calcDistortion(job)'),
  'calcFourier': ('return the harmonic coefficients of a kernel', 'coefficients = calcFourier(psi)'),
  'calcGND': ('estimate geometrically necessary dislocation content from orientation gradients', 'gnd, rho = calcGND(ebsd, dS)'),
  'calcGROD': ('compare every measurement with the reference orientation of its grain', 'mis2mean = calcGROD(ebsd, grains)'),
  'calcMisorientation': ('sample uncorrelated misorientations from separated pairs of map measurements', 'mori = calcMisorientation(ebsd)'),
  'caliper': ('measure the Feret width of each grain outline along a direction or at its extrema', 'width = caliper(grains, direction)'),
  'characteristicShape': ('combine grain boundary segments into a characteristic particle shape', 'outline = characteristicShape(grains)'),
  'conj': ('take the quaternion or complex conjugate of an object', 'result = conj(x)'),
  'conv': ('convolve functions or kernels on their domain', 'smoothed = conv(f, psi)'),
  'conventionalSector': ('return the conventional spherical sector used to display crystal directions', 'sector = conventionalSector(cs)'),
  'convexity': ('compare each grain area with the area of its convex hull', 'ratio = convexity(grains)'),
  'cross': ('form the cross product of vectors or vector fields', 'normal = cross(v, w)'),
  'cubochoric': ('express rotations in equal-volume cubochoric coordinates', 'c = cubochoric(rot)'),
  'curvature': ('calculate the lattice curvature tensor from spatial orientation derivatives', 'kappa = curvature(ebsd)'),
  'det': ('calculate tensor determinants or vector triple products', 'value = det(T)'),
  'diff': ('differentiate a function on the circle', 'df = diff(f)'),
  'discretize': ('sample a spatial transform as a displacement field at given positions', 'field = discretize(T, pos)'),
  'disjoint': ('find the symmetry elements shared by two point groups or frames', 'common = disjoint(cs1, cs2)'),
  'displacement': ('evaluate the displacement produced by a spatial transform', 'u = displacement(T, pos)'),
  'dot': ('calculate an inner product, with the symmetry conventions of its operands', 'value = dot(v, w)'),
  'dspacing': ('return the separation of lattice planes from their Miller normals', 'd = dspacing(h)'),
  'even': ('extract the part of a function unchanged by reversal of its argument', 'fEven = even(f)'),
  'exp': ('take the pointwise exponential or the exponential map of an object', 'result = exp(x)'),
  'extent': ('return the spatial bounding box of a map or image', 'bounds = extent(ebsd)'),
  'factor': ('factor two point groups relative to their common symmetry elements', 'left, common, right = factor(cs1, cs2)'),
  'fill': ('fill missing measurements in a map using nearby data or grain information', 'filled = fill(ebsd)'),
  'fillByGrainId': ('replace map measurements by the mean orientations assigned to their grain ids', 'filled = fillByGrainId(ebsd, grains)'),
  'find': ('search geometric data for neighbours, or find nonzero entries of numeric data', 'indices, distances = find(grid, query)'),
  'findByLocation': ('locate map measurements or grains at specified spatial positions', 'measurements = findByLocation(ebsd, pos)'),
  'findByOrientation': ('select measurements or grains close to an orientation or fibre', 'selection = findByOrientation(ebsd, ori, epsilon)'),
  'fitEllipse': ('fit an ellipse to each grain outline using its second area moments', 'center, long, short = fitEllipse(grains)'),
  'fitStage': ('fit one stage of a composite spatial transform to supplied point pairs', 'fitted = fitStage(T, stage, source, target)'),
  'flip': ('exchange the two grain sides of boundary segments or faces', 'flipped = flip(boundary)'),
  'fullSym': ('return the largest registered symmetry group for the same frame basis', 'full = fullSym(cs)'),
  'fundamentalRegion': ('return the region representing distinct orientations under the given symmetries', 'region = fundamentalRegion(cs, ss)'),
  'fundamentalSector': ('return a spherical region containing one representative of each symmetry orbit', 'sector = fundamentalSector(cs)'),
  'grad': ('differentiate a scalar function or a smoothing kernel', 'gradient = grad(f)'),
  'gradient': ('calculate spatial orientation derivatives of a map', 'derivatives = gradient(ebsd)'),
  'gradientX': ('differentiate map orientations along the x direction', 'dx = gradientX(ebsd)'),
  'gradientY': ('differentiate map orientations along the y direction', 'dy = gradientY(ebsd)'),
  'gradientZ': ('differentiate map orientations along the z direction', 'dz = gradientZ(ebsd)'),
  'grainMean': ('average a map property separately within each grain', 'values = grainMean(ebsd, propertyValues)'),
  'gridify': ('arrange measurements on the cells of their spatial lattice', 'grid = gridify(ebsd)'),
  'halfwidth': ('return the angular halfwidth of a kernel in radians', 'h = halfwidth(psi)'),
  'homochoric': ('express rotations as equal-volume coordinates inside the homochoric ball', 'h = homochoric(rot)'),
  'hull': ('replace particle outlines by their convex hulls', 'convex = hull(grains)'),
  'inpolygon': ('test whether map positions or grain centroids lie inside a polygon', 'inside = inpolygon(ebsd, polygon)'),
  'interp': ('read a map or spherical data at new sample positions', 'interpolated = interp(ebsd, pos)'),
  'inv': ('invert a rotation, tensor or spatial transform', 'inverse = inv(x)'),
  'invRadon': ('invert the spherical Funk transform on its even modes', 'f = invRadon(transformed)'),
  'isAligned': ('test whether two reference frames have aligned basis vectors', 'aligned = isAligned(frame1, frame2)'),
  'isCompatible': ('check whether a coordinate transition between two frames is available', 'compatible, transition = isCompatible(frame1, frame2)'),
  'isclose': ('compare objects within the numerical tolerances of their type', 'close = isclose(a, b)'),
  'isnan': ('identify objects containing missing coordinates or coefficients', 'missing = isnan(x)'),
  'laplace': ('apply the Laplace-Beltrami operator to a harmonic function', 'laplacian = laplace(f)'),
  'log': ('take a pointwise logarithm or map rotations to tangent vectors', 'result = log(x)'),
  'longAxis': ('return the longest fitted half axis of each grain outline', 'a = longAxis(grains)'),
  'matrix': ('express rotations, transforms or tensors as numeric matrices', 'M = matrix(x)'),
  'mean': ('average objects using the mean defined for their type', 'average = mean(x)'),
  'merge': ('join grains that meet the specified boundary, label or orientation criteria', 'merged = merge(grains, pairs)'),
  'multiplicity': ('count distinct symmetry-equivalent directions or fibres', 'n = multiplicity(h)'),
  'nfold': ('return the order of the rotational symmetry axes of a point group', 'order = nfold(cs)'),
  'norm': ('measure vector lengths, tensor magnitudes or function L2 norms', 'magnitude = norm(x)'),
  'normalize': ('scale vectors, tensors or densities to the normalisation of their type', 'normalised = normalize(x)'),
  'numProper': ('count the proper rotations in a point group', 'n = numProper(cs)'),
  'numSym': ('count all operations in a point group', 'n = numSym(cs)'),
  'orth': ('choose a unit direction perpendicular to each vector', 'normal = orth(v)'),
  'orthProj': ('project vectors onto the plane perpendicular to a given normal', 'inPlane = orthProj(v, normal)'),
  'paris': ('measure outline indentation relative to the convex hull, in percent', 'indentation = paris(grains)'),
  'paror': ('calculate the cumulative particle projection function of grain outlines', 'projection = paror(grains)'),
  'perp': ('fit an axis perpendicular to a collection of directions', 'normal = perp(v)'),
  'pixelSizeMatch': ('resample the maps and images of a registration job to a common pixel spacing', 'pixelSizeMatch(job)'),
  'polar': ('return polar vector coordinates or a tensor polar decomposition', 'theta, rho, radius = polar(v)'),
  'pos2ind': ('convert physical image positions to pixel indices', 'indices = pos2ind(image, pos)'),
  'principalComponents': ('return the principal half axes of grain outlines or volumes', 'axes = principalComponents(grains)'),
  'project': ('project directions to a plane or embeddings back to orientations', 'projected = project(plane, positions)'),
  'projectIntoFundamentalRegion': ('choose symmetry-equivalent representatives in a fundamental region', 'representatives = projectIntoFundamentalRegion(ori)'),
  'properGroup': ('form a proper rotation group from all rotational parts of a point group', 'proper = properGroup(cs)'),
  'properSubGroup': ('retain only the proper rotations already present in a point group', 'proper = properSubGroup(cs)'),
  'radon': ('apply an orientation Radon transform or a spherical Funk transform', 'poleDensity = radon(odf, h)'),
  'reduce': ('coarsen a map lattice or a boundary discretisation', 'coarse = reduce(ebsd, 2)'),
  'reduceBoundary': ('coarsen the boundary discretisation of grains', 'coarse = reduceBoundary(grains, 2)'),
  'refineBoundary': ('subdivide boundary segments or triangular faces', 'fine = refineBoundary(grains)'),
  'setOptions': ('set registration options for a job or one of its maps', 'setOptions(job, **options)'),
  'shapeFactor': ('compare grain perimeters with circles of the same area', 'factor = shapeFactor(grains)'),
  'shift': ('translate map positions by a spatial displacement', 'shifted = shift(ebsd, displacementVector)'),
  'shortAxis': ('return the shortest fitted half axis of each grain outline', 'a = shortAxis(grains)'),
  'sim': ('compare the symmetry classes of reference frames without requiring aligned axes', 'sameSymmetry = sim(frame1, frame2)'),
  'simplifyBoundary': ('remove boundary vertices while limiting the change in outline shape', 'simplified = simplifyBoundary(grains, epsilon)'),
  'smooth': ('smooth sampled orientations, boundaries or harmonic functions', 'smoothed = smooth(x)'),
  'smoothBoundary': ('smooth grain boundary curves or surfaces', 'smoothed = smoothBoundary(grains)'),
  'spatialProfile': ('sample map measurements along a spatial polyline', 'profile, distance = spatialProfile(ebsd, points)'),
  'sqrt': ('take the pointwise square root of a function or numeric array', 'root = sqrt(f)'),
  'std': ('measure the angular spread of rotations about their mean', 'spread = std(ori)'),
  'stripSym': ('return the same frame basis with the trivial symmetry group', 'plain = stripSym(cs)'),
  'subGrid': ('select grid points by indices or by a geometric neighbourhood', 'selected = subGrid(grid, mask)'),
  'subsample': ('keep every n-th map measurement along each lattice direction', 'coarse = subsample(ebsd, 2)'),
  'surfor': ('calculate the cumulative directional projection length of boundary segments', 'projection = surfor(boundary)'),
  'symmetrise': ('apply the symmetry action appropriate to directions, tensors or fields', 'equivalents = symmetrise(h)'),
  'transform': ('apply a spatial transform to map positions', 'transformed = transform(ebsd, T)'),
  'transformReferenceFrame': ('express the same geometric data in another reference frame', 'transformed = transformReferenceFrame(x, frame)'),
  'transformationMatrix': ('return the coordinate transition between two reference frames', 'M = transformationMatrix(frame1, frame2)'),
  'truncate': ('cut or pad a harmonic expansion to a specified maximum degree', 'truncated = truncate(f, bandwidth)'),
  'undistort': ('resample a registration job onto its reference grid using the fitted transforms', 'undistort(job)'),
  'union': ('combine point groups or compatible pole figure data sets', 'combined = union(a, b)'),
  'unique': ('remove duplicate geometric objects, respecting their symmetry and tolerance', 'distinct = unique(x)'),
  'weightedBurgersVec': ('estimate net Burgers vector per unit area from an orientation map', 'b = weightedBurgersVec(ebsd)'),
  'withExchange': ('impose grain exchange symmetry f(g) = f(inv(g)) on an orientation function', 'symmetric = withExchange(mdf)'),
}


def _method(name, fallback=None):
  def f(x, *args, **kwargs):
    if isinstance(x, MtexArray) or hasattr(x, name) and not isinstance(x, np.ndarray):
      return getattr(x, name)(*args, **kwargs)
    if fallback is None:
      raise TypeError(f"{name} is not defined for {type(x).__name__}")
    return fallback(x, *args, **kwargs)
  f.__name__ = name
  summary, form = _METHOD_HELP.get(name, (f"apply the {name} method of an object", f"result = {name}(x, *args, **kwargs)"))
  f.__doc__ = (f"{summary}.\n\n  {form}\n\n"
               f"For an MTEX object, {name}(x, ...) calls x.{name}(...). Additional positional\n"
               "and keyword arguments are passed to that method unchanged. Its documentation\n"
               "specifies the accepted inputs, units, symmetry handling and optional outputs.\n"
               "Consult the corresponding method for the behaviour of a particular object type.")
  if fallback is not None:
    module = "numpy.linalg" if name in ("norm", "inv", "det") else "numpy"
    f.__doc__ += f"\n\nFor numeric arrays without an MTEX method, this calls {module}.{fallback.__name__}."
  f.forwardsTo = name
  return f


angle = _method("angle", np.angle)
dot = _method("dot", np.dot)
cross = _method("cross", np.cross)
norm = _method("norm", np.linalg.norm)
normalize = _method("normalize")
inv = _method("inv", np.linalg.inv)
mean = _method("mean", np.mean)
std = _method("std", np.std)
unique = _method("unique", np.unique)
discretize = _method("discretize")
displacement = _method("displacement")
fitStage = _method("fitStage")
project = _method("project")
matrix = _method("matrix")
Euler = _method("Euler")
Rodrigues = _method("Rodrigues")
axis = _method("axis")
log = _method("log", np.log)
homochoric = _method("homochoric")
cubochoric = _method("cubochoric")
orth = _method("orth")
perp = _method("perp")
polar = _method("polar")
orthProj = _method("orthProj")
det = _method("det", np.linalg.det)
conj = _method("conj", np.conj)
isnan = _method("isnan", np.isnan)
Laue = _method("Laue")
properGroup = _method("properGroup")
properSubGroup = _method("properSubGroup")
disjoint = _method("disjoint")
union = _method("union", np.union1d)
factor = _method("factor")
nfold = _method("nfold")
numSym = _method("numSym")
numProper = _method("numProper")
LaueName = _method("LaueName")
sim = _method("sim")
isAligned = _method("isAligned")
isCompatible = _method("isCompatible")
transformationMatrix = _method("transformationMatrix")
alignment = _method("alignment")
stripSym = _method("stripSym")
fullSym = _method("fullSym")
dspacing = _method("dspacing")
symmetrise = _method("symmetrise")
multiplicity = _method("multiplicity")
projectIntoFundamentalRegion = _method("projectIntoFundamentalRegion")
transformReferenceFrame = _method("transformReferenceFrame")
fundamentalSector = _method("fundamentalSector")
conventionalSector = _method("conventionalSector")
fundamentalRegion = _method("fundamentalRegion")


def round(x, *args, **kwargs):
  """round(x) of an MTEX object by its own round, of an array, a list or with an axis as NumPy
  rounds, of a number as Python rounds, so that from mtex import * keeps Python's round"""
  if isinstance(x, MtexArray) or hasattr(x, "round") and not isinstance(x, (np.ndarray, np.generic)):
    return x.round(*args, **kwargs)
  if isinstance(x, (np.ndarray, list, tuple)) or kwargs:
    return np.round(x, *args, **kwargs)
  return builtins.round(x, *args)


def sum(x, *args, **kwargs):
  """sum(x) of an MTEX object by its own sum, of an array or with an axis as NumPy sums, of any
  other iterable as Python sums, so that from mtex import * keeps Python's sum"""
  if isinstance(x, MtexArray) or hasattr(x, "sum") and not isinstance(x, (np.ndarray, np.generic)):
    return x.sum(*args, **kwargs)
  if isinstance(x, (np.ndarray, np.generic)) or kwargs:
    return np.sum(x, *args, **kwargs)
  return builtins.sum(x, *args)


def checkInside(region, x, **kw):
  """whether x lies in the region, region.checkInside(x)"""
  return region.checkInside(x, **kw)
find = _method("find", np.flatnonzero)
subGrid = _method("subGrid")


exp = _method("exp", np.exp)
isclose = _method("isclose", np.isclose)
merge = _method("merge")


def rotate(v, rot, *args, **kwargs):
  """rotate(x, rot) is x.rotate(rot), or rot * x for an object without that method; 'left' and
  'right' name the side, a flag is a keyword

    rotate(ebsd, rot, keepEuler=True)
    rotate(odf, rot, 'right')"""
  refuseFlags(args, "rotate", names=("left", "right"))
  side = next((a for a in args if isinstance(a, str)), None)
  args = tuple(a for a in args if not isinstance(a, str))
  if side is not None:
    kwargs["side"] = side
  return v.rotate(rot, *args, **kwargs) if hasattr(v, "rotate") else rot * v


__all__ = ["angle", "dot", "cross", "orthProj", "norm", "normalize", "inv", "mean", "sum", "std", "unique", "matrix", "Euler",
           "Rodrigues", "axis", "log", "exp", "isclose", "homochoric", "cubochoric", "orth", "perp", "polar",
           "det", "conj", "isnan", "rotate", "cat", "merge", "Laue", "properGroup", "properSubGroup", "disjoint", "union", "factor", "nfold", "numSym", "numProper", "LaueName", "sim", "isAligned", "isCompatible", "transformationMatrix", "alignment", "stripSym", "fullSym", "dspacing", "round", "symmetrise", "multiplicity", "projectIntoFundamentalRegion", "transformReferenceFrame", "fundamentalSector", "conventionalSector", "fundamentalRegion", "checkInside", "find", "subGrid", "discretize", "displacement", "fitStage", "project"]

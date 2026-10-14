"""the geometry layer: vectors, rotations, orientations, symmetry, frames, grids"""
import importlib as _importlib
from .units import degree, mtexdegchar
from .session import getMTEXpref, setMTEXpref, reset
from .vector3d import vector3d, isCrystalDirection
from .quaternion import quaternion, rotation, orthogonalTransform, reflection
from .homochoric import homochoricRadius, cubo2homo, homo2cubo
from .plottingconvention import AXIS_CONVENTIONS, conventionNames, closestConvention, plottingConvention
from .gridlayout import gridLayout, layoutIndex
from .latticetype import latticeType
from .symmetry import (symmetry, findPointGroup, isPointGroupName, hm2point, monoclinicSetting, spaceGroupToPointGroup,
                       identifyGroup)
from .crystalaxes import crystalAxes, reciprocalAxes
from .frames import referenceFrame, crystalFrame, specimenFrame, notIndexedFrame, alignmentSetups, closestAlignment
from .miller import Miller
from .orientation import orientation
from .products import *  # noqa: F401,F403
from .symreduce import *  # noqa: F401,F403
from .symmetryops import *  # noqa: F401,F403
from .sphericalregion import sphericalRegion, dirichletSector, tilesSphere, sectorCenter, projectIntoRegion
from .s2triangulation import S2Triangulation
from .orientationregion import orientationRegion, eulerRegion, fundamentalRegionEuler, projectIntoEulerRegion, maxAngle
from .frametransform import align, byScreenAlignment
from .standardorientations import variants, STANDARD_EULER
from .csl import CSL
from .tangent import SO3TangentVector, spinTensor, left, right, geodesic
from .fibre import fibre
from .hull import convexHull, convexHullLoops
from .slipsystem import slipSystem, dislocationSystem, mPrime
from .crystalshape import crystalShape, innerFace, CRYSTAL_SHAPES
from .plane import plane3d, dist, intersect, lineIntersect
from .parent2child import (calcParent2Child, misfit, calcVariantId, calcParent, round2Miller, allHKL, packetPlanes,
                           bainPlanes)
from .grids.s1grid import S1Grid
from .grids.s2grid import S2Grid, equispacedS2Grid, regularS2Grid
from .grids.fibonacci import fibonacciS2Grid
from .grids.healpix import HEALPixS2Grid
from .grids.so3grid import SO3Grid, equispacedSO3Grid, regularSO3Grid
from .grids.superfibonacci import SO3FibonacciGrid, superfibonacciSO3Grid, fibonacciSO3Grid
from .grids.localgrids import localOrientationGrid, localS2Grid
from .spatialtransform import (spatialTransformList, spatialTransform, spatialTransformId, spatialTransformShift,
                               spatialTransformRigid, spatialTransformProjective, spatialTransformPoly,
                               spatialTransformDrift, spatialTransformField, spatialTransformHandle,
                               spatialTransformComposite, spatialTransformTilt, spatialTransformInverse)
from .settings import settingsFile, loadSettings
from .funcs import (angle, dot, cross, orthProj, norm, normalize, inv, mean, sum, std, unique, matrix, Euler, Rodrigues,
                    axis, log, exp, isclose, homochoric, cubochoric, orth, perp, polar, det, conj, isnan, rotate, cat,
                    merge, Laue, properGroup, properSubGroup, disjoint, union, factor, nfold, numSym, numProper,
                    LaueName, sim, isAligned, isCompatible, transformationMatrix, alignment, stripSym, fullSym,
                    dspacing, round, symmetrise, multiplicity, projectIntoFundamentalRegion, transformReferenceFrame,
                    fundamentalSector, conventionalSector, fundamentalRegion, checkInside, find, subGrid, discretize,
                    displacement, fitStage, project)

__all__ = ["degree", "mtexdegchar", "getMTEXpref", "setMTEXpref", "reset", "vector3d", "isCrystalDirection",
           "quaternion", "rotation", "orthogonalTransform", "reflection", "homochoricRadius", "cubo2homo", "homo2cubo",
           "AXIS_CONVENTIONS", "conventionNames", "closestConvention", "plottingConvention", "gridLayout",
           "layoutIndex", "latticeType", "symmetry", "findPointGroup", "isPointGroupName", "hm2point",
           "monoclinicSetting", "spaceGroupToPointGroup", "identifyGroup", "crystalAxes", "reciprocalAxes",
           "referenceFrame", "crystalFrame", "specimenFrame", "notIndexedFrame", "alignmentSetups", "closestAlignment",
           "Miller", "orientation", "sphericalRegion", "fundamentalSector", "conventionalSector", "dirichletSector",
           "tilesSphere", "sectorCenter", "projectIntoRegion", "S2Triangulation", "orientationRegion",
           "fundamentalRegion", "eulerRegion", "fundamentalRegionEuler", "projectIntoEulerRegion", "maxAngle", "align",
           "byScreenAlignment", "variants", "STANDARD_EULER", "CSL", "SO3TangentVector", "spinTensor", "left", "right",
           "geodesic", "fibre", "convexHull", "convexHullLoops", "slipSystem", "dislocationSystem", "mPrime",
           "crystalShape", "innerFace", "CRYSTAL_SHAPES", "plane3d", "dist", "intersect", "lineIntersect",
           "calcParent2Child", "misfit", "calcVariantId", "calcParent", "round2Miller", "allHKL", "packetPlanes",
           "bainPlanes", "S1Grid", "S2Grid", "equispacedS2Grid", "regularS2Grid", "fibonacciS2Grid", "HEALPixS2Grid",
           "SO3Grid", "equispacedSO3Grid", "regularSO3Grid", "SO3FibonacciGrid", "superfibonacciSO3Grid",
           "fibonacciSO3Grid", "localOrientationGrid", "localS2Grid", "spatialTransformList", "spatialTransform",
           "spatialTransformId", "spatialTransformShift", "spatialTransformRigid", "spatialTransformProjective",
           "spatialTransformPoly", "spatialTransformDrift", "spatialTransformField", "spatialTransformHandle",
           "spatialTransformComposite", "spatialTransformTilt", "spatialTransformInverse", "settingsFile",
           "loadSettings", "angle", "dot", "cross", "orthProj", "norm", "normalize", "inv", "mean", "sum", "std",
           "unique", "matrix", "Euler", "Rodrigues", "axis", "log", "exp", "isclose", "homochoric", "cubochoric",
           "orth", "perp", "polar", "det", "conj", "isnan", "rotate", "cat", "merge", "Laue", "properGroup",
           "properSubGroup", "disjoint", "union", "factor", "nfold", "numSym", "numProper", "LaueName", "sim",
           "isAligned", "isCompatible", "transformationMatrix", "alignment", "stripSym", "fullSym", "dspacing", "round",
           "symmetrise", "multiplicity", "projectIntoFundamentalRegion", "transformReferenceFrame", "checkInside",
           "find", "subGrid", "discretize", "displacement", "fitStage", "project"]

from typing import TYPE_CHECKING

_lazy = {name: mod for name in ("xvector", "yvector", "zvector") for mod in [_importlib.import_module(f"{__name__}.vector3d")]}


# the vector constants are minted on every access, see vector3d.py
def __getattr__(name):
  if name in _lazy:
    return getattr(_lazy[name], name)
  raise AttributeError(name)


if TYPE_CHECKING:
  from .vector3d import xvector, yvector, zvector

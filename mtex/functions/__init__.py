"""the function layer: harmonics, kernels, transforms, functions on S1, S2 and SO(3), pole figures"""
from .harmonics import (deg2dim, dim2deg, degreeRange, degreeBlock, wignerd, wignerdDegrees, WignerD,
                        InvariantProjection, invariantProjection, project, projectLeft, projectRight,
                        normalizedLegendre, sphericalY, clebschGordan, wigner3j, clebschGordanTensor)
from .quadrature import (gaussLegendre, clenshawCurtis, legendreP, folds, bandwidthFor, quadratureSO3Grid,
                         quadratureS2Grid, quadratureS1Grid)
from .kernelfunctions import (SO3Kernel, SO3DeLaValleePoussinKernel, SO3AbelPoissonKernel, SO3vonMisesFisherKernel,
                              SO3GaussWeierstrassKernel, SO3SquareSingularityKernel, SO3BumpKernel, SO3DirichletKernel,
                              SO3RestrictedDistanceKernel, SO3LaplaceKernel, SO3SobolevKernel,
                              fibreVonMisesFisherKernel, SO3KernelHandle, S2Kernel, S2DeLaValleePoussinKernel,
                              S2DirichletKernel, S2BumpKernel, S2RestrictedDistanceKernel, S2KernelHandle, S1Kernel,
                              S1DeLaValleePoussinKernel, S1DirichletKernel, S1FejerKernel, S1BumpKernel, S1GaussKernel,
                              S1KernelHandle)
from .transforms import (halfPiDegrees, foldedDims, usableFolds, wignerTransform, wignerTransformAdjoint,
                         sphericalTransform, sphericalTransformAdjoint, isRealWigner, isRealSpherical, isRealFourier,
                         SO3FourierPlan, S2FourierPlan, S1FourierPlan, SO3GridPlan, S2GridPlan, adjointTransform,
                         wignerPlan, sphericalPlan, wignerSeries, wignerSeriesAdjoint, wignerSeriesAdjointGridded,
                         sphericalSeries, sphericalSeriesAdjoint, fourierSeries, fourierSeriesAdjoint)
from .so3fun import SO3FunHarmonic
from .s2fun import S2FunHarmonic
from .s2funtri import S2FunTri
from .s1fun import S1FunHarmonic, silvermanWidth
from .funops import transformODF
from .rbf import SO3FunRBF, SO3FunCBF, uniformODF, unimodalODF, fibreODF, FourierODF, supportCosine, SantaFe
from .rbfinterpolation import interpolateRBF, summationMatrix, mlsq, interpSO3
from .kde import Gaussian, griddedDensity, kde1d, kdeN
from .density import *  # noqa: F401,F403
from .handles import SO3FunHandle, SO3FunComposition, S2FunHandle, S2FunSum, S1FunHandle
from .bingham import mhyper, SO3FunBingham, BinghamODF, calcBinghamODF, S2FunBingham
from .so3vectorfield import (SO3VectorFieldHarmonic, SO3VectorFieldHandle, SO3VectorFieldRBF, componentFrames, mapField,
                             differentiate, equivariantProjection)
from .s2vectorfield import S2VectorFieldHarmonic, S2VectorFieldHandle, thetaVector, rhoVector, angularDerivatives
from .s2axisfield import S2AxisFieldHarmonic, S2AxisFieldHandle
from .characteristics import (textureIndex, entropy, volume, fibreVolume, calcMDF, calcAngleDistribution, calcMIndex,
                              calcAxisDistribution, calcComponents, discreteSample, calcKernel, centerSpecimen)
from .polefigure import (PoleFigure, calcPoleFigure, calcQuadratureWeights, calcNormalization, calcErrorPF, calcError,
                         correct, noisepf, isOutlier, refine, SchulzDefocusingKernel)
from .funcs import (conv, radon, grad, laplace, invRadon, mapFun, halfwidth, calcFourier, truncate, withExchange, even,
                    sqrt, diff, calcDensity, interpolate, max, min, curl, div, antiderivative, component)
from .so3funclass import SO3Fun
from .s2funclass import S2Fun
from .discrepancy import discrepancy, sphericalDiscrepancy, rotationalDiscrepancy

__all__ = ["deg2dim", "dim2deg", "degreeRange", "degreeBlock", "wignerd", "wignerdDegrees", "WignerD", "wignerSeries",
           "InvariantProjection", "invariantProjection", "project", "projectLeft", "projectRight", "normalizedLegendre",
           "sphericalY", "clebschGordan", "wigner3j", "clebschGordanTensor", "gaussLegendre", "clenshawCurtis",
           "legendreP", "folds", "bandwidthFor", "quadratureSO3Grid", "quadratureS2Grid", "quadratureS1Grid",
           "SO3Kernel", "SO3DeLaValleePoussinKernel", "SO3AbelPoissonKernel", "SO3vonMisesFisherKernel",
           "SO3GaussWeierstrassKernel", "SO3SquareSingularityKernel", "SO3BumpKernel", "SO3DirichletKernel",
           "SO3RestrictedDistanceKernel", "SO3LaplaceKernel", "SO3SobolevKernel", "fibreVonMisesFisherKernel",
           "SO3KernelHandle", "S2Kernel", "S2DeLaValleePoussinKernel", "S2DirichletKernel", "S2BumpKernel",
           "S2RestrictedDistanceKernel", "S2KernelHandle", "S1Kernel", "S1DeLaValleePoussinKernel", "S1DirichletKernel",
           "S1FejerKernel", "S1BumpKernel", "S1GaussKernel", "S1KernelHandle", "halfPiDegrees", "foldedDims",
           "usableFolds", "wignerTransform", "wignerTransformAdjoint", "sphericalTransform",
           "sphericalTransformAdjoint", "isRealWigner", "isRealSpherical", "isRealFourier", "SO3FourierPlan",
           "S2FourierPlan", "S1FourierPlan", "SO3GridPlan", "S2GridPlan", "adjointTransform", "wignerPlan",
           "sphericalPlan", "wignerSeriesAdjoint", "wignerSeriesAdjointGridded", "sphericalSeries",
           "sphericalSeriesAdjoint", "fourierSeries", "fourierSeriesAdjoint", "SO3FunHarmonic", "S2FunHarmonic",
           "S2FunTri", "S1FunHarmonic", "silvermanWidth", "mapFun", "transformODF", "SO3FunRBF", "SO3FunCBF",
           "uniformODF", "unimodalODF", "fibreODF", "FourierODF", "supportCosine", "SantaFe", "interpolateRBF",
           "summationMatrix", "mlsq", "interpSO3", "Gaussian", "griddedDensity", "kde1d", "kdeN", "calcDensity",
           "interpolate", "SO3FunHandle", "SO3FunComposition", "S2FunHandle", "S2FunSum", "S1FunHandle", "mhyper",
           "SO3FunBingham", "BinghamODF", "calcBinghamODF", "S2FunBingham", "SO3VectorFieldHarmonic",
           "SO3VectorFieldHandle", "SO3VectorFieldRBF", "componentFrames", "component", "curl", "div", "antiderivative",
           "mapField", "differentiate", "equivariantProjection", "S2VectorFieldHarmonic", "S2VectorFieldHandle",
           "thetaVector", "rhoVector", "angularDerivatives", "S2AxisFieldHarmonic", "S2AxisFieldHandle", "textureIndex",
           "entropy", "volume", "fibreVolume", "calcMDF", "calcAngleDistribution", "calcMIndex", "calcAxisDistribution",
           "calcError", "calcComponents", "discreteSample", "calcKernel", "centerSpecimen", "PoleFigure",
           "calcPoleFigure", "calcQuadratureWeights", "calcNormalization", "calcErrorPF", "correct", "noisepf",
           "isOutlier", "refine", "SchulzDefocusingKernel", "conv", "radon", "grad", "laplace", "invRadon", "halfwidth",
           "calcFourier", "truncate", "withExchange", "even", "sqrt", "diff", "max", "min", "SO3Fun", "S2Fun",
           "discrepancy", "sphericalDiscrepancy", "rotationalDiscrepancy"]

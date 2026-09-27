# the analysis layer: the algorithms that iterate or optimise over the cores, from the
# pole figure inversion to the denoising of a map
from .elasticity import (velocity, energyVector, YoungsModulus, linearCompressibility, volumeCompressibility,
                         shearModulus, PoissonRatio, bulkModulus, isotropicShearModulus, HashinShtrikmanModulus,
                         hashinShtrikmanBounds)
from .mechanics import (meanStress, deviatoric, deviatoricStress, deviatoricStrain, equivalentStress, equivalentStrain,
                        octahedralStrain, axisRatios, maxStrain, calcShearStress, SchmidFactor, CauchyStrain,
                        GreenStrain, strainRate, spin, angularVelocityVector, rotationRate, vorticity, spinTensorOf,
                        dislocationDensity)
from .optics import refractiveIndex, birefringence, opticalAxis, spectralTransmission
from .taylor import calcTaylor, linearProgram, TaylorModel
from .texture import calcTensor, calcLankford, calcKearnsFactor, SO3FunSBF, doEulerStep
from .pf2odf import MLSSolver, zeroRangeMethod, calcODF, calcODFIterative
from .embedding import embedding, embeddingSpec, double, setDouble
from .optimalsample import optimalSample, OptimalSampleResult
from .cluster import calcCluster, classix, knnRadius
from .variational import EBSDFilter, splineFilter, halfQuadraticFilter, l1TVFilter, infimalConvolutionFilter
from .dislocations import fitDislocationSystems
from .denoising import meanFilter, medianFilter, KuwaharaFilter
from .gbnd import calcGBNDKernel
from .gbnd3d import calcGBND

__all__ = ["velocity", "energyVector", "YoungsModulus", "linearCompressibility", "volumeCompressibility",
           "shearModulus", "PoissonRatio", "bulkModulus", "isotropicShearModulus", "HashinShtrikmanModulus",
           "hashinShtrikmanBounds", "meanStress", "deviatoric", "deviatoricStress", "deviatoricStrain",
           "equivalentStress", "equivalentStrain", "octahedralStrain", "axisRatios", "maxStrain", "calcShearStress",
           "SchmidFactor", "CauchyStrain", "GreenStrain", "strainRate", "spin", "angularVelocityVector", "rotationRate",
           "vorticity", "spinTensorOf", "dislocationDensity", "refractiveIndex", "birefringence", "opticalAxis",
           "spectralTransmission", "calcTaylor", "linearProgram", "TaylorModel", "calcTensor", "calcLankford",
           "calcKearnsFactor", "SO3FunSBF", "doEulerStep", "MLSSolver", "zeroRangeMethod", "calcODF",
           "calcODFIterative", "embedding", "embeddingSpec", "double", "setDouble", "optimalSample",
           "OptimalSampleResult", "calcCluster", "classix", "knnRadius", "EBSDFilter", "splineFilter",
           "halfQuadraticFilter", "l1TVFilter", "infimalConvolutionFilter", "fitDislocationSystems", "meanFilter",
           "medianFilter", "KuwaharaFilter", "calcGBND", "calcGBNDKernel"]

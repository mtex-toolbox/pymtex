# the strings the io layer's arguments take, as Literal types an editor completes in quotes;
# written by tools/staticapi.py from the tables they name, never by hand
from typing import Literal

# the data sets mtexdata reads
SampleDataName = Literal["aachen", "sharp", "small", "csl", "mylonite", "epidote", "forsterite", "olivine", "twins",
                         "copper", "single", "alu", "titanium", "ferrite", "dc06", "martensite", "emsland",
                         "alphaBetaTitanium", "EMSphinx", "testgrains", "dubna", "geesthacht", "steel", "ptx", "zncuti",
                         "aachenexp", "dubnaODF", "eclogite", "vectors", "dubnaValues", "smiley", "tongue", "vpsc",
                         "trueEbsdWCCoSmall", "xnovo", "SmallIN100", "odf", "quartzPiezo", "olivine1997",
                         "quartzPattern"]

# the data sets mtexdatafile fetches, those of the data repository included
SampleFileName = Literal["aachen", "sharp", "small", "csl", "mylonite", "epidote", "forsterite", "olivine", "twins",
                         "copper", "single", "alu", "titanium", "ferrite", "dc06", "martensite", "emsland",
                         "alphaBetaTitanium", "EMSphinx", "testgrains", "dubna", "geesthacht", "steel", "ptx", "zncuti",
                         "aachenexp", "dubnaODF", "eclogite", "vectors", "dubnaValues", "smiley", "tongue", "vpsc",
                         "trueEbsdWCCoSmall", "xnovo", "SmallIN100", "odf", "quartzPiezo", "olivine1997",
                         "quartzPattern", "3d", "copper29", "my100grains", "trueEbsdWCCo"]

# the maps among them
SampleEBSDName = Literal["aachen", "sharp", "small", "csl", "mylonite", "epidote", "forsterite", "olivine", "twins",
                         "copper", "single", "alu", "titanium", "ferrite", "dc06", "martensite", "emsland",
                         "alphaBetaTitanium", "EMSphinx", "eclogite", "trueEbsdWCCoSmall"]

# the grains among them
SampleGrainsName = Literal["testgrains"]

# the pole figures among them
SamplePoleFigureName = Literal["dubna", "geesthacht", "steel", "ptx", "zncuti", "aachenexp"]

# the maps of volumes among them
SampleEBSD3Name = Literal["xnovo", "SmallIN100"]

# the ODFs among them
SampleSO3FunName = Literal["dubnaODF", "odf"]

# the spherical functions among them
SampleS2FunName = Literal["quartzPattern"]

# the tensors among them
SampleTensorName = Literal["quartzPiezo", "olivine1997"]

# the documentation pages openDoc writes and the chapters it lists
DocPageName = Literal["BoundaryTutorial", "EBSDTutorial", "GrainTutorial", "ODFTutorial", "PoleFigureTutorial",
                      "changelog", "ClusterDemo", "Contribute2Doc", "DensityEstimation", "GeneralConceptsConfiguration",
                      "GeneralConceptsOptions", "Glossary", "Installation", "ListsAndIndexing", "MATLABtoPython",
                      "MTEXScripts", "NotationAndConventions", "OptimalKernel", "Properties", "PythonDevelopers",
                      "VectorDefinition", "VectorGrids", "VectorsAxes", "VectorsDensityEstimation", "VectorsExport",
                      "VectorsImport", "VectorsOperations", "RotationDefinition", "RotationFibre", "RotationImproper",
                      "RotationOperations", "RotationPlotting", "RotationRepresentations", "RotationSpinTensor",
                      "RotationTangentSpace", "CrystalDirections", "CrystalOperations", "CrystalReferenceSystem",
                      "CrystalShapes", "CrystalShapeSmorf", "CrystalSymmetries", "FundamentalSector", "LatticeMetric",
                      "QuasiCrystals", "SymmetryAlignment", "DefinitionAsCoordinateTransform", "MTEXvsBungeConvention",
                      "OrientationDefinition", "OrientationEmbeddings", "OrientationExport", "OrientationFibre",
                      "OrientationFundamentalRegion", "OrientationGrid", "OrientationImport",
                      "OrientationInversePoleFigure", "OrientationPoleFigure", "OrientationStandard",
                      "OrientationSymmetry", "OrientationVisualization3d", "OrientationVisualizationSections",
                      "SpecimenSymmetry", "AngleDistributionFunction", "AxisDistributionFunction",
                      "MisorientationDistributionFunction", "MisorientationGrainExchangeSym", "MisorientationTheory",
                      "Twinning", "BinghamODFs", "DetectionOfSampleSymmetry", "EulerAngleSections", "FibreODFs",
                      "ODFCharacteristics", "ODFComponents", "ODFExport", "ODFImport", "ODFInversePoleFigure",
                      "ODFModeling", "ODFPlot", "ODFPoleFigure", "ODFShapes", "ODFTheory", "RadialODFs",
                      "RandomSampling", "SigmaSections", "SO3FunHarmonicRepresentation", "PoleFigure2ODF",
                      "PoleFigure2ODFAmbiguity", "PoleFigure2ODFGhostCorrection", "PoleFigureCorrection",
                      "PoleFigureDubna", "PoleFigureExport", "PoleFigureImport", "PoleFigurePlot",
                      "PoleFigureRefinement", "PoleFigureSantaFe", "PoleFigureSimulation", "EBSD2ODF",
                      "EBSDAdvancedMaps", "EBSDDenoising", "EBSDExport", "EBSDFilling", "EBSDGrid", "EBSDGROD",
                      "EBSDImport", "EBSDIndex", "EBSDInter", "EBSDInterfaceHDF5", "EBSDIPFMap", "EBSDKAM",
                      "EBSDMapsAndImages", "EBSDOrientationAnalysis", "EBSDOrientationPlots", "EBSDPerceptualKey",
                      "EBSDPlotting", "EBSDProfile", "EBSDPseudoSymmetry", "EBSDReferenceFrame", "EBSDSelect",
                      "EBSDSharpPlot", "EBSDSimulation", "EBSDSpatialTransform", "EBSDTrueEbsd", "Dream3dGrains",
                      "EllipseBasedParameters", "Grain_dispersion_axes", "GrainExport", "GrainMerge", "GrainNeighbours",
                      "GrainOrientationParameters", "GrainReconstruction", "GrainReconstructionAdvanced",
                      "GrainReconstructionMCL", "GrainSpatialPlots", "HullBasedParameters", "ProjectionBasedParameters",
                      "SelectingGrains", "ShapeParameters", "BoundaryCurvature", "BoundaryIntersections",
                      "BoundaryMisorientations", "BoundaryNormalDistribution", "BoundaryPlots", "BoundaryProperties",
                      "BoundarySelect", "CSLBoundaries", "GrainSmoothing", "GrainSmoothingAdvanced", "QuadruplePoints",
                      "SubGrainBoundaries", "TiltAndTwistBoundaries", "TriplePoints", "TwinningBoundaries",
                      "EBSD3Analysis", "EBSD3Plotting", "Grains3D", "Grains3DBoundaries", "Grains3DOperations",
                      "Grains3DProperties", "Grains3DReconstruction", "Grains3DSmoothing", "NeperInterface",
                      "BirefringenceDemo", "MagneticAnisotropy", "PiezoElectricity", "TensorArithmetics",
                      "TensorAverage", "TensorDefinition", "TensorImport", "TensorVisualisation", "AnisotropicTheory",
                      "CPOSeismicProperties", "IsotropicTheory", "WaveVelocities", "DislocationSystems", "GND",
                      "Lankford", "SachsModel", "SchmidFactor", "SingleSlipModel", "SlipSystems", "SlipTransmission",
                      "TaylorHex", "TaylorModel", "TextureEvolution", "TwinningTutorial", "VPSCImport", "WBV",
                      "GrainGraphBasedReconstruction", "LowLevelParentGrainReconstruction",
                      "MaParentGrainReconstruction", "MaParentGrainReconstructionAdvanced", "MartensiteVariants",
                      "OrientationRelationshipFit", "ParentChildVariants", "PhaseTransitions", "TiBetaReconstruction",
                      "TransformationTexture", "TriplePointBasedReconstruction", "S1FunHarmonics", "S2Bingham",
                      "S2FunApproximationInterpolation", "S2FunAxisField", "S2FunConcept",
                      "S2FunHarmonicRepresentation", "S2FunOperations", "S2FunPlotting", "S2FunQuadrature",
                      "S2FunRadon", "S2FunSampling", "S2FunSym", "S2FunVectorField", "S2FunVectorValued", "S2Kernels",
                      "SphericalHarmonics", "HarmonicApproximationTheory", "RBFApproximationTheory",
                      "SO3FunApproximationTheory", "SO3FunConcept", "SO3FunConvolution", "SO3FunDefinition",
                      "SO3FunOperations", "SO3FunQuadrature", "SO3FunSymmetricFunctions", "SO3FunVectorField",
                      "SO3FunVectorValued", "SO3Kernels", "WignerFunctions", "Annotations", "AxesAlignment",
                      "ColorMaps", "CombinedPlots", "ContourPlots", "Legends", "Multiplot", "PlottingExport",
                      "PlotTypes", "SphericalProjections", "TransparencyDemo", "Tutorials", "GeneralConcepts",
                      "Vectors", "Rotations", "CrystalGeometry", "CrystalOrientations", "Misorientations",
                      "ODFAnalysis", "PoleFigureAnalysis", "EBSDAnalysis", "Grains", "GrainBoundaries", "Tensors",
                      "Elasticity", "Plasticity", "SphericalFunctions", "SO3Functions", "Plotting"]

__all__ = []

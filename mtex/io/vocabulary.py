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
                         "quartzPattern", "3d", "NeperGrain3d", "copper29", "my100grains", "trueEbsdWCCo"]

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

__all__ = []

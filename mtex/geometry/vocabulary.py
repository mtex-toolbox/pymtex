# the strings the geometry layer's arguments take, as Literal types an editor completes in quotes;
# written by tools/staticapi.py from the tables they name, never by hand
from typing import Literal

# the point groups crystalFrame and symmetry take
PointGroupName = Literal["1", "C1", "-1", "Ci", "S2", "triclinic", "211", "C2", "m11", "Cs", "C1h", "2/m11", "C2h",
                         "121", "2", "1m1", "m", "12/m1", "2/m", "112", "11m", "112/m", "monoclinic", "222", "D2", "22",
                         "V", "2mm", "C2v", "m2m", "mm2", "mmm", "D2h", "2/m2/m2/m", "Vh", "orthotropic",
                         "orthorhombic", "3", "C3", "-3", "C3i", "S6", "321", "D3", "32", "3m1", "C3v", "3m", "-3m1",
                         "D3d", "-32/m", "-3m", "312", "31m", "-31m", "trigonal", "4", "C4", "-4", "S4", "4/m", "C4h",
                         "422", "D4", "42", "4mm", "C4v", "-42m", "D2d", "Vd", "-4m2", "4/mmm", "D4h", "4/m2/m2/m",
                         "tetragonal", "6", "C6", "-6", "C3h", "6/m", "C6h", "622", "D6", "62", "6mm", "C6v", "-62m",
                         "D3h", "-6m2", "6/mmm", "D6h", "6/m2/m2/m", "hexagonal", "23", "T", "m-3", "Th", "m3", "2/m-3",
                         "432", "O", "43", "-43m", "Td", "m-3m", "Oh", "m3m", "4/m-32/m", "cubic", "532", "I",
                         "-5-32/m", "Ih", "icosahedral"]

# the alignments of the Cartesian axes to the lattice crystalFrame takes
AlignmentName = Literal["X||a", "X||b", "X||c", "X||a*", "X||b*", "X||c*", "X||m", "Y||a", "Y||b", "Y||c", "Y||a*",
                        "Y||b*", "Y||c*", "Y||m", "Z||a", "Z||b", "Z||c", "Z||a*", "Z||b*", "Z||c*", "Z||m", "EDAX",
                        "TSL", "OIM"]

# the Euler angle conventions
EulerConventionName = Literal["Bunge", "ZXZ", "Matthies", "ZYZ", "ABG", "nfft", "Roe", "Kocks", "Canova"]

# the eight plotting conventions with both screen axes along reference axes
ConventionName = Literal["y↑→x", "y↓→x", "x↑→y", "x↓→y", "x←↑y", "x←↓y", "y←↑x", "y←↓x"]

# the colour names, matplotlib's
ColorName = Literal["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black",
                    "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse",
                    "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
                    "darkgoldenrod", "darkgray", "darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen",
                    "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue",
                    "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
                    "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia",
                    "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "green", "greenyellow", "grey", "honeydew",
                    "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
                    "lemonchiffon", "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray",
                    "lightgreen", "lightgrey", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
                    "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen",
                    "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
                    "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred",
                    "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive",
                    "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
                    "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple",
                    "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen",
                    "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow",
                    "springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat",
                    "white", "whitesmoke", "yellow", "yellowgreen", "b", "g", "r", "c", "m", "y", "k", "w"]

__all__ = []

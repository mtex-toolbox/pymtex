# the strings the plotting layer's arguments take, as Literal types an editor completes in quotes;
# written by tools/staticapi.py from the tables they name, never by hand
from typing import Literal

# the colour maps, MTEX's and matplotlib's
ColormapName = Literal["WhiteJet", "LaboTeX", "blue2red", "red2blue", "white2black", "black2white", "parula", "Accent",
                       "afmhot", "autumn", "berlin", "binary", "Blues", "bone", "BrBG", "brg", "BuGn", "BuPu", "bwr",
                       "cividis", "CMRmap", "cool", "coolwarm", "copper", "cubehelix", "Dark2", "flag", "gist_earth",
                       "gist_gray", "gist_grey", "gist_heat", "gist_ncar", "gist_rainbow", "gist_stern", "gist_yarg",
                       "gist_yerg", "GnBu", "gnuplot", "gnuplot2", "gray", "Grays", "Greens", "grey", "Greys", "hot",
                       "hsv", "inferno", "jet", "magma", "managua", "nipy_spectral", "ocean", "okabe_ito", "Oranges",
                       "OrRd", "Paired", "Pastel1", "Pastel2", "pink", "PiYG", "plasma", "PRGn", "prism", "PuBu",
                       "PuBuGn", "PuOr", "PuRd", "Purples", "rainbow", "RdBu", "RdGy", "RdPu", "RdYlBu", "RdYlGn",
                       "Reds", "seismic", "Set1", "Set2", "Set3", "Spectral", "spring", "summer", "tab10", "tab20",
                       "tab20b", "tab20c", "terrain", "turbo", "twilight", "twilight_shifted", "vanimo", "viridis",
                       "winter", "Wistia", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd"]

# the markers, by MATLAB's names and matplotlib's letters
MarkerName = Literal["square", "diamond", "circle", "triangle", "pentagram", "hexagram", "plus", "cross", "point",
                     "asterisk", "s", "D", "o", "^", "p", "h", "+", "x", ".", "*", "d", "v", "<", ">", "none"]

# the spherical projections
ProjectionName = Literal["earea", "schmidt", "eangle", "stereo", "stereographic", "edist", "orthographic", "gnomonic",
                         "plain"]

# the figure sizes by name
FigureSizeName = Literal["tiny", "small", "medium", "normal", "large", "huge"]

__all__ = []

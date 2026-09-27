# # Exporting figures
#
# Export is the last step of making a figure. Add its colour scale, labels,
# and other reading aids first, then choose a file type that suits the plotted
# objects. Dense spatial maps usually belong in a bitmap. Lines, markers, and
# text usually benefit from a vector format.
#
# Every MTEX figure is an ordinary matplotlib figure. You can export it through
# the *Save* button of the figure window or with matplotlib's
# [savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html)
# command. This route usually leaves a broad white margin around an MTEX plot.
# On a spatial EBSD map, it may also interpolate between neighbouring pixels. That
# interpolation can invent colours which are absent from the data.
#
# [saveFigure](saveFigure.html) handles these details for MTEX figures and
# produces cropped, publication-ready files. Complete the workflow from
# [Annotations](Annotations.html) before calling it.
#
# ## A first export
#
# Start with an orientation map. Each coloured cell is one measurement, so
# an exported bitmap must keep neighbouring cell colours discrete. This is a
# case where preserving the values matters more than making cell boundaries
# look smooth.

# +
import os
import tempfile

from mtex import *

plottingConvention.default('y↑→x')
ebsd = mtexdata('forsterite')

ebsdF = ebsd['Forsterite']
plot(ebsdF, ebsdF.orientations)
# -

# Notice the sharp changes of colour between neighbouring measurements. The
# exported image should retain those changes rather than blend across them.
#
# The filename extension determines the output format. The vector formats
# are `pdf`, `eps`, `ps` and `svg`. The bitmap formats are `png`, `jpg`, `tif`,
# and the others Pillow writes. `saveFigure` overwrites an existing file with the
# same name, so choose the output path deliberately. Check that the file you asked
# for exists before relying on it, as the next step does.

# +
# write into a temporary folder rather than the MTEX folder
outDir = os.path.join(tempfile.gettempdir(), 'mtex-docrun', 'scratch')
os.makedirs(outDir, exist_ok=True)
outMap = os.path.join(outDir, 'forsterite.png')

saveFigure(outMap)
# -

# Confirm that the export produced a file before using it elsewhere. The
# assertion remains silent when it succeeds. It stops the page if exporting
# failed.

assert os.path.isfile(outMap), 'The orientation-map export was not created.'

# ## Vector graphics and bitmaps
#
# A vector file stores drawn objects as paths and text. A bitmap stores a
# fixed grid of coloured pixels. [saveFigure](saveFigure.html) treats the
# two groups differently. Choose a vector file when paths and text should
# remain sharp at any magnification. Choose a bitmap when the figure contains
# many cells or patches and a compact, fixed-resolution image is preferable.
#
# *Vector formats* are written by matplotlib's vector backends. `saveFigure`
# crops the page to the drawn content, so the resulting page is exactly as
# large as the plot and has no broad margin. A spatial EBSD map contains one
# patch per measurement. It can therefore make a very large vector file, so a
# bitmap is usually the better choice for maps.
#
# *Bitmap formats* are written at 1.5 times the screen resolution. For a
# spatial map, MTEX increases the factor to 2.5 and turns the interpolation
# of its images off. This prevents new colours from being introduced
# between map cells. It matters when orientations will be read back from
# their colour.
#
# MATLAB's two options `'crop'` and `'pdf'`, which call `pdfcrop` and
# `epstopdf` from a TeX distribution, are not needed: every format is cropped,
# and a PDF is written directly.
#
# ## Controlling output size
#
# Bitmap resolution follows the figure size. The output is enlarged
# by the factors above, but its proportions still come from that
# figure. To obtain a larger or smaller image, create a larger or smaller
# figure with `figSize` before exporting it.

plot(ebsdF, ebsdF.orientations, figSize='small')

# This map uses a smaller canvas than the first map. Saving it now would
# therefore produce a smaller bitmap with the same map content.
#
# The recognized values are `'tiny'`, `'small'`, `'normal'`, `'large'` and
# `'huge'`, and they are relative to the default figure size. For a one-off
# export, passing `figSize` to the plot keeps the choice local to that
# figure.
#
# For a multi-plot, `figSize` refers to each panel rather than to the whole
# figure. Three pole figures at `'normal'` therefore give three panels the
# size of a single normal-sized plot.

plotPDF(ebsdF.orientations, Miller([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ebsdF.CS), contourf=True, figSize='normal')
mtexColorbar()

# The figure is three times as wide as a single pole figure, while each panel
# keeps the size it would have on its own. Increasing `figSize` enlarges
# every panel and the figure with them. Export the complete figure as a
# bitmap and verify the file just as for the map.

outPole = os.path.join(outDir, 'poleFigures.png')
saveFigure(outPole)
assert os.path.isfile(outPole), 'The pole-figure export was not created.'

# ## Exporting data instead of a picture
#
# A figure file stores the visual result, not the numerical values behind it.
# When those values are the required result, use a class-specific export
# method instead. Most MTEX classes provide one. Examples are
# [export](quaternion.export.html) for orientations,
# [export](PoleFigure.export.html) for pole-figure data, and
# [export](SO3Fun.export.html) for ODFs. Their corresponding chapters
# describe the data formats. This distinction avoids treating pixels in a
# saved picture as though they were the original measurements.

# clean up
os.remove(outMap)
os.remove(outPole)

# ## References
#
# * The [matplotlib savefig documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html)
#   describes the backends `saveFigure` writes through; MATLAB's `saveFigure` uses
#   O. Woodford and Y. Altman's [export_fig](https://github.com/altmany/export_fig).
#
# ## Next
#
# Export completes the figure workflow. Continue with
# [Spherical Projections](SphericalProjections.html) to understand how the
# projection itself changes the positions, shapes, and areas seen in pole
# figures and other spherical plots.

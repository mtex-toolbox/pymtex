# # EBSD Maps and SEM Images
#
# An EBSD map and a forescatter or backscattered-electron (BSE) image can record the same
# area of a specimen. Each is a raster: a rectangular array whose entries sample that area.
#
# Comparing the two requires two independent questions to be answered:
#
# * where on the specimen each raster sits - its
#   [reference frame](referenceFrame.referenceFrame.html)
# * which specimen direction each array index follows - its
#   [layout](gridLayout.gridLayout.html)
#
# A reference frame is the coordinate system in which data are expressed. Its identity,
# basis and plotting convention say what the axes mean and how they are drawn. A layout
# instead says what `img[i, j]` means: the direction in which the row index `i` and column
# index `j` increase.
#
# Read [Gridded EBSD Data](EBSDGrid.html) first if matrix-shaped EBSD data is new to you.
# [Reference Frame Alignment](EBSDReferenceFrame.html) explains how map coordinates and
# Euler angles are related.
#
# This page uses an EBSD map and four forescatter images of the same 20 × 15 micron WC-Co
# area. It first reconciles their frames, then their layouts, and finally their pixel grids.

import numpy as np

from mtex import *

plottingConvention.default('y↓→x')

ebsd = mtexdata('trueEbsdWCCoSmall', verbose=False)

img = ebsd.meta['trueEbsdImgs']

# ## Put the Map and Images in One Sequence
#
# A [mapImage](mapImage.mapImage.html) combines a raster with the geometry that locates it.
# The geometry consists of the centre position of pixel `(0, 0)`, one step vector for each
# array dimension and a reference frame.
#
# Passing an EBSD map supplies that geometry from the map. A plain image has only its pixel
# size here. MTEX initially gives it an independent frame. The map channel and the SEM
# images then form one `mapImageList` rather than two different container types.

imgList = mapImageList([mapImage(ebsd.bc, ebsd, name='bcImg'),
                        mapImage(img.fsdT1, dxy=img.pixSzImg, name='fsdT1'),
                        mapImage(img.fsdT10, dxy=img.pixSzImg, name='fsdT10')])
imgList

# The table reports image size, pixel size, frame and layout for every entry. The EBSD
# band-contrast channel is coarser than the two forescatter images, but all three initially
# have the same screen alignment.

plot(imgList, layout=[1, 3])

# The same grain outlines appear upright and in the same part of all three panels. Their
# grey values need not match because the three imaging signals measure different contrast.
#
# ## What an Image Does Not Know
#
# The map entry brings the map's specimen frame with it. Its display names the specimen
# axes and states their screen directions.

imgList[0].frame

# A plain image array cannot reveal which specimen direction its horizontal axis follows.
# Until that relation is supplied, MTEX uses a separate frame. Its axes `iX`, `iY` and `iZ`
# expose rather than hide that independence.

imgList[1].frame

# ## Two Pictures That Disagree
#
# The original data were collected in one session, so the images above agree. To construct
# the other case, rotate the EBSD map through 90 degrees. This represents a stage rotation
# between the map and image acquisitions.

# +
ebsd = rotate(ebsd, 90 * degree)

imgList = mapImageList([mapImage(ebsd.bc, ebsd, name='bcImg'),
                        mapImage(img.fsdT1, dxy=img.pixSzImg, name='fsdT1'),
                        mapImage(img.fsdT10, dxy=img.pixSzImg, name='fsdT10')])
imgList
# -

# The rotation moved the data, not the recorded convention, so all three frame columns
# still read down-then-right. MATLAB stores the rotated map anew in its default layout, a
# 128 × 96 array, and its table shows nothing but the interchanged size. The map here keeps
# its 96 × 128 array and turns its layout with the positions: its rows now run against x and
# its columns along y, the `row←↓col` of the layout column, the one sign in the table that
# the three rasters have fallen out of step.

plot(imgList, layout=[1, 3])

# The pictures show it plainly. The EBSD panel now stands upright beside two
# landscape images and occupies a different specimen extent, while the frame indicators in
# the corners still read the same arrangement in all three panels.
#
# ## Establish the Frame Relation on Screen
#
# A plotting convention states how a reference frame is laid out on screen. It never
# changes the data. Set the map frame so that its x axis points up and its y axis points
# right. The image frames keep their original setting.
#
# This alignment is experimental information. No inspection of the array values can recover
# it. Use acquisition metadata or a known specimen feature.

# +
ebsdFrame = imgList[0].frame
ebsdFrame.how2plot = 'x↑→y'

plot(imgList, layout=[1, 3])
# -

# The corresponding grain outlines are now the same way up. Changing `how2plot` only
# established how the two frames appear on screen. It has not yet expressed the images in
# the map frame.
#
# [byScreenAlignment](orientation.byScreenAlignment.html) records the assertion that the
# plotted frames are physically aligned.
# [transformReferenceFrame](mapImage.transformReferenceFrame.html) then re-expresses every
# entry in the map frame.

imgList = transformReferenceFrame(imgList, ebsdFrame, 'byScreenAlignment')
imgList

# The new table shows one common frame and the corresponding layouts. This operation only
# transposes or flips the arrays and updates their geometry; it does not resample any
# values.

plot(imgList, layout=[1, 3])

# The panels remain the same way up after the frame change. That unchanged appearance is the
# point. The arrays and their frame labels changed together, while the physical pictures did
# not.
#
# ## How the Array Is Stored
#
# In MATLAB the transformed table also shows that the map changed from 128 × 96 back to
# 96 × 128; here it was already stored with its rows against x, which is the layout the
# map frame is drawn in, so it kept its array. This is the second question, and it is
# independent of the specimen frame.
#
# A [gridLayout](gridLayout.gridLayout.html) contains two directions. The first is the
# direction in which the row index advances, and the second is the direction in which the
# column index advances.

imgList[0].layout

# Every `mapImage` states its layout, and so does every gridded EBSD map.

ebsd.layout

# ## Put a Map in an Image Layout
#
# [gridify](EBSD.gridify.html) accepts a layout and stores a square-grid map in that
# order. [Gridded EBSD Data](EBSDGrid.html) introduces the named layouts `'columnMajor'`
# and `'rowMajor'`. Give any other signed pair of axis directions as a `gridLayout`.
#
# Start again from the imported map. Suppose the detector was mounted a quarter turn from
# the scan. Its rows run against x and its columns along y.

# +
ebsd = mtexdata('trueEbsdWCCoSmall', verbose=False)

gL = gridLayout(-xvector, yvector)
gL
# -

# Store the map in that layout and display its resulting matrix shape.

# +
ebsdI = gridify(ebsd, gL)

ebsdI.shape
# -

# No measurement was resampled or invented. On the same square lattice, a layout change
# applies only a transpose and two possible flips. It is safe for orientation data. For this
# pair the result is exactly a quarter turn.

np.array_equal(ebsdI.bc, np.rot90(ebsd.bc))

# Returning to the original layout is exact as well.

np.array_equal(gridify(ebsdI, ebsd.layout).bc, ebsd.bc)

# The specimen has not moved. A layout changes how measurements are stored, not where they
# are. MTEX adjusts the screen mapping, so both maps still plot the same way up.

# +
newMtexFigure(layout=[1, 2])
plot(ebsd, ebsd.bc, micronbar='off')
mtexColorMap('gray')
mtexTitle('as imported')
nextAxis()
plot(ebsdI, ebsdI.bc, micronbar='off')
mtexColorMap('gray')
mtexTitle('row ||-x, col ||y')
# -

# The same WC grains occupy the same screen positions in both panels even though the two
# band-contrast matrices are quarter-turn permutations.
#
# ## Compare the Rasters Pixel by Pixel
#
# Agreeing frames and layouts makes the rasters geometrically comparable, but pixel-wise
# operations also require one common grid. The EBSD spacing is 0.159 micron here, whereas
# the image spacing is 0.0795 micron.
#
# [interp](mapImage.interp.html) samples an image at arbitrary positions. A bare array
# carries no physical origin, so state where the centre of its first pixel lies. The origin
# and pixel size must use the same length unit as the EBSD positions.

# +
mgI = mapImage(img.fsdT1, dxy=img.pixSzImg, origin=ebsd.pos[0, 0])

ebsd.fsdT1 = interp(mgI, ebsd.pos)
# -

# The image is now a per-pixel property: one value per measurement point, indexed and subset
# in lockstep with the map. It travels through cropping, gridding and indexing with the EBSD
# data. The two channels can now be passed side by side to image-processing tools.
#
# Image values are interpolated linearly by default, and positions outside the image return
# `NaN`. No EBSD orientation is interpolated in this step.

# +
newMtexFigure(layout=[1, 2])
plot(ebsd, ebsd.bc, micronbar='off')
mtexTitle('band contrast')
nextAxis()
plot(ebsd, ebsd.fsdT1, micronbar='off')
mtexTitle('forescatter on EBSD grid')
mtexColorMap('gray')
# -

# The same grain-scale features occupy roughly the same places in both panels. Their edges do
# not overlay perfectly, which shows that matching frames, layouts, origins and pixel sizes is
# necessary but not sufficient.
#
# ## From Bookkeeping to Registration
#
# Beam drift during a scan, camera motion between acquisitions and specimen tilt all change
# positions continuously. They are not layout problems. Each is a
# [spatial transform](EBSDSpatialTransform.html) that can be fitted from corresponding image
# features and then removed.
#
# [TrueEBSD Distortion Correction](EBSDTrueEbsd.html) performs that workflow. It starts with
# exactly the sequence built here: entries that agree about the specimen frame, array order
# and pixel grid.
#
# ## Further Reading
#
# Britton et al., [*Tutorial: Crystal orientations and EBSD - or which way is
# up?*](https://doi.org/10.1016/j.matchar.2016.04.008). *Materials Characterization* 117
# (2016), 113-126. The paper gives a practical calibration of specimen, diffraction-pattern
# and crystal frames.
#
# [ISO 24173:2024](https://www.iso.org/standard/82749.html), *Microbeam analysis -
# Guidelines for orientation measurement using electron backscatter diffraction*. The
# standard covers specimen preparation, instrument configuration, calibration and
# acquisition.
#
# Tong and Britton, [*TrueEBSD: Correcting spatial distortions in electron backscatter
# diffraction maps*](https://doi.org/10.1016/j.ultramic.2020.113130). *Ultramicroscopy* 221
# (2021), 113130. The paper describes the method used by the next tutorial.
#
# Zitová and Flusser, [*Image registration methods: a
# survey*](https://doi.org/10.1016/S0262-8856(03)00137-9). *Image and Vision Computing* 21
# (2003), 977-1000. The survey relates feature detection, matching, transform fitting and
# resampling.
#
# ## Technical Details
#
# A list of images is a `mapImageList`, a Python list that displays as MATLAB's table of a
# `mapImage` array; `rescale`, `imboxfilt`, `transformReferenceFrame` and `plot` take one
# image or a list. Pixels are counted from zero, so the origin is the centre of pixel
# `(0, 0)`. Moving a map never reorders its arrays: `rotate` keeps the array of a gridded map
# and turns its layout with the positions, where MATLAB stores the rotated map anew in its
# default layout. A new order is asked for explicitly, by `gridify(ebsd, gL)` or
# `transformReferenceFrame`.

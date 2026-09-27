# # Legends
#
# A legend pairs a plotted symbol or line with a label. Use one when a small
# number of discrete objects must be distinguished, such as selected grains,
# boundary classes, or reference directions.
#
# In MTEX, legend entries are opt in. An object appears only when it is
# plotted with a `DisplayName`. This is the opposite of MATLAB's default
# behaviour. It prevents markers, construction lines, and a colour-coded
# background from each acquiring an unhelpful entry.
#
# ## Choose the legend entries
#
# Name only the objects that the reader needs to distinguish. This example
# names a point and a great circle. The pole at the centre remains unnamed.

# +
from mtex import *

plot(vector3d.X, upper=True, DisplayName='point')
hold(True)

circle(vector3d.X, DisplayName='great circle', lineColor='r', lineWidth=2)

plot(vector3d.Z, Marker='p', MarkerSize=30)
hold(False)

lgd = legend('show')
# -

# ## Read and adjust the result
#
# The figure contains three plotted objects, but the legend has two entries.
# The central star is visible in the axes but absent from the legend because
# it has no `DisplayName`.
#
# After MTEX has selected the entries, `legend` hands its options to matplotlib's
# [legend](https://matplotlib.org/stable/api/legend_api.html). Calling it again
# replaces the legend: an outside location avoids covering the spherical plot,
# and a title explains what the entries classify.

lgd = legend('show', location='eastoutside', title='Geometry')

# ## Choose the right colour guide
#
# A legend represents discrete identities. When colour represents a numerical
# value, use a colour bar because it shows the complete value-to-colour scale.
# When colour represents a direction or an orientation, use a colour key that
# maps each possible direction or orientation to a colour. The next page
# introduces numerical colour scales; [IPF Maps](EBSDIPFMap.html) explains
# direction colour keys for orientation maps.
#
# The matplotlib
# [Legend properties](https://matplotlib.org/stable/api/legend_api.html#matplotlib.legend.Legend)
# provide further controls for placement and styling.
#
# ## References
#
# * S. R. Midway,
#   [Principles of Effective Data Visualization](https://doi.org/10.1016/j.patter.2020.100141),
#   _Patterns_ 1 (2020), 100141, explains how selective legends
#   and consistent visual scales make plots easier to compare.
#
# ## Next
#
# Continue with [Color Maps](ColorMaps.html) to control how numerical values
# are translated into colours.
#
# ## Technical Details
#
# MATLAB moves and titles the legend by setting `lgd.Location` and `lgd.Title.String` on its
# handle; a matplotlib legend cannot be moved outside its axes after it is made, so the page
# calls `legend` a second time with `location` and `title`.

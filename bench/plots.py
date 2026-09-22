# rendered checks of the spherical plots: every figure is written to bench/out for a look,
# numbered as ../mtexJulia/dev/plots.jl numbers its figures
import os

import matplotlib
import matplotlib.pyplot
import numpy as np

matplotlib.use("Agg")

from mtex import (Miller, S2AxisFieldHarmonic, S2FunHarmonic, contour, contourf, crystalFrame, degree, discreteSample,
                  fundamentalRegion, fundamentalSector, gcm, grad, hold, ipfHSVKey, orientation, plot, plot3d, plotIPDF, plotPDF,
                  plotSection, scatter, sigmaSections, specimenFrame, text, unimodalODF, vector3d, xvector, yvector, zvector,
                  HSVDirectionKey, TSLDirectionKey, HKLDirectionKey)

out = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
cs, ss = crystalFrame("m-3m"), specimenFrame("222")
tri = crystalFrame("-3m")


def save(name):
  gcm().savefig(os.path.join(out, name), dpi=120)
  matplotlib.pyplot.close(gcm().fig)


# 1 directions as markers on both hemispheres, and axes on one
vs = vector3d(vector3d.rand(40, rng=rng), frame=ss)
plot(vs, grid=True)
save("01_scatter.png")
plot(vector3d.rand(40, antipodal=True, rng=rng), MarkerSize=8, MarkerFaceColor="red")
save("02_axes.png")
plot(vs, projection="stereo", upper=True, label=[str(i) for i in range(1, 41)], MarkerSize=6)
save("03_stereo.png")

# 2 crystal directions, symmetrised, and in the fundamental sector
m = Miller([1, 1, 1, 2], [0, 1, 1, 1], [0, 0, 1, 0], cs)
labels = [m[i].char() for i in range(4)]
plot(m, symmetrised=True, upper=True, grid=True)
hold(True)
text(m, labels)
hold(False)
save("04_miller.png")
plot(m, fundamentalRegion=True, label=labels)
save("05_sector.png")
plot(fundamentalSector(tri))
save("06_sector_trigonal.png")

# 3 a function on the sphere: pcolor, contourf and contours, colour bar
s = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True) + 3
plot(s, colorbar=True, minmax=True)
save("07_pcolor.png")
contourf(s, upper=True, colormap="LaboTeX")
save("08_contourf.png")
contour(s, upper=True, contours=8)
save("09_contour.png")
plot(s, projection="plain")
save("10_plain.png")

# 4 a density of directions
contourf(vector3d.rand(200, rng=rng), halfwidth=10 * degree, minmax=True)
save("11_density.png")

# 5 pole figures and inverse pole figures of an ODF
odf = (unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=15 * degree)
       + 0.5 * unimodalODF(orientation.byEuler(100 * degree, 20 * degree, 60 * degree, cs, ss), halfwidth=15 * degree))
plotPDF(odf, Miller([1, 1, 1], [0, 1, 1], [0, 0, 1], cs))
save("12_pdf.png")
plotPDF(odf, Miller([1, 1], [0, 1], [0, 1], cs), contourf=True, contours=8, colorbar=True)
save("13_pdf_contourf.png")
plotIPDF(odf, [xvector, yvector, zvector])
save("14_ipdf.png")
odfTri = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, tri, specimenFrame("1")), halfwidth=15 * degree)
plotIPDF(odfTri, zvector, contourf=True)
save("15_ipdf_trigonal.png")
plotPDF(odfTri, [Miller(0, 0, 0, 1, tri), Miller(1, 0, -1, 0, tri)], complete=True)
save("16_pdf_trigonal.png")

# 6 a vector field and an axis field
G = grad(s)
plot(G, upper=True, resolution=10 * degree)
save("17_quiver.png")
A = S2AxisFieldHarmonic(lambda v: vector3d(v.cross(zvector).data + 0.3 * v.cross(xvector).data, antipodal=True), bandwidth=24)
plot(A, upper=True)
save("18_axisfield.png")
contourf(s, upper=True)
hold(True)
plot(G, color="white")
hold(False)
save("19_overlay.png")

# 7 a symmetric function with its sector, contour lines on a Laue group
sc = S2FunHarmonic(rng.standard_normal(81) + 1j * rng.standard_normal(81), tri, real=True) + 4
contourf(sc, colorbar=True)
save("20_symmetric.png")
# 8 ODF sections: phi2 and sigma sections of the ODF, a list of orientations on them, and
# axis angle sections of a misorientation function
plotSection(odf, "phi2", sections=6)
save("40_phi2.png")
plotSection(odf, "sigma", sections=6, contourf=True, colorbar=True)
save("41_sigma.png")
ori = orientation.byEuler(30 * degree + 5 * degree * rng.standard_normal(500), 50 * degree + 5 * degree * rng.standard_normal(500),
                          10 * degree + 5 * degree * rng.standard_normal(500), cs, ss)
plotSection(ori, sigmaSections(cs, ss, sections=6))
save("42_sigma_points.png")
mdf = unimodalODF(orientation.byAxisAngle([1, 1, 1], 40 * degree, cs, cs), halfwidth=10 * degree)
plotSection(mdf, sections=6)
save("43_axisangle.png")
# 9 the colour keys, each drawn as the sector it colours: the cubic triangle with the
# primaries on its corners, the folded sectors with a white centre in one half and a black
# one in the other, the vendors' keys, and the key of a specimen frame
for i, name in enumerate(("m-3m", "622", "-3m", "mmm", "4/mmm", "2/m")):
  HSVDirectionKey(crystalFrame(name)).plot()
  save(f"{21 + i:02d}_key_{name.replace('/', '_')}.png")
TSLDirectionKey(crystalFrame("m-3m")).plot()
save("27_key_tsl.png")
HKLDirectionKey(crystalFrame("6/mmm")).plot()
save("28_key_hkl.png")
ipfHSVKey(specimenFrame("1")).plot()
save("29_key_specimen.png")
# 9 pole figures of a list of orientations, and the Dubna measurements
ori = discreteSample(odf, 3000, rng=np.random.default_rng(1))
plotPDF(ori, Miller([1, 1], [0, 1], [0, 1], cs))
save("44_pdf_points.png")
plotPDF(ori, Miller([1, 1], [0, 1], [0, 1], cs), contourf=True)
save("45_pdf_density.png")
plotIPDF(ori, np.degrees(ori.angle()), [xvector, zvector], colorbar=True)
save("46_ipdf_values.png")
dubna = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data")
if os.path.exists(os.path.join(dubna, "PoleFigure", "dubna")):
  from mtex import calcODF, mtexdata, plotDiff
  pf = mtexdata("dubna", root=dubna, download=False)
  plot(pf)
  save("47_pf.png")
  plot(pf, smooth=True, colorbar=True)
  save("48_pf_smooth.png")
  plotDiff(pf, calcODF(pf))
  save("49_pf_diff.png")

# 11 plots in space: directions, a function and its gradient on the sphere, the fundamental
# region in axis angle space with orientations in it, and the region of cubic misorientations
plot3d(vs)
save("50_scatter3d.png")
plot3d(s, colorbar=True)
save("51_fun3d.png")
plot3d(s, radial=True)
save("52_fun3d_radial.png")
plot3d(G)
save("53_quiver3d.png")
oR = fundamentalRegion(cs, ss)
plot(oR)
save("54_region.png")
scatter(ori)
save("55_ori_axisangle.png")
scatter(ori, Euler=True)
save("56_ori_euler.png")
scatter(ori, Rodrigues=True, region=False)
save("57_ori_rodrigues.png")
plot(fundamentalRegion(cs, cs))
save("58_misregion.png")

# 10 EBSD and grain maps, on the Forsterite map of the local sample data where it is there
sample = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD", "Forsterite.ctf")
if os.path.exists(sample):
  from mtex import calcGrains, loadEBSD, quiver
  ebsd = loadEBSD(sample)
  ebsd = ebsd[(ebsd.x < 8000) & (ebsd.y < 6000)]
  plot(ebsd, legend=True)
  save("30_map_phase.png")
  plot(ebsd, ebsd.bc, colorbar=True, backend="imagesc")
  save("31_map_bc.png")
  plot(ebsd, "orientation", legend=True)
  save("32_map_ipf.png")
  fo = ebsd["Forsterite"]
  plot(fo, fo.orientations)
  save("33_map_ori.png")
  g = calcGrains(ebsd["indexed"], angle=10 * degree)
  plot(g)
  save("34_grains.png")
  plot(g["Forsterite"], g["Forsterite"].meanOrientation)
  hold(True)
  plot(g.boundary, lineWidth=1.5)
  hold(False)
  save("35_grains_ori.png")
  gB = g.boundary[g.boundary.hasPhase("Forsterite", "Forsterite")]
  plot(ebsd, ebsd.bc, colormap="gray")
  hold(True)
  plot(gB, np.degrees(gB.misorientation.angle()), lineWidth=2, colorbar=True)
  hold(False)
  save("36_boundary_angle.png")
  plot(g)
  hold(True)
  quiver(g, g.longAxis(), color="w")
  hold(False)
  save("37_quiver.png")
  plot(g, g.GOS, colorbar=True, how2plot="y↑→x", coordinates=True)
  save("38_gos.png")
  plot(ebsd[(ebsd.x < 1000) & (ebsd.y < 800)], "orientation", backend="patch", edgeColor="k", lineWidth=0.3)
  save("39_cells.png")
# 12 crystal shapes: the hexagonal prism of titanium with two slip systems, quartz with its
# forms coloured and a face selected, olivine with an inner plane and direction, a twin, and
# the shapes of the Forsterite grains in their map
from mtex import crystalShape, plotInnerDirection, plotInnerFace, slipSystem
ti = crystalFrame("6/mmm", [3, 3, 4.7], mineral="Titanium")
cS = crystalShape.hex(ti)
plot(cS, faceAlpha=0.2)
save("60_hex.png")
sS = [slipSystem.pyramidal2CA(ti), slipSystem.pyramidalA(ti)]
plot(cS, faceAlpha=0.2)
hold(True)
plot(cS, sS[1], faceColor="blue")
plot(cS, sS[0], faceColor="red")
hold(False)
save("61_slip_systems.png")
q = crystalShape.quartz(simple=True)
plot(q, colored=True)
save("62_quartz_colored.png")
plot(q)
hold(True)
plot(q[Miller(0, -1, 1, 0, q.CS)], faceColor="DarkRed")
hold(False)
save("63_quartz_face.png")
ol = crystalShape.olivine()
plot(ol, faceAlpha=0.2, colored=True)
hold(True)
plotInnerFace(ol, Miller(1, 0, -1, ol.CS), faceColor="red", faceAlpha=0.5)
plotInnerDirection(ol, Miller(1, 0, 1, ol.CS, "uvw"), faceColor="blue", arrowWidth=0.05)
hold(False)
save("64_olivine_inner.png")
twinning = orientation.map(Miller(0, 1, -1, -2, ti), Miller(0, -1, 1, -2, ti), Miller(2, -1, -1, 0, ti), Miller(2, -1, -1, 0, ti))
plot(cS, faceAlpha=0.5)
hold(True)
plot(twinning * cS * 0.9, faceColor="orange")
hold(False)
save("65_twin.png")
if os.path.exists(sample):
  fo = g["Forsterite"]
  cSFo = crystalShape.forsterite(fo.CS)
  color = ipfHSVKey(fo.CS).orientation2color(fo.meanOrientation) if hasattr(ipfHSVKey(fo.CS), "orientation2color") else None
  big = fo.numPixel > 50
  plot(fo, fo.meanOrientation, alpha=0.5)
  hold(True)
  plot(fo[big], 0.7 * cSFo, faceColor=color[big] if color is not None else None, faceAlpha=0.7, lineWidth=1)
  hold(False)
  save("66_shapes_on_map.png")
  plot(fo[big], cSFo, colored=True)
  save("67_shapes_colored.png")
from mtex import plane3d
plot(q, faceAlpha=0.6)
hold(True)
plot(plane3d(vector3d(1, 1, 1), 0), faceColor=[0.75, 0.75, 0.75], edgeColor=None)
hold(False)
save("68_plane.png")
# 13 tensors: the seismic overview of olivine, and the interference colours of its
# refractive index tensor through a 30 micrometre plate
from mtex import plotSeismicVelocities, spectralTransmission
from mtex.tensors.tensor import refractiveIndexTensor, stiffnessTensor
olv = crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="olivine")
M = np.array([[320.5, 68.15, 71.6, 0, 0, 0], [68.15, 196.5, 76.8, 0, 0, 0], [71.6, 76.8, 233.5, 0, 0, 0],
              [0, 0, 0, 64, 0, 0], [0, 0, 0, 0, 77, 0], [0, 0, 0, 0, 0, 78.7]])
plotSeismicVelocities(stiffnessTensor(M, olv, density=3.355))
save("70_seismic_velocities.png")
rI = refractiveIndexTensor(np.diag([1.635, 1.651, 1.670]), olv)
plot(spectralTransmission(rI, 30000, tau=45 * degree), rgb=True, upper=True, complete=True)
save("71_spectral_transmission.png")
print(f"figures written to {out}")

# the drawing of a scene3d: PyVista for the pictures and the interactive window, matplotlib's
# 3D axes when PyVista is missing, and a standalone HTML page that draws the scene with
# three.js in the browser. Unlit and flat, as MATLAB draws a patch without a light.
import base64
import json
import zlib

import numpy as np

from .scene3d import lines3d, mesh3d, picture3d, points3d, slices3d

def hasPyVista():
  """whether PyVista can be imported, the renderer of the 3D scenes"""
  try:
    import pyvista  # noqa: F401
    return True
  except ImportError:
    return False


# ------------------------------------------------------------------ PyVista

# the polygons in VTK's layout: the corner count before the corners of every face
def _vtkFaces(m):
  n = m.numCorners
  out = np.empty(len(m.idx) + len(m), dtype=np.int64)
  head = m.ptr[:-1] + np.arange(len(m))
  out[head] = n
  mask = np.ones(len(out), dtype=bool)
  mask[head] = False
  out[mask] = m.idx
  return out


def _uint8(rgb):
  return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# the corners of the voxel faces on a slice, a grid of (nb + 1) x (nc + 1) points over its rectangle
def _sliceGrid(corners, nb, nc):
  u, v = np.meshgrid(np.linspace(0, 1, nb + 1), np.linspace(0, 1, nc + 1), indexing="ij")
  return corners[0] + u[..., None] * (corners[1] - corners[0]) + v[..., None] * (corners[2] - corners[0])


def pyvistaPlotter(scene, off_screen=False, window_size=(800, 800), plotter=None):
  """a PyVista plotter drawing the scene, its camera set by the plotting convention"""
  import pyvista as pv
  p = pv.Plotter(off_screen=off_screen, window_size=list(window_size)) if plotter is None else plotter
  p.set_background("white")
  for it in scene.items:
    if isinstance(it, mesh3d):
      poly = pv.PolyData(it.V, _vtkFaces(it))
      poly.cell_data["rgb"] = _uint8(it.faceColor)
      edges = it.edgeColor is not None and it.edgeAlpha > 0
      p.add_mesh(poly, scalars="rgb", rgb=True, opacity=it.faceAlpha, lighting=False, show_edges=edges,
                 edge_color=tuple(it.edgeColor) if edges else "black", edge_opacity=it.edgeAlpha, line_width=it.lineWidth)
    elif isinstance(it, slices3d):
      for rgb, vis, corners in it.planes():
        P = _sliceGrid(corners, *vis.shape)
        grid = pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2])
        grid.cell_data["rgb"] = _uint8(rgb.reshape(-1, 3, order="F"))
        p.add_mesh(grid, scalars="rgb", rgb=True, lighting=False)
    elif isinstance(it, picture3d):
      # the pixels are opaque or transparent, which depth peeling keeps from blending
      p.enable_depth_peeling()
      quad = pv.PolyData(it.corners, np.array([4, 0, 1, 3, 2]))
      quad.active_texture_coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
      tex = pv.Texture(_uint8(it.rgba)[::-1])
      tex.interpolate = True
      p.add_mesh(quad, texture=tex, lighting=False)
    elif isinstance(it, lines3d):
      k = len(it.segments)
      poly = pv.PolyData(it.segments.reshape(-1, 3), lines=np.column_stack([np.full(k, 2), 2 * np.arange(k), 2 * np.arange(k) + 1]).reshape(-1))
      poly.cell_data["rgb"] = _uint8(it.color)
      p.add_mesh(poly, scalars="rgb", rgb=True, line_width=it.lineWidth, opacity=it.alpha, lighting=False)
    elif isinstance(it, points3d):
      cloud = pv.PolyData(it.P)
      cloud.point_data["rgb"] = _uint8(it.color)
      p.add_mesh(cloud, scalars="rgb", rgb=True, point_size=it.markerSize * 1.5, render_points_as_spheres=True, lighting=False)
  _pvCamera(p, scene)
  return p


def _pvCamera(p, scene):
  o, up = scene.camera()
  lo, hi = scene.bounds()
  c = (lo + hi) / 2
  r = max(np.linalg.norm(hi - lo) / 2, 1e-12)
  p.enable_parallel_projection()
  p.camera_position = [tuple(c + 4 * r * o), tuple(c), tuple(up)]
  p.reset_camera()


def renderImage(scene, size=(800, 800)):
  """the scene drawn offscreen, an (h, w, 3) uint8 image"""
  p = pyvistaPlotter(scene, off_screen=True, window_size=size)
  img = p.screenshot(return_img=True)
  p.close()
  return img


# ------------------------------------------------------------------ matplotlib

def drawMatplotlib(scene, ax):
  """the scene on matplotlib's 3D axes, the faces sorted by depth as matplotlib does"""
  from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

  from .spaceplot import _camera, _setBox
  for it in scene.items:
    if isinstance(it, mesh3d):
      T, face = it.triangles()
      fc = np.column_stack([it.faceColor[face], np.full(len(face), it.faceAlpha)])
      pc = Poly3DCollection(it.V[T], facecolors=fc, linewidths=0)
      ax.add_collection3d(pc)
      if it.edgeColor is not None and it.edgeAlpha > 0:
        E = it.edges()
        ax.add_collection3d(Line3DCollection(it.V[E], colors=[(*it.edgeColor, it.edgeAlpha)], linewidths=it.lineWidth))
    elif isinstance(it, slices3d):
      for rgb, vis, corners in it.planes():
        P = _sliceGrid(corners, *vis.shape)
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=rgb, shade=False, rstride=1, cstride=1, linewidth=0)
    elif isinstance(it, picture3d):
      # a picture of at most 150 x 150 cells, the cells nothing is drawn in left out
      r, c = it.rgba.shape[:2]
      step = max(1, int(np.ceil(max(r, c) / 150)))
      rgba = it.rgba[::step, ::step]
      u, v = np.meshgrid(np.linspace(0, 1, rgba.shape[1] + 1), np.linspace(0, 1, rgba.shape[0] + 1))
      C = it.corners
      P = C[0] + u[..., None] * (C[1] - C[0]) + v[..., None] * (C[2] - C[0])
      ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], facecolors=rgba, shade=False, rstride=1, cstride=1, linewidth=0)
    elif isinstance(it, lines3d):
      ax.add_collection3d(Line3DCollection(it.segments, colors=np.column_stack([it.color, np.full(len(it.color), it.alpha)]), linewidths=it.lineWidth))
    elif isinstance(it, points3d):
      ax.scatter(it.P[:, 0], it.P[:, 1], it.P[:, 2], c=it.color, s=it.markerSize ** 2, depthshade=False)
  # a flat scene is given a thickness of a thousandth of its size
  lo, hi = scene.bounds()
  pad = np.where(hi - lo > 0, 0, 5e-4 * max(np.max(hi - lo), 1e-12))
  _setBox(ax, lo - pad, hi + pad)
  ax.set_axis_off()
  from ..geometry.plottingconvention import plottingConvention
  _camera(ax, plottingConvention.default3D if scene.how2plot is None else scene.how2plot)


# ------------------------------------------------------------------ HTML

def _blob(a):
  return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 6)).decode("ascii")


# the scene as the JSON the page reads: per mesh the vertices, the fan triangles, a colour
# per triangle and the polygon edges, per slice an RGBA picture and its rectangle, all
# binary arrays compressed and in base64, the vertices about the centre of the scene
def _payload(scene):
  lo, hi = scene.bounds()
  c = (lo + hi) / 2
  items = []
  for it in scene.items:
    if isinstance(it, mesh3d):
      T, face = it.triangles()
      d = {"kind": "mesh", "V": _blob((it.V - c).astype(np.float32)), "T": _blob(T.astype(np.uint32)),
           "C": _blob(_uint8(it.faceColor[face])), "faceAlpha": it.faceAlpha}
      if it.edgeColor is not None and it.edgeAlpha > 0:
        d.update(E=_blob(it.edges().astype(np.uint32)), edgeColor=list(map(float, it.edgeColor)), edgeAlpha=it.edgeAlpha)
      items.append(d)
    elif isinstance(it, slices3d):
      for rgb, vis, corners in it.planes():
        # the picture with the first index along its rows, as a texture wants it
        rgba = np.concatenate([_uint8(rgb), np.full(vis.shape + (1,), 255, dtype=np.uint8)], axis=-1)
        items.append({"kind": "slice", "rgba": _blob(np.transpose(rgba, (1, 0, 2))), "w": int(vis.shape[0]), "h": int(vis.shape[1]),
                      "corners": (corners - c).round(9).tolist()})
    elif isinstance(it, picture3d):
      items.append({"kind": "picture", "rgba": _blob(_uint8(it.rgba)), "w": int(it.rgba.shape[1]), "h": int(it.rgba.shape[0]),
                    "corners": (it.corners - c).round(9).tolist()})
    elif isinstance(it, lines3d):
      items.append({"kind": "lines", "P": _blob((it.segments.reshape(-1, 3) - c).astype(np.float32)),
                    "C": _blob(_uint8(np.repeat(it.color, 2, axis=0))), "alpha": it.alpha})
    elif isinstance(it, points3d):
      items.append({"kind": "points", "P": _blob((it.P - c).astype(np.float32)), "C": _blob(_uint8(it.color)), "size": it.markerSize})
  o, up = scene.camera()
  return {"items": items, "out": o.tolist(), "up": up.tolist(), "radius": float(max(np.linalg.norm(hi - lo) / 2, 1e-12))}


THREE = "https://cdn.jsdelivr.net/npm/three@0.170.0"

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
<style>html,body{margin:0;height:100%%;background:#fff;overflow:hidden}canvas{display:block;width:100%%;height:100%%;touch-action:none}
#hint{position:absolute;left:8px;bottom:6px;font:12px sans-serif;color:#888;pointer-events:none}</style>
<script type="importmap">{"imports":{"three":"%(three)s/build/three.module.js","three/addons/":"%(three)s/examples/jsm/"}}</script>
</head><body><div id="hint">drag to rotate, right drag to pan, wheel to zoom, double click to reset</div>
<script id="scene" type="application/json">%(scene)s</script>
<script type="module">
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
THREE.ColorManagement.enabled = false;
const S = JSON.parse(document.getElementById("scene").textContent);
async function inflate(b64, Type) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  const buf = await new Response(stream).arrayBuffer();
  return new Type(buf);
}
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(1, 1, 1);
for (const it of S.items) {
  if (it.kind === "mesh") {
    const V = await inflate(it.V, Float32Array), T = await inflate(it.T, Uint32Array), C = await inflate(it.C, Uint8Array);
    // a colour per triangle: every triangle its own three vertices
    const n = T.length / 3, P = new Float32Array(9 * n), K = new Uint8Array(9 * n);
    for (let t = 0; t < n; t++) for (let j = 0; j < 3; j++) {
      const v = T[3 * t + j];
      for (let k = 0; k < 3; k++) { P[9 * t + 3 * j + k] = V[3 * v + k]; K[9 * t + 3 * j + k] = C[3 * t + k]; }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(P, 3));
    g.setAttribute("color", new THREE.BufferAttribute(K, 3, true));
    const translucent = it.faceAlpha < 1;
    scene.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide,
      transparent: translucent, opacity: it.faceAlpha, depthWrite: !translucent,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })));
    if (it.E) {
      const E = await inflate(it.E, Uint32Array), ge = new THREE.BufferGeometry();
      ge.setAttribute("position", new THREE.BufferAttribute(V, 3));
      ge.setIndex(new THREE.BufferAttribute(E, 1));
      scene.add(new THREE.LineSegments(ge, new THREE.LineBasicMaterial({ color: new THREE.Color(...it.edgeColor),
        transparent: it.edgeAlpha < 1, opacity: it.edgeAlpha, depthWrite: false })));
    }
  } else if (it.kind === "lines") {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(await inflate(it.P, Float32Array), 3));
    g.setAttribute("color", new THREE.BufferAttribute(await inflate(it.C, Uint8Array), 3, true));
    scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: it.alpha < 1, opacity: it.alpha })));
  } else if (it.kind === "points") {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(await inflate(it.P, Float32Array), 3));
    g.setAttribute("color", new THREE.BufferAttribute(await inflate(it.C, Uint8Array), 3, true));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({ vertexColors: true, size: 1.5 * it.size * window.devicePixelRatio, sizeAttenuation: false })));
  } else if (it.kind === "slice" || it.kind === "picture") {
    const tex = new THREE.DataTexture(await inflate(it.rgba, Uint8Array), it.w, it.h, THREE.RGBAFormat);
    tex.magFilter = tex.minFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    const c = it.corners, g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([...c[0], ...c[1], ...c[2], ...c[3]]), 3));
    g.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2));
    g.setIndex([0, 1, 3, 0, 3, 2]);
    const pic = it.kind === "picture";
    if (pic) tex.magFilter = tex.minFilter = THREE.LinearFilter;
    scene.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: pic, alphaTest: pic ? 0.05 : 0 })));
  }
}
const r = S.radius, camera = new THREE.OrthographicCamera(-r, r, r, -r, -100 * r, 100 * r);
const controls = new TrackballControls(camera, renderer.domElement);
controls.rotateSpeed = 3;
function home() {
  camera.position.set(...S.out.map(x => 4 * r * x));
  camera.up.set(...S.up);
  camera.zoom = 1;
  controls.target.set(0, 0, 0);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
}
function resize() {
  const w = window.innerWidth, h = window.innerHeight, a = w / h, s = 1.05 * r;
  renderer.setSize(w, h, false);
  Object.assign(camera, a > 1 ? { left: -s * a, right: s * a, top: s, bottom: -s } : { left: -s, right: s, top: s / a, bottom: -s / a });
  camera.updateProjectionMatrix();
  controls.handleResize();
}
home(); resize();
window.addEventListener("resize", resize);
renderer.domElement.addEventListener("dblclick", home);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script></body></html>
"""


def sceneHTML(scene, title="MTEX 3D plot"):
  """the scene as a standalone HTML page drawn by three.js"""
  data = json.dumps(_payload(scene), separators=(",", ":")).replace("</", "<\\/")
  return _PAGE % {"title": title, "three": THREE, "scene": data}


def writeHTML(scene, fname, title="MTEX 3D plot"):
  """writeHTML(scene, 'fig.html'): the scene as an interactive page"""
  with open(fname, "w", encoding="utf-8") as f:
    f.write(sceneHTML(scene, title))
  return fname


__all__ = ["hasPyVista", "pyvistaPlotter", "renderImage", "drawMatplotlib", "writeHTML", "sceneHTML"]

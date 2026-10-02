# the analysis cases of bench/analysis_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/analysis_ops.py julia [op ...]
# The halfwidth selection of the suite has no counterpart in this port.
using MTEX, LinearAlgebra
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "analysis")

load(name, k) = permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), k, :))
rotationsOf(Q) = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]
vec3sOf(X) = [Vec3(X[i, 1], X[i, 2], X[i, 3]) for i in axes(X, 1)]
quats(q::AbstractArray{<:Rotation}) = [getfield(r, f) for r in vec(q), f in (:a, :b, :c, :d)]
quats(o) = quats(rotations(o))
vecs(v) = [getfield(w, f) for w in vec(v), f in (:x, :y, :z)]

ops, algo = crossargs(["gnd", "gbnd", "schmid", "birefringence", "components", "odf_max", "cluster", "classix", "cluster_directions",
                       "cluster_orientations", "optimal_s2", "optimal_so3"] .=> nothing, "analysis")

# the maps, read only for the cases that need them
dc06 = "gnd" in ops ? mtexdata("dc06") : nothing
twins, gB = if "gbnd" in ops
  tw = mtexdata("twins")
  grains = calcGrains!(tw; threshold = 5degree, minPixel = 3)
  tw, grains.boundary["Magnesium", "Magnesium"]
else
  nothing, nothing
end

# the bcc systems with the edge energy one and the screw energy 1 - nu, in 1/m² for a map in µm
function gnd()
  cs = orientations(dc06).frameA
  dS = [DislocationSystem(d.b, d.l, isEdge(d) ? 1.0 : 1 - 0.3) for d in dislocationSystems(:bcc, cs)]
  g = first(calcGND(dc06, dS))
  hcat(vecs(data(dc06.pos))[:, 1:2], 1e16 .* vec(g))
end

cub = crystalFrame("m-3m")
ori = Orientation(rotationsOf(load("O", 4)), cub)
sS = symmetrise(fcc(cub))
schmid() = (h = inv(ori) * xvector; [SchmidFactor(s, x) for x in vec(h), s in sS])

ol = crystalFrame("mmm", [4.756, 10.207, 5.980]; mineral = "Forsterite")
rI = 0.86 * RefractiveIndexTensor(Matrix(Diagonal([1.670, 1.635, 1.651])), ol) +
     0.14 * RefractiveIndexTensor(Matrix(Diagonal([1.879, 1.827, 1.869])), ol)
V = vec3sOf(load("V", 3))

# 300 random centres in m-3m, a 10 degree kernel each
odf = SO3FunRBF(Orientation(rotationsOf(load("R", 4)), cub), DeLaValleePoussinKernel(halfwidth = 10degree), vec(load("Rw", 1)))
components() = (r = calcComponents(odf); hcat(quats(r.modes), r.weights))
odfMax() = ((v, o) = findmax(odf); hcat(v, quats([o.rot])))

# the clusters as rows of their centre and their size
sized(centre, id) = hcat(centre, [count(==(k), id) for k in axes(centre, 1)])
X = load("X", 3)
classixCentres(id) = sized(permutedims(reduce(hcat, [vec(sum(X[id .== k, :]; dims = 1)) ./ count(==(k), id) for k in 1:maximum(id)])), id)
D = load("D", 3)
dirs = Vector3d([Direction(normalize(Vec3(D[j, 1], D[j, 2], D[j, 3]))) for j in axes(D, 1)], nothing)
Q = Orientation(rotationsOf(load("Q", 4)), crystalFrame("432"))

# the port has CLASSIX only: its default radius on the four clouds, not the hierarchical method
cluster() = calcCluster(Orientation(rotationsOf(load("C", 4)), crystalFrame("432"))).center

# 1 + 0.6 z in the coefficients of degree 0 and 1
density = S2FunHarmonic(ComplexF64[sqrt(4π), 0, 0.6sqrt(4π / 3), 0])
# a unimodal ODF of 20 degree without symmetry
unimodal = SO3FunRBF(Orientation([byEuler(10degree, 20degree, 30degree)], crystalFrame("1")), DeLaValleePoussinKernel(halfwidth = 20degree), [1.0])

table = [
  "gnd" => (gnd, identity),
  "gbnd" => (() -> calcGBND(gB, twins; halfwidth = 5degree, kernel = :nonneg), f -> real.(f(vec3sOf(load("G", 3))))),
  "schmid" => (schmid, identity),
  "birefringence" => (() -> [birefringence(rI, v).delta for v in V], identity),
  "components" => (components, identity),
  "odf_max" => (odfMax, identity),
  "cluster" => (cluster, quats),
  "classix" => (() -> classix(permutedims(X), 0.1; minPts = 2000), classixCentres),
  "cluster_directions" => (() -> calcCluster(dirs; radius = 0.1, minPoints = 1000), r -> sized(vecs(vec3.(data(r.center))), r.id)),
  "cluster_orientations" => (() -> calcCluster(Q; radius = 0.1, minPoints = 200), r -> sized(quats(r.center), r.id)),
  "optimal_s2" => (() -> optimalSample(density, 300; bandwidth = 16), r -> vecs(vec3.(r.nodes))),
  "optimal_so3" => (() -> optimalSample(unimodal, 200; bandwidth = 16), r -> quats(r.nodes)),
]

crossrun(OUT, algo, table, ops; reps = Dict(op => (op in ("schmid", "birefringence") ? 3 : 1) for op in first.(table)))

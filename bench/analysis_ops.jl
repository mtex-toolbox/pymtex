# the analysis cases of bench/analysis_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/analysis_ops.py julia [op ...]
using MTEX, LinearAlgebra
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "analysis")

load(name, k) = permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), k, :))
rotationsOf(Q) = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]
vec3sOf(X) = [Vec3(X[i, 1], X[i, 2], X[i, 3]) for i in axes(X, 1)]
quats(q::AbstractArray{<:Rotation}) = [getfield(r, f) for r in vec(q), f in (:a, :b, :c, :d)]
quats(o) = quats(rotations(o))
vecs(v) = [getfield(w, f) for w in vec(v), f in (:x, :y, :z)]

ops, algo = crossargs(["gnd", "gbnd", "schmid", "birefringence", "cluster", "optimal_s2"] .=> nothing, "analysis")

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

# the port has CLASSIX only: its default radius on the four clouds, not the hierarchical method
cluster() = calcCluster(Orientation(rotationsOf(load("C", 4)), crystalFrame("432"))).center

# 1 + 0.6 z in the coefficients of degree 0 and 1
density = S2FunHarmonic(ComplexF64[sqrt(4π), 0, 0.6sqrt(4π / 3), 0])

table = [
  "gnd" => (gnd, identity),
  "gbnd" => (() -> calcGBND(gB, twins; halfwidth = 5degree, kernel = :nonneg), f -> real.(f(vec3sOf(load("G", 3))))),
  "schmid" => (schmid, identity),
  "birefringence" => (() -> [birefringence(rI, v).delta for v in V], identity),
  "cluster" => (cluster, quats),
  "optimal_s2" => (() -> optimalSample(density, 300; bandwidth = 16), r -> vecs(vec3.(r.nodes))),
]

crossrun(OUT, algo, table, ops; reps = Dict("gnd" => 1, "gbnd" => 1, "cluster" => 1, "optimal_s2" => 1, "schmid" => 3, "birefringence" => 3))

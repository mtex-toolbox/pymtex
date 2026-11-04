# the cases of bench/smoothing_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/smoothing_ops.py julia [op ...]
using MTEX
using MTEX: Vec3, calcGrains!, smoothBoundary, simplifyBoundary
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "smoothing")

# rows (x, y, angle about z, phase), phase 0 notIndexed; this port's phase 1 is notIndexed
A = reshape(reinterpret(Float64, read(joinpath(OUT, "map.bin"))), 4, :)
pos = [Vec3{Float64}(A[1, k], A[2, k], 0) for k in axes(A, 2)]
rot = [Rotation(AxisAngle(Direction(zvector), A[3, k])) for k in axes(A, 2)]
e = EBSD(pos, rot, Int8.(A[4, :]) .+ Int8(1), PhaseList(crystalFrame("m-3m"; mineral = "test")))

# the segments of both boundaries between two grains, as rows (x1, y1, x2, y2)
inner(g) = vcat(([gB.V[gB.F[k, c]][i] for k in findall(k -> all(>(0), gB.grainId[k, :]), axes(gB.F, 1)),
  (c, i) in ((1, 1), (1, 2), (2, 1), (2, 2))] for gB in (g.boundary, g.innerBoundary))...)

s = Dict{String,Any}()
table = [
  "staircase" => (() -> (s["g"] = calcGrains!(copy(e); threshold = 5degree)), inner),
  "smooth" => (() -> smoothBoundary(s["g"]), inner),
  "smooth_noSimplify" => (() -> smoothBoundary(s["g"]; simplify = false), inner),
  "simplify_staircase" => (() -> simplifyBoundary(s["g"]), inner),
]

ops, algo = crossargs(table, "boundary smoothing")
crossrun(OUT, algo, table, ops; always = ("staircase",))

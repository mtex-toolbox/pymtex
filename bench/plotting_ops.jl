# the inverse pole figure colour cases of bench/plotting_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/plotting_ops.py julia [op ...]
using MTEX
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "plotting")
const P = MTEX.MTEXPlotting

Q = permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "O.bin"))), 4, :))
O = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]
colorRows(c) = [x[k] for x in vec(c), k in 1:3]

# through the key's grid as a map takes it, and exactly through the key, along the specimen z
table = reduce(vcat, [begin
  ori = Orientation(O, crystalFrame(sym))
  ["ipf_$s" => (() -> P.keyColors(ori.frameA, rotations(ori); direction = zvector, exact = false,
                                  resolution = P.COLOR_GRID_RESOLUTION), colorRows),
   "ipf_exact_$s" => (() -> color(IPFColorKey(ori.frameA), ori), colorRows)]
end for (s, sym) in ("m3m" => "m-3m", "6mmm" => "6/mmm")])

ops, algo = crossargs(table, "plotting")
crossrun(OUT, algo, table, ops; reps = 3)

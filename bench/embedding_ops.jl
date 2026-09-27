# the embedding cases of bench/embedding_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/embedding_ops.py julia [op ...]
using MTEX, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "embedding")
const NP = 10^4

load(name) = permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), 4, :))
rotationsOf(Q) = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]
quats(q::AbstractArray{<:Rotation}) = [getfield(r, f) for r in vec(q), f in (:a, :b, :c, :d)]
quats(q::Rotation) = quats([q])
quats(o) = quats(rotations(o))

A, B, C = (rotationsOf(load(x)) for x in ("O", "P", "C"))

table = Pair{String,Any}[]
for (s, sym) in (("m3m", "m-3m"), ("6mmm", "6/mmm"))
  cs = crystalFrame(sym)
  O, P, Cs = Orientation(A, cs), Orientation(B, cs), Orientation(C, cs)
  # the distance of every image to the first, over the norm of the identity's image
  embed() = norm.(embedding(O) .- embedding(O[1])) ./ norm(embedding(cs))
  # the seeds once for all midpoints: project builds its seed grid anew on every call
  function midpoints()
    seeds = vec(rotations(equispacedSO3Grid(cs; points = 20)))
    [project(e; seeds) for e in (embedding(O[1:NP]) .+ embedding(P[1:NP])) ./ 2]
  end
  push!(table,
    "embed_$s" => (embed, d -> d),
    "project_$s" => (midpoints, quats),
    "mean_$s" => (() -> project(mean(embedding(Cs))), quats))
end

ops, algo = crossargs(table, "orientation embedding")
crossrun(OUT, algo, table, ops; reps = Dict(op => startswith(op, "project") ? 1 : 3 for op in first.(table)))

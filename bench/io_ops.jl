# the import cases of bench/io_ops.py in the Julia port, on the same files
#   .venv/bin/python bench/io_ops.py julia [op ...]
using MTEX
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "io")
const DATA = normpath(joinpath(@__DIR__, "..", "..", "master", "data"))
const FILES = ["load_ctf" => joinpath(DATA, "EBSD", "Forsterite.ctf"), "load_ang" => joinpath(DATA, "EBSD", "ferrite.ang"),
               "load_osc" => joinpath(DATA, "EBSD", "olivineopticalmap.ang"), "load_1C_1" => "/home/hielscher/mtex/data/1C_1.ctf"]

# a map as rows (x, y, phase, a, b, c, d); the list starts with the not indexed phase, which is phase 0
function mapRows(e)
  p, r, ph = vec(data(e.pos)), vec(e.rot), vec(e.phaseId)
  hcat([q.x for q in p], [q.y for q in p], Float64.(ph .- 1), [getfield(q, f) for q in r, f in (:a, :b, :c, :d)])
end

# the pole figures as rows (x, y, z, intensity), one pole figure after the other
pfRows(pfs) = reduce(vcat, [hcat([getfield(w, f) for w in vec(vec3.(pf.r)), f in (:x, :y, :z)], vec(pf.intensities)) for pf in pfs])

# the seven files in the order of their names, as MATLAB's dir lists them
const DUBNA = sort(filter(endswith(".cnv"), readdir(joinpath(DATA, "PoleFigure", "dubna"); join = true)))
const QUARTZ = crystalFrame("321", [4.916, 4.916, 5.4054]; mineral = "Quartz")

table = [
  [op => (() -> loadEBSD(f; grid = false), mapRows) for (op, f) in FILES];
  "load_dubna" => (() -> loadPoleFigure(DUBNA, QUARTZ; superposition = [[1.0], [1.0], [0.52, 1.23], [1.0], [1.0], [1.0], [1.0]]), pfRows)
]

ops, algo = crossargs(table, "io")
crossrun(OUT, algo, table, ops; reps = Dict("load_1C_1" => 1))

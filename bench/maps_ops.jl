# the maps cases of bench/maps_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/maps_ops.py julia [op ...]
using MTEX
using MTEX: Vec3, calcGrains!, gridify, isIndexed, data, centroid, area, perimeter, aspectRatio,
  smoothBoundary, KAM, calcGROD
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "maps")
const LARGE = joinpath(@__DIR__, "out", "calcgrains")
const DENOISE = ["smooth_$(f)_$(m)" for m in ("twins", "ferrite") for f in ("mean", "spline", "l1tv", "hq", "fill")]
const ONCE = ("calcgrains_1C_1", "smooth_l1tv_twins", "smooth_l1tv_ferrite", "smooth_hq_twins", "smooth_hq_ferrite")
const FILTERS = Dict("mean" => () -> MeanFilter(; radius = 1), "spline" => () -> SplineFilter(),
  "l1tv" => () -> L1TVFilter(), "hq" => () -> HalfQuadraticFilter(), "fill" => () -> SplineFilter())

# the synthetic map of rows (x, y, angle about z); this port's phase 1 is notIndexed
A = reshape(reinterpret(Float64, read(joinpath(OUT, "map.bin"))), 3, :)
pos = [Vec3{Float64}(A[1, k], A[2, k], 0) for k in axes(A, 2)]
rot = [Rotation(AxisAngle(Direction(zvector), A[3, k])) for k in axes(A, 2)]
e = EBSD(pos, rot, fill(Int8(2), size(A, 2)), PhaseList(crystalFrame("m-3m"; mineral = "test")))

# the large map as bench/calcgrains.py wrote it, on its grid
function loadMap(name)
  A = reshape(reinterpret(Float64, read(joinpath(LARGE, "$name.bin"))), 7, :)
  frames = map(filter(!isempty, readlines(joinpath(LARGE, "$name.phases")))) do line
    mineral, pg, abc, abg, al = split(line, '\t')
    crystalFrame(String(pg), parse.(Float64, split(abc)), parse.(Float64, split(abg)) .* degree;
      mineral = String(mineral), alignment = isempty(al) ? String[] : String.(split(al, ',')))
  end
  pos = [Vec3{Float64}(A[1, k], A[2, k], 0) for k in axes(A, 2)]
  rot = [Rotation(A[4, k], A[5, k], A[6, k], A[7, k]) for k in axes(A, 2)]
  gridify(EBSD(pos, rot, Int8.(A[3, :]) .+ Int8(1), PhaseList(frames)))
end

# the measured map as the EBSDDenoising page prepares it
function measured(name)
  ebsd = mtexdata(name)
  calcGrains!(ebsd; threshold = 10degree, minPixel = 5)
  ebsd[isIndexed(ebsd)]
end

# rows keyed by the grain's centroid or the pixel's position; a smoothed grain by the one it came from
xy(m) = [p[i] for p in vec(data(m.pos)), i in 1:2]
byGrain(g, v) = hcat([c[i] for c in centroid(g), i in 1:2], vec(v))
byPixel(m, v) = hcat(xy(m), vec(v))
# a smoothed map by its positions and quaternions
quat(m) = hcat(xy(m), [getfield(r, f) for r in vec(m.rot), f in (:a, :b, :c, :d)])

s = Dict{String,Any}()
grains() = (s["g"] = calcGrains!(e; threshold = 5degree))
large = Ref{Any}()
maps = Dict{String,Any}()
denoise(op) = function ()
  _, f, m = split(op, "_")
  r = smooth(maps[m], FILTERS[f](); fill = f == "fill")
  r[isIndexed(r)]
end

table = [
  "calcgrains_1C_1" => (() -> calcGrains!(large[]; threshold = 10degree), g -> byGrain(g, area(g))),
  "calcgrains" => (grains, g -> byGrain(g, area(g))),
  "perimeter" => (() -> perimeter(s["g"]), v -> byGrain(s["g"], v)),
  "aspect" => (() -> aspectRatio(s["g"]), v -> byGrain(s["g"], v)),
  "smooth_boundary" => (() -> smoothBoundary(s["g"], 5), g -> byGrain(s["g"], perimeter(g))),
  "kam" => (() -> KAM(e), v -> byPixel(e, v)),
  "grod" => (() -> angle(calcGROD(e, s["g"])), v -> byPixel(e, v)),
  (op => (denoise(op), quat) for op in DENOISE)...,
]

ops, algo = crossargs(table, "maps")
"calcgrains_1C_1" in ops && (large[] = loadMap("1C_1"))
# the measured maps prepared outside the clock
for m in ("twins", "ferrite")
  any(op -> op in DENOISE && endswith(op, m), ops) && (maps[m] = measured(m))
end
crossrun(OUT, algo, table, ops; reps = Dict(op => 1 for op in ONCE), always = ("calcgrains",))

# the Julia side of bench/calcgrains.py: the maps it exported, the same cases, the same files out
#
#   julia -t auto --project=../mtexJulia/dev bench/calcgrains.jl [map ...]
using MTEX, Printf
using MTEX: Vec3, Rotation, calcGrains!, triplePoints, gridify, isIndexed, data

const OUT = joinpath(@__DIR__, "out", "calcgrains")
const MAPS = ["twins", "copper", "csl", "forsterite", "emsland", "synthetic"]
# the cases of bench/calcgrains.py: angle in degrees, minPixel, alpha, removeQuadruplePoints
const CASES = Dict("a" => (10, 1, 3.1, false), "b" => (5, 5, 3.1, false), "c" => (10, 1, 0.0, false),
  "d" => (10, 1, 3.1, true))
casesOf(m) = m in ("twins", "forsterite") ? "abcd" : m == "1C_1" ? "ac" : "abc"

# the map as bench/calcgrains.py wrote it, on its grid; this port's phase 1 is notIndexed
function loadMap(name)
  A = reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), 7, :)
  frames = map(filter(!isempty, readlines(joinpath(OUT, "$name.phases")))) do line
    mineral, pg, abc, abg, al = split(line, '\t')
    crystalFrame(String(pg), parse.(Float64, split(abc)), parse.(Float64, split(abg)) .* degree;
      mineral = String(mineral), alignment = isempty(al) ? String[] : String.(split(al, ',')))
  end
  pos = [Vec3{Float64}(A[1, k], A[2, k], 0) for k in axes(A, 2)]
  rot = [Rotation(A[4, k], A[5, k], A[6, k], A[7, k]) for k in axes(A, 2)]
  gridify(EBSD(pos, rot, Int8.(A[3, :]) .+ Int8(1), PhaseList(frames)))
end

function writeResult(name, case, t, g, e)
  ind = isIndexed(g)
  stats = [("time", t), ("grains", length(g)), ("indexed", count(ind)), ("notIndexed", count(!, ind)),
    ("boundary", length(g.boundary)), ("inner", length(g.innerBoundary)), ("triplePoints", length(triplePoints(g.boundary))),
    ("largest", any(ind) ? maximum(g.numPixel[ind]) : 0)]
  open(joinpath(OUT, "julia_$(name)_$case.txt"), "w") do io
    for (k, v) in stats
      println(io, k, " ", v)
    end
  end
  P = vec(data(e.pos))
  gid = vec(e.props[:grainId])
  rows = Float64[]
  for k in eachindex(P)
    isfinite(P[k][1]) || continue
    push!(rows, P[k][1], P[k][2], gid[k])
  end
  write(joinpath(OUT, "julia_$(name)_$case.bin"), rows)
end

function run(names)
  for name in names
    e0 = loadMap(name)
    for case in casesOf(name)
      angle, minPixel, alpha, rqp = CASES[string(case)]
      kw = (threshold = angle * degree, minPixel, alpha, removeQuadruplePoints = rqp)
      # calcGrains! writes into its map, so every call gets a copy made outside the clock
      calcGrains!(copy(e0); kw...)
      t, g, e = Inf, nothing, nothing
      for _ in 1:3
        ek = copy(e0)
        t0 = time()
        gk = calcGrains!(ek; kw...)
        dt = time() - t0
        dt < t && ((t, g, e) = (dt, gk, ek))
      end
      writeResult(name, case, t, g, e)
      @printf("%s %s: %.3f s, %d grains\n", name, case, t, length(g))
      flush(stdout)
    end
  end
end

@printf("julia %s, %d threads\n", VERSION, Threads.nthreads())
run(isempty(ARGS) ? MAPS : ARGS)

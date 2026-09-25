# the denoising cases of bench/maps_ops.py in the Julia port, times and results
#   julia -t auto --project=../mtexJulia/dev bench/maps_ops.jl [op ...]
using MTEX, Printf

const OUT = joinpath(@__DIR__, "out", "maps")
const FILTERS = Dict("mean" => () -> MeanFilter(; radius = 1), "spline" => () -> SplineFilter(),
  "l1tv" => () -> L1TVFilter(), "hq" => () -> HalfQuadraticFilter())

# the measured map as the EBSDDenoising page prepares it
function measured(name)
  ebsd = mtexdata(name)
  calcGrains!(ebsd; threshold = 10degree, minPixel = 5)
  ebsd[isIndexed(ebsd)]
end

function run(ops)
  maps = Dict{String,Any}()
  for op in ops
    _, f, m = split(op, "_")
    e = get!(() -> measured(m), maps, m)
    F = FILTERS[f]()
    s = smooth(e, F)
    t = minimum(@elapsed(smooth(e, F)) for _ in 1:3)
    s = s[isIndexed(s)]
    P = vec(data(s.pos))
    r = s.rot
    rows = Float64[]
    for k in eachindex(P)
      push!(rows, P[k][1], P[k][2], r.a[k], r.b[k], r.c[k], r.d[k])
    end
    write(joinpath(OUT, "julia_$op.bin"), rows)
    write(joinpath(OUT, "julia_$op.txt"), "$t\n")
    @printf("%-22s %9.1f ms\n", op, 1e3t)
    flush(stdout)
  end
end

run(isempty(ARGS) ? ["smooth_$(f)_$(m)" for m in ("twins", "ferrite") for f in ("mean", "spline", "l1tv", "hq")] : ARGS)

# the Julia side of bench/maps.py: the same map, the same verbs, best of three after a warm up
#
#   julia -t auto --project=../mtexJulia/dev bench/maps.jl [n] [grains]
using MTEX, Printf, Statistics
using MTEX: Vec3, calcGrains!, simplifyBoundary, refineBoundary, reduceBoundary, smoothBoundary,
  CurvatureFilter, aspectRatio, hull, paris, caliper, paror, surfor, characteristicShape,
  gradient, curvature, KAM, calcGROD, grainMean, calcMisorientation, calcGND, dislocationSystems,
  smooth, MeanFilter, MedianFilter, KuwaharaFilter, SplineFilter, L1TVFilter

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 600
const NG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 800
const HERE = @__DIR__
const CS = crystalFrame("m-3m", mineral = "test")

# the angles bench/maps.py wrote, or a small map of blocks for the warm up
function angles(n, ng)
  f = joinpath(HERE, "maps_$(n)_$(ng).bin")
  isfile(f) && return reinterpret(Float64, read(f))
  [((i ÷ 8) + (j ÷ 8) * 7) * 20 * pi / 180 + 0.002 * i for j in 0:n-1 for i in 0:n-1]
end

function mapOf(n, ang)
  pos = [Vec3{Float64}(i, j, 0) for j in 0:n-1 for i in 0:n-1]
  rot = [Rotation(AxisAngle(Direction(zvector), w)) for w in ang]
  EBSD(pos, rot, fill(Int8(2), n * n), PhaseList(CS))
end

function best(f, k = 3)
  t = Inf
  out = nothing
  for _ in 1:k
    t0 = time()
    out = f()
    t = min(t, time() - t0)
  end
  t, out
end

function run(n, ng, report)
  ang = angles(n, ng)
  rows = Tuple{String,Float64,String}[]
  t, g = best(() -> calcGrains!(mapOf(n, ang); threshold = 5 * pi / 180))
  push!(rows, ("calcGrains", t, "$(length(g)) grains, $(length(g.boundary)) segments"))
  e = mapOf(n, ang)
  g = calcGrains!(e; threshold = 5 * pi / 180)
  kam = KAM(e)
  for (name, f) in [("simplifyBoundary", () -> simplifyBoundary(g)), ("refineBoundary", () -> refineBoundary(g)),
    ("reduceBoundary", () -> reduceBoundary(g, 2)), ("smoothBoundary laplace 5", () -> smoothBoundary(g, 5)),
    ("smoothBoundary curvature", () -> smoothBoundary(g, CurvatureFilter())),
    ("aspectRatio", () -> aspectRatio(g)), ("hull", () -> hull(g)), ("paris", () -> paris(g)),
    ("caliper", () -> caliper(g)), ("caliper shortest", () -> caliper(g, :shortest)), ("paror", () -> paror(g)),
    ("surfor", () -> surfor(g)), ("characteristicShape", () -> characteristicShape(g)),
    ("gradient oneSided", () -> gradient(e)), ("gradient full", () -> gradient(e; stencil = :full)),
    ("curvature", () -> curvature(e)), ("KAM", () -> KAM(e)), ("KAM order 3", () -> KAM(e; order = 3)),
    ("calcGROD", () -> calcGROD(e, g)), ("grainMean", () -> grainMean(e, kam, g)),
    ("calcMisorientation 1e5", () -> calcMisorientation(e; sampleSize = 100000)),
    ("smooth mean", () -> smooth(e, MeanFilter())), ("smooth median", () -> smooth(e, MedianFilter())),
    ("smooth Kuwahara", () -> smooth(e, KuwaharaFilter())), ("smooth spline", () -> smooth(e, SplineFilter())),
    ("smooth l1TV", () -> smooth(e, L1TVFilter()))]
    t, = best(f)
    push!(rows, (name, t, ""))
  end
  dS = dislocationSystems(:fcc, CS)
  sub = e[1:1600]
  t, out = best(() -> calcGND(sub, dS), 1)
  push!(rows, ("calcGND 1600 px", t, "$(count(isfinite, out[1])) finite"))
  if report
    @printf("julia %s, %d threads, map %d x %d, %d seeds\n", VERSION, Threads.nthreads(), n, n, ng)
    println("| verb | s | note |\n|---|---:|---|")
    for (name, t, note) in rows
      @printf("| %s | %.3f | %s |\n", name, t, note)
    end
  end
end

run(56, 8, false)
run(N, NG, true)

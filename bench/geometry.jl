# the Julia twin of bench/geometry.py: run with julia --project=../mtexJulia/dev bench/geometry.jl
using MTEX, LinearAlgebra, Random, Printf, Statistics
BLAS.set_num_threads(1)
const G = MTEX.MTEXGeometry
const N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1_000_000
cs, hx = crystalFrame("m-3m"), crystalFrame("6/mmm")
rng = MersenneTwister(0)
r, s = rand(rng, Rotation{Float64}, N), rand(rng, Rotation{Float64}, N)
v = rand(rng, Direction{Float64}, N)
ori, ori2 = rand(rng, Orientation, (N,), cs), rand(rng, Orientation, (N,), cs)
ref = ori[1]
mori, mori2 = rand(rng, Orientation, (N,), cs, hx), rand(rng, Orientation, (N,), cs, hx)
mref = mori[1]
m = rand(rng, Miller, (N,), cs)

# the projection against a list of references, one per element, composed from the batched
# pair kernel since the API takes a single reference
function project2refs(o, refs)
  L, R = G.reductionGroups(o)
  qs = vec(rotations(o))
  _, li, rj, usedInv = G.batchedSymReduce(vec(rotations(refs)), qs, L, R)
  out = similar(qs)
  for i in eachindex(qs)
    p = L[li[i]] * (usedInv[i] ? inv(qs[i]) : qs[i]) * R[rj[i]]
    out[i] = dot(p, rotations(refs)[i]) < 0 ? Rotation(-p.a, -p.b, -p.c, -p.d) : p
  end
  Orientation(out, o.frameA, o.frameB)
end

toRef(p, q) = mean(abs.(dot.(rotations(p), rotations(q))))
zmean(w) = mean(abs.(getfield.(w, :z)))
best(f) = (f(); minimum(@elapsed(f()) for _ in 1:5))
rows = [
  ("r * s", () -> r .* s, q -> mean(abs.(getfield.(q, :a)))),
  ("r * v", () -> r .* v, zmean),
  ("dot(ori, ori2)", () -> dot(ori, ori2), mean),
  ("dot(ori, ref)", () -> dot(ori, ref), mean),
  ("dot(mori, mori2)", () -> dot(mori, mori2), mean),
  ("dot(mori, mref)", () -> dot(mori, mref), mean),
  ("project2FR(m)", () -> project2FundamentalRegion(m), w -> mean(getfield.(vec3.(w), :z))),
  ("project2FR(ori, ref)", () -> project2FundamentalRegion(ori, ref), p -> toRef(p, ref)),
  ("project2FR(ori, ori2)", () -> project2refs(ori, ori2), p -> toRef(p, ori2)),
  ("project2FR(mori, mref)", () -> project2FundamentalRegion(mori, mref), p -> toRef(p, mref)),
  ("project2FR(mori, mori2)", () -> project2refs(mori, mori2), p -> toRef(p, mori2)),
]
println("Julia, N = $N, m-3m and m-3m -> 6/mmm, best of 5, threads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads())")
for (name, f, check) in rows
  out = f()
  t = best(f)
  @printf("%-26s %9.1f ms %8.1f ns/elem   check %.4f\n", name, 1e3t, 1e9t / N, check(out))
end

# the Julia twin of bench/gridsearch.py: run with julia --project=../mtexJulia/dev bench/gridsearch.jl
using MTEX, Random, Printf, Statistics, SparseArrays
res = 2.5degree; eps = 7.5degree; nn = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 10000; nr = 2000
cs = crystalFrame("m-3m")
rng = MersenneTwister(0)
t = time(); g = equispacedSO3Grid(cs; resolution = res); tbuild = time() - t
o = rand(rng, Orientation, (nn,), cs, defaultFrame())
nearest(g, o[1:5]); neighbours(g, o[1:5], eps)
best(f, reps = 3) = minimum((f(); t = time(); f(); time() - t) for _ in 1:reps)
tn = best(() -> nearest(g, o))
tr = best(() -> neighbours(g, o[1:nr], eps))
M = neighbours(g, o[1:nr], eps)
println("Julia, cubic m-3m $(res/degree) deg grid")
@printf("  build            %8.0f ms   (%d points)\n", 1e3tbuild, length(g))
@printf("  nearest %6d   %8.1f ms   %6.1f us/query\n", nn, 1e3tn, 1e6tn/nn)
@printf("  radius  %6d   %8.1f ms   %6.1f us/query   mean %.1f nb @ %.1f deg\n", nr, 1e3tr, 1e6tr/nr, nnz(M)/nr, eps/degree)

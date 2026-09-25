# the cases of bench/harmonics_user.py in the Julia port, one section per argument
# run: julia --project=../mtexJulia/dev -t 10 bench/harmonics_user.jl eval adjoint grid sym s2
using MTEX, Printf, Random

rng = MersenneTwister(0)
timed(f, reps = 3) = (f(); minimum(@elapsed(f()) for _ in 1:reps))
report(name, t) = @printf("%-48s %9.1f ms\n", name, 1000t)
want(tag) = isempty(ARGS) || tag in ARGS

randomODF(L, cs, ss) = SO3FunHarmonic(randn(rng, ComplexF64, deg2dim(L + 1)), cs, ss; real = true)
tri, cub, spec = crystalFrame("1"), crystalFrame("m-3m"), specimenFrame("1")

if want("eval")
  for (L, J) in ((64, 10^6), (128, 10^4), (128, 10^6)), (name, cs) in (("1", tri), ("m-3m", cub))
    F = randomODF(L, cs, spec)
    o = Orientation(rand(rng, Rotation, J), cs, spec)
    report(@sprintf("SO3 eval L=%d J=%.0e %s", L, J, name), timed(() -> F(o), 5))
  end
end

if want("adjoint")
  for (L, J) in ((64, 10^6), (128, 10^4), (128, 10^6)), (name, cs) in (("1", tri), ("m-3m", cub))
    o = Orientation(rand(rng, Rotation, J), cs, spec)
    v = rand(rng, J)
    report(@sprintf("SO3 quadrature L=%d J=%.0e %s", L, J, name), timed(() -> SO3FunHarmonic(o, v; bandwidth = L), 5))
  end
end

if want("grid")
  for (name, cs, ss) in (("1/1", tri, spec), ("432/1", crystalFrame("432"), spec), ("622/222", crystalFrame("622"), specimenFrame("222")))
    L = bandwidthFor(127, cs, ss)
    F = randomODF(L, cs, ss)
    g = quadratureSO3Grid(L, cs, ss)
    report(@sprintf("SO3 eval Clenshaw-Curtis L=%d %s", L, name), timed(() -> F(g), 2))
    v = F(g)
    report(@sprintf("SO3 quadrature Clenshaw-Curtis L=%d %s", L, name),
           timed(() -> SO3FunHarmonic(wignerSeriesAdjoint(g, v, L), cs, ss; real = true), 2))
  end
end

if want("sym")
  for (name, cs, ss) in (("622/222", crystalFrame("622"), specimenFrame("222")), ("432/mmm", crystalFrame("432"), specimenFrame("mmm")),
                         ("432/1", crystalFrame("432"), spec))
    fhat = randn(rng, ComplexF64, deg2dim(129))
    report("SO3 symmetrise L=128 $name", timed(() -> SO3FunHarmonic(fhat, cs, ss)))
  end
end

if want("s2")
  L = 128
  sF = S2FunHarmonic(randn(rng, ComplexF64, (L + 1)^2); real = true)
  v = [rand(rng, Vec3{Float64}) for _ in 1:10^6]
  report("S2 eval L=128 J=1e6", timed(() -> sF(v), 2))
  vals = rand(rng, 10^6)
  report("S2 quadrature L=128 J=1e6", timed(() -> S2FunHarmonic(v, vals; bandwidth = L), 2))
  for L in (256, 512, 1024)
    g = quadratureS2Grid(L)
    sG = S2FunHarmonic(randn(rng, ComplexF64, (L + 1)^2))
    report("S2 eval Clenshaw-Curtis L=$L", timed(() -> sG(g), 2))
  end
  sC = S2FunHarmonic(randn(rng, ComplexF64, 129^2), cub; symmetrise = false)
  report("S2 symmetrise m-3m L=128", timed(() -> MTEX.symmetrise(sC)))
  w = [rand(rng, Vec3{Float64}) for _ in 1:10^5]
  report("S2 calcDensity 1e5", timed(() -> calcDensity(w), 2))
end

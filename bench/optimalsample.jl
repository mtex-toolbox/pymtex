# the Julia optimalSample on the cases of bench/optimalsample.py, for docs/research/benchmark-sampling.md
#
#   (cd ../mtexJulia && julia -t auto --project=dev ../mtexPython/bench/optimalsample.jl)
using MTEX, LinearAlgebra, Printf
const out = joinpath(@__DIR__, "out")
mkpath(out)

function targets()
  fs = S2FunHarmonic(ComplexF64[sqrt(4π), 0, 0.6sqrt(4π / 3), 0])
  c = zeros(ComplexF64, 10); c[1] = 1; c[6] = 0.6 / sqrt(3)
  fr = SO3FunHarmonic(c)
  cs = crystalFrame("222")
  fibre = SO3FunHarmonic(0.5 * uniformODF(cs) + 0.5 * fibreODF(Miller(0, 0, 1, cs), zvector; halfwidth = 20degree); bandwidth = 32)
  Dict("S2 1+0.6z" => fs, "SO3 1+0.6(Rz).z" => fr, "SO3 222 fibre" => fibre)
end

const CASES = [("S2 1+0.6z", 32, 100, 20, false), ("S2 1+0.6z", 32, 100, 20, true), ("S2 1+0.6z", 128, 1000, 20, false),
               ("SO3 1+0.6(Rz).z", 16, 100, 20, false), ("SO3 1+0.6(Rz).z", 16, 100, 20, true),
               ("SO3 222 fibre", 16, 500, 20, false), ("SO3 222 fibre", 32, 500, 20, false), ("SO3 222 fibre", 32, 2000, 50, false),
               ("SO3 222 fibre", 64, 500, 20, false)]

function main()
  fs = targets()
  @printf("julia %s, %d threads, %d BLAS threads\n", VERSION, Threads.nthreads(), BLAS.get_num_threads())
  println("target,L,M,steps,weights,seconds,initial,final,iterations")
  open(joinpath(out, "optimalsample_julia.csv"), "w") do fh
    for (name, L, M, steps, opt) in CASES
      f = fs[name]
      solve() = optimalSample(f, M; bandwidth = L, maxIter = steps, tol = 0, tolWeights = 0, tolJ = 0, warmUp = 5, optimizeWeights = opt)
      solve()
      best, r = Inf, nothing
      for _ in 1:3
        GC.gc()
        t = @elapsed r = solve()
        best = min(best, t)
      end
      @printf("%s,%d,%d,%d,%d,%.4f,%.6g,%.6g,%d\n", name, L, M, steps, Int(opt), best, r.history[1], r.discrepancy, r.iterations)
      flush(stdout)
      @printf(fh, "%s,%d,%d,%d,%d,%.4f,%.6g,%.6g,%d\n", name, L, M, steps, Int(opt), best, r.history[1], r.discrepancy, r.iterations)
    end
  end
end

main()

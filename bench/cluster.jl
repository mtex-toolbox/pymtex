# the Julia classix and calcCluster on the point sets of bench/cluster.py, for docs/research/benchmark-sampling.md
#
#   (cd ../mtexJulia && julia -t auto --project=dev ../mtexPython/bench/cluster.jl)
using MTEX, LinearAlgebra, DelimitedFiles, Printf
const out = joinpath(@__DIR__, "out")

function timed(fun)
  fun()
  best, r = Inf, nothing
  for _ in 1:3
    GC.gc()
    t = @elapsed r = fun()
    best = min(best, t)
  end
  best, r
end

function main()
  @printf("julia %s, %d threads, %d BLAS threads\n", VERSION, Threads.nthreads(), BLAS.get_num_threads())
  println("case,n,seconds,clusters")
  open(joinpath(out, "cluster_julia.csv"), "w") do fh
    for n in (10_000, 100_000, 1_000_000)
      X = readdlm(joinpath(out, "cluster_points_$(n).csv"), ',')
      Xt = permutedims(X)
      t, labels = timed(() -> MTEX.classix(Xt, 0.1; minPts = n ÷ 100))
      @printf("classix,%d,%.4f,%d\n", n, t, maximum(labels)); flush(stdout)
      @printf(fh, "classix,%d,%.4f,%d\n", n, t, maximum(labels))
      if n <= 100_000
        v = Vector3d([Direction(normalize(Vec3(X[j, 1], X[j, 2], X[j, 3]))) for j in axes(X, 1)], nothing)
        t, r = timed(() -> calcCluster(v))
        @printf("calcCluster directions,%d,%.4f,%d\n", n, t, length(r.center)); flush(stdout)
        @printf(fh, "calcCluster directions,%d,%.4f,%d\n", n, t, length(r.center))
      end
    end
    cs = crystalFrame("432")
    for n in (2_000, 20_000)
      Q = readdlm(joinpath(out, "cluster_quats_$(n).csv"), ',')
      o = Orientation([Rotation(Q[j, 1], Q[j, 2], Q[j, 3], Q[j, 4]) for j in axes(Q, 1)], cs)
      t, r = timed(() -> calcCluster(o; radius = 0.1, minPoints = n ÷ 100))
      @printf("calcCluster orientations,%d,%.4f,%d\n", n, t, length(r.center)); flush(stdout)
      @printf(fh, "calcCluster orientations,%d,%.4f,%d\n", n, t, length(r.center))
    end
  end
end

main()

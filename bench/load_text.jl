# the Julia loaders on the same files, for docs/research/benchmark-io.md
#
#   (cd ../mtexJulia && julia -t auto --project=dev ../pymtex/bench/load_text.jl)
using MTEX, Printf
files = ["/home/hielscher/mtex/frameSymmetry/data/EBSD/Forsterite.ctf", "/home/hielscher/mtex/frameSymmetry/data/EBSD/ferrite.ang",
         "/home/hielscher/mtex/frameSymmetry/data/EBSD/olivineopticalmap.ang", "/home/hielscher/mtex/data/1C_1.ctf"]
@printf("julia %s, %d threads\n", VERSION, Threads.nthreads())
for f in files
  loadEBSD(f); loadEBSD(f; grid = false)
end
for f in files
  t1 = @elapsed e = loadEBSD(f; grid = false)
  t2 = @elapsed g = loadEBSD(f)
  @printf("%-28s %9d rows  list %.3f s  grid %.3f s\n", basename(f), length(e), t1, t2)
end

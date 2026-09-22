# the Julia inverse pole figure colouring and map plot on the maps of bench/mapplot.py, for docs/research/benchmark-mapplot.md
#
#   (cd ../mtexJulia && julia -t auto --project=dev ../mtexPython/bench/mapplot.jl)
using MTEX, CairoMakie, LinearAlgebra, Printf
import CairoMakie: save
const out = joinpath(@__DIR__, "out")
const D = "/home/hielscher/mtex/frameSymmetry/data/EBSD"

function timed(fun, repeat = 3)
  fun()
  best, r = Inf, nothing
  for _ in 1:repeat
    GC.gc()
    t = @elapsed r = fun()
    best = min(best, t)
  end
  best, r
end

function main()
  @printf("julia %s, %d threads, %d BLAS threads\n", VERSION, Threads.nthreads(), BLAS.get_num_threads())
  println("map,n,colors grid,colors exact,key exact,plot,save")
  files = [("Forsterite", joinpath(D, "Forsterite.ctf")), ("Emsland", joinpath(D, "Emsland_plessite_500x_15.ctf")),
           ("Forsterite 2x2", joinpath(out, "forsterite_2x2.ctf")), ("Forsterite 4x4", joinpath(out, "forsterite_4x4.ctf"))]
  open(joinpath(out, "mapplot_julia.csv"), "w") do fh
    for (name, f) in files
      ebsd = loadEBSD(f)
      n = length(ebsd)
      rep = n > 2_000_000 ? 1 : 3
      tGrid, _ = timed(() -> ipfColors(ebsd))
      tExact, _ = timed(() -> ipfColors(ebsd; exact = true), rep)
      pl = phases(ebsd)
      ids = vec(ebsd.phaseId)
      big = argmax([MTEX.isIndexed(pl, i) ? count(==(Int8(i)), ids) : -1 for i in eachindex(pl.frames)])
      ori = orientations(ebsd[ids .== Int8(big)])
      key = IPFColorKey(pl[big])
      tKey, _ = timed(() -> MTEX.color(key, ori), rep)
      fig = nothing
      tPlot, fig = timed(() -> plot(ebsd; color = :orientation), rep)
      png = joinpath(out, "mapplot_julia_$(replace(name, ' ' => '_')).png")
      tSave, _ = timed(() -> save(png, fig; px_per_unit = 1), rep)
      @printf("%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", name, n, tGrid, tExact, tKey, tPlot, tSave); flush(stdout)
      @printf(fh, "%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", name, n, tGrid, tExact, tKey, tPlot, tSave)
    end
  end
end

main()

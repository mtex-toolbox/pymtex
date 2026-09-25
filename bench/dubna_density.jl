# odf = SO3Fun.dubna; ori = odf.discreteSample(1e7); odf2 = calcDensity(ori) in the Julia port:
# the Dubna ODF as MATLAB's mtexdata('dubnaODF') makes it, calcODF of the seven pole figures
# run: julia --project=../mtexJulia/dev -t 10 bench/dubna_density.jl
using MTEX, Printf, Random

function step(name, f)
  t = time()
  out = f()
  @printf("%-28s %8.2f s\n", name, time() - t)
  out
end

cs = crystalFrame("321", [4.916, 4.916, 5.4054])
dubna() = calcODF(loadPoleFigure(mtexdatafile("dubna"), cs; superposition = [nothing, nothing, [0.52, 1.23], nothing, nothing, nothing, nothing]))
# the first call compiles; the second is the one timed
step("compile", () -> (o = dubna(); calcDensity(discreteSample(MersenneTwister(1), o, 1000))))
odf = step("odf = dubna", dubna)
ori = step("discreteSample(1e7)", () -> discreteSample(MersenneTwister(0), odf, 10^7))
odf2 = step("calcDensity(ori)", () -> calcDensity(ori))
println(typeof(odf).name.name, " ", typeof(odf2).name.name, " bandwidth ", MTEX.bandwidth(odf2))

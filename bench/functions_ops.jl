# the functions cases of bench/functions_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/functions_ops.py julia [op ...]
using MTEX
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "functions")

load(name, k = 1) = k == 1 ? reinterpret(Float64, read(joinpath(OUT, "$name.bin"))) |> collect :
  permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), k, :))
rotationsOf(Q) = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]

cs, tri = crystalFrame("m-3m"), crystalFrame("1")
O, V = rotationsOf(load("O", 4)), load("V", 3)
v = [Vec3(V[i, 1], V[i, 2], V[i, 3]) for i in axes(V, 1)]
fO, fV = load("fO"), load("fV")
odf = SO3FunRBF(Orientation(rotationsOf(load("C", 4)), cs), DeLaValleePoussinKernel(halfwidth = 10degree), load("w"))
ori, oriT = Orientation(O, cs), Orientation(O, tri)
# the weights MATLAB gives unweighted nodes: 1 / N on SO(3), 4π / N on the sphere
wO, wV = fill(1 / length(O), length(O)), fill(4π / length(v), length(v))

# the phase MATLAB's basis differs by: (-1)^k for an odd positive order k
phase(k) = k > 0 && isodd(k) ? -1 : 1
# the Wigner coefficients in MATLAB's order and phase: each degree block transposed, both orders phased
function matlabOrder(f::MTEX.SO3FunHarmonic)
  out, s = similar(f.fhat), 0
  for n in 0:MTEX.bandwidth(f)
    B = reshape(view(f.fhat, s+1:s+(2n+1)^2), 2n + 1, 2n + 1)
    out[s+1:s+(2n+1)^2] = [phase(k) * phase(l) * B[l+n+1, k+n+1] for k in -n:n, l in -n:n]
    s += (2n + 1)^2
  end
  out
end
# the spherical coefficients in MATLAB's phase
matlabOrder(f::MTEX.S2FunHarmonic) = [phase(m) * f.fhat[l*l+m+l+1] for l in 0:MTEX.bandwidth(f) for m in -l:l]
# complex coefficients as rows (real, imaginary)
coefficients(F) = (c = matlabOrder(F); hcat(real.(c), imag.(c)))
col(x) = real.(vec(x))

F, G = Ref{Any}(), Ref{Any}()
table = [
  "rbf_harmonic" => (() -> (F[] = SO3FunHarmonic(odf; bandwidth = 64)), coefficients),
  "rbf_direct" => (() -> odf(ori[1:10^4]), col),
  "so3_eval" => (() -> F[](ori), col),
  "so3_quadrature" => (() -> SO3FunHarmonic(oriT, fO; bandwidth = 32, weights = wO), coefficients),
  "s2_quadrature" => (() -> (G[] = S2FunHarmonic(v, fV; bandwidth = 128, weights = wV)), coefficients),
  "s2_eval" => (() -> G[](v), col),
  "so3_interpolate" => (() -> interpolate(oriT[1:20000], fO[1:20000]; bandwidth = 16), H -> col(H(oriT[1:10^4]))),
  "s2_interpolate" => (() -> interpolate(v[1:20000], fV[1:20000]; bandwidth = 48), H -> col(H(v[1:10^4]))),
]

ops, algo = crossargs(table, "functions")
crossrun(OUT, algo, table, ops; reps = 3, always = ("rbf_harmonic", "s2_quadrature"))
# the slow variant: the harmonic form against the kernel sum, in this port
@printf("harmonic form against the kernel sum: %.1e\n", maximum(abs.(real.(F[](ori[1:10^4])) .- odf(ori[1:10^4]))))

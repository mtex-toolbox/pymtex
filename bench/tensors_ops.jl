# the tensor cases of bench/tensors_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/tensors_ops.py julia [op ...]
using MTEX, LinearAlgebra
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "tensors")
const NT = 10^4
const OLIVINE = [320.5 68.15 71.6 0 0 0; 68.15 196.5 76.8 0 0 0; 71.6 76.8 233.5 0 0 0;
                 0 0 0 64 0 0; 0 0 0 0 77 0; 0 0 0 0 0 78.7]
const DENSITY = 3.355

load(name, k) = permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), k, :))

# the 81 coordinates of every tensor, column-major as MATLAB writes them
coordinates(t::Tensor) = permutedims(vec(Array(t.M)))
coordinates(ts::AbstractVector{<:Tensor}) = reduce(vcat, coordinates.(ts))

ol = crystalFrame("mmm", [4.7646, 10.2296, 5.9942]; mineral = "Olivine")
C = StiffnessTensor(; voigt = OLIVINE, frame = ol)
Q = load("O", 4)
rots = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]
V = load("V", 3)
v = [Vec3(V[i, 1], V[i, 2], V[i, 3]) for i in axes(V, 1)]
ori = Orientation(rots, ol)
odf = unimodalODF(Orientation(Rotation(Bunge(10degree, 20degree, 30degree)), ol); halfwidth = 15degree)
cub = crystalFrame("m-3m")
sS = symmetrise(MTEX.slipSystems(:fcc, cub))
eps = Tensor{2,Strain}(diagm([1.0, -0.5, -0.5]))
oriT = Orientation(rots[1:NT], cub)

R = Ref{Any}()
rotated() = (R[] = [o * C for o in ori])

# vp, vs1, vs2 of every direction, one row each
function speeds()
  w = [velocity(C, n; density = DENSITY) for n in v]
  [getfield(x, k) for x in w, k in (:vp, :vs1, :vs2)]
end
# the compliance once, then the modulus of every direction
youngs() = (S = inv(C); [YoungsModulus(S, n) for n in v])

table = [
  "rotate" => (rotated, coordinates),
  "inv" => (() -> inv.(R[]), coordinates),
  "average_ori" => (() -> calcTensor(ori, C; average = :Hill), coordinates),
  "average_odf" => (() -> calcTensor(odf, C), coordinates),
  "velocity" => (speeds, identity),
  "youngs" => (youngs, identity),
  "taylor" => (() -> calcTaylor(eps, oriT, sS).M, vec),
  "taylor_fun" => (() -> calcTaylor(eps, sS).M, F -> real.(vec(F(oriT)))),
]

# three runs as in Python, one for the velocities, which take half a minute each
reps = Dict(op => op == "velocity" ? 1 : 3 for op in first.(table))
ops, algo = crossargs(table, "tensors")
crossrun(OUT, algo, table, ops; reps, always = ("rotate",))

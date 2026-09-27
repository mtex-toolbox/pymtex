# the geometry cases of bench/geometry_ops.py in the Julia port, on the same input
#   .venv/bin/python bench/geometry_ops.py julia [op ...]
using MTEX, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "tools", "graph", "crosscheck.jl"))

const OUT = joinpath(@__DIR__, "out", "geometry")
const K, NQ, NR = 20000, 10^4, 2000

load(name, k) = permutedims(reshape(reinterpret(Float64, read(joinpath(OUT, "$name.bin"))), k, :))
rotationsOf(Q) = [Rotation(Q[i, 1], Q[i, 2], Q[i, 3], Q[i, 4]) for i in axes(Q, 1)]
quats(q::AbstractArray{<:Rotation}) = [getfield(r, f) for r in vec(q), f in (:a, :b, :c, :d)]
quats(o) = quats(rotations(o))
vecs(v) = [getfield(w, f) for w in vec(vec3.(v)), f in (:x, :y, :z)]

cs, hx = crystalFrame("m-3m"), crystalFrame("6/mmm")
A, B, C, D = (rotationsOf(load(x, 4)) for x in ("A", "B", "C", "D"))
V = load("V", 3)
v = [Direction(V[i, 1], V[i, 2], V[i, 3]) for i in axes(V, 1)]
W = load("W", 3)
m = Miller([Vec3(W[i, 1], W[i, 2], W[i, 3]) for i in axes(W, 1)], cs)
ori, ori2 = Orientation(A, cs), Orientation(B, cs)
mori, mori2 = Orientation(C, cs, hx), Orientation(D, cs, hx)
ref, mref = ori[1], mori[1]

# the projection against a list of references, one per element, as bench/geometry.jl composes it
const G = MTEX.MTEXGeometry
function project2refs(o, refs)
  L, R = G.reductionGroups(o)
  qs, rs = vec(rotations(o)), vec(rotations(refs))
  _, li, rj, usedInv = G.batchedSymReduce(rs, qs, L, R)
  out = similar(qs)
  for i in eachindex(qs)
    p = L[li[i]] * (usedInv[i] ? inv(qs[i]) : qs[i]) * R[rj[i]]
    out[i] = dot(p, rs[i]) < 0 ? -p : p
  end
  Orientation(out, o.frameA, o.frameB)
end

grid = Ref{Any}()
build() = (grid[] = equispacedSO3Grid(cs; resolution = 2.5degree))

# find returns the cosine of half the distance in Python; the angle is compared, as there
table = [
  "r_s" => (() -> A .* B, q -> quats(q)[1:K, :]),
  "r_v" => (() -> A .* v, w -> vecs(w)[1:K, :]),
  "dot_ori_ori2" => (() -> dot(ori, ori2), d -> d[1:K]),
  "dot_ori_ref" => (() -> dot(ori, ref), d -> d[1:K]),
  "dot_mori_mori2" => (() -> dot(mori, mori2), d -> d[1:K]),
  "dot_mori_mref" => (() -> dot(mori, mref), d -> d[1:K]),
  "p2fr_m" => (() -> project2FundamentalRegion(m), w -> vecs(w)[1:K, :]),
  "p2fr_ori_ref" => (() -> project2FundamentalRegion(ori, ref), o -> quats(o)[1:K, :]),
  "p2fr_ori_ori2" => (() -> project2refs(ori, ori2), o -> quats(o)[1:K, :]),
  "p2fr_mori_mref" => (() -> project2FundamentalRegion(mori, mref), o -> quats(o)[1:K, :]),
  "p2fr_mori_mori2" => (() -> project2refs(mori, mori2), o -> quats(o)[1:K, :]),
  "grid_build" => (build, g -> [Float64(length(g))]),
  "grid_nearest" => (() -> nearest(grid[], ori[1:NQ]), r -> angle(ori[1:NQ], grid[][r])),
  "grid_radius" => (() -> neighbours(grid[], ori[1:NR], 7.5degree), M -> Float64.(vec(sum(M .!= 0; dims = 1)))),
]

ops, algo = crossargs(table, "geometry")
crossrun(OUT, algo, table, ops; always = ("grid_build",))

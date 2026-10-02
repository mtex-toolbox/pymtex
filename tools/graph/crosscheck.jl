# the Julia half of crosscheck.py: each operation timed, profiled and its results written
#
# An operation writes to <out>: julia_<op>.txt (the best time), julia_<op>.bin (its results as
# float64 rows, the first K) and profile_julia_<op>.json in the record format of profiling.py,
# which tools/graph/build.py maps onto the jl: nodes by file and line. The profile samples the
# call repeated for at least a second; a record's times are per call, summed over the threads,
# and only samples that pass through the port's sources or the benchmark count, so idle threads
# are left out. Code of other packages inlined into the port's is the port's time.
using Profile, Printf

const PORT_SOURCES = (normpath(joinpath(@__DIR__, "..", "..", "..", "mtexJulia")), normpath(joinpath(@__DIR__, "..", "..", "bench")))

# ----
# results as rows

# float64 rows of a vector (one column) or a matrix (one row per result), the first k
rows(x::AbstractVector, k = typemax(Int)) = Float64.(x[1:min(end, k)])
rows(x::AbstractMatrix, k = typemax(Int)) = vec(permutedims(Float64.(x[1:min(end, k), :])))

# ----
# the profile

# the record key of a stack frame: its method's file and first line, else the frame's own
function frameKey(fr)
  li = fr.linfo
  li isa Core.CodeInstance && (li = li.def)
  li isa Core.MethodInstance && li.def isa Method && return (string(li.def.file), Int(li.def.line), string(li.def.name))
  (string(fr.file), fr.line, string(fr.func))
end

ownFile(file) = any(startswith(file, p) for p in PORT_SOURCES)

# the call run repeatedly for `seconds`, its samples as graph records, times per call
function profileCall(f; seconds = 1.0)
  Profile.clear()
  n, t0 = 0, time()
  Profile.@profile while n == 0 || time() - t0 < seconds
    f()
    n += 1
  end
  data, lidict = Profile.retrieve(; include_meta = false)
  index, recs = Dict{Tuple{String,Int,String},Int}(), Dict{Symbol,Any}[]
  children = Dict{Int,Dict{Int,Int}}()
  slot(k) = get!(index, k) do
    push!(recs, Dict(:file => isabspath(k[1]) ? k[1] : "~" * k[1], :line => k[2], :name => k[3], :self => 0, :total => 0, :calls => 0))
    length(recs)
  end
  stack = Int[]
  for ip in data
    if ip != 0
      for fr in lidict[ip]
        fr.from_c || fr.inlined && !ownFile(string(fr.file)) || push!(stack, slot(frameKey(fr)))
      end
      continue
    end
    # a sample ends: the leaf first; it counts when it passes through the port's sources
    if !isempty(stack) && any(i -> ownFile(recs[i][:file]), stack)
      recs[stack[1]][:self] += 1
      for i in unique(stack)
        recs[i][:total] += 1
      end
      seen = Set{Tuple{Int,Int}}()
      for j in 2:length(stack)
        p, c = stack[j], stack[j - 1]
        p != c && (p, c) ∉ seen || continue
        push!(seen, (p, c))
        d = get!(Dict{Int,Int}, children, p)
        d[c] = get(d, c, 0) + 1
      end
    end
    empty!(stack)
  end
  dt = Profile.init()[2] / n
  for (i, r) in enumerate(recs)
    r[:self] *= dt
    r[:total] *= dt
    r[:children] = [Any[c - 1, k * dt] for (c, k) in get(children, i, Dict{Int,Int}())]
  end
  recs
end

# ----
# the CPU time of the process, every thread's counted

const CLOCK_PROCESS_CPUTIME_ID = Sys.isapple() ? Int32(12) : Int32(2)

function cpuTime()
  ts = Ref((0, 0))
  ccall(:clock_gettime, Cint, (Int32, Ptr{Tuple{Int,Int}}), CLOCK_PROCESS_CPUTIME_ID, ts)
  ts[][1] + 1e-9 * ts[][2]
end

# ----
# the JSON of a profile, written without a package

json(x::AbstractString) = "\"" * escape_string(x) * "\""
json(x::Real) = isfinite(x) ? string(x) : "null"
json(x::Symbol) = json(string(x))
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::AbstractDict) = "{" * join((json(k) * ":" * json(v) for (k, v) in x), ",") * "}"

# cores: the CPU time of the timed runs over their wall time
function writeProfile(path, algorithm, case, wall, recs; cores = nothing, threads = Threads.nthreads())
  mkpath(dirname(path))
  d = Dict{Symbol,Any}(:algorithm => algorithm, :port => "julia", :case => case, :wall => wall, :threads => threads, :functions => recs)
  cores === nothing || (d[:cores] = cores)
  write(path, json(d))
end

# ----
# the run

# the ops of the command line and the algorithm of every case: crosscheck.py passes the ops asked
# for, "--", then "case=algorithm" for every case; without it all of the table's run under `name`
function crossargs(table, name)
  cases = first.(table)
  k = findfirst(==("--"), ARGS)
  k === nothing && return (isempty(ARGS) ? cases : ARGS), Dict(c => name for c in cases)
  ops = ARGS[1:k - 1]
  isempty(ops) ? cases : ops, Dict(Pair(split(a, "="; limit = 2)...) for a in ARGS[k + 1:end])
end

# table: a vector of op => (call, result -> rows); ops run in table order, the ones not asked for only when a
# later one needs them (always); reps a number or one per op, timed after one warm up, since a
# first call compiles; algo maps each op to the name of its algorithm
function crossrun(out, algo, table, ops; reps = 5, always = ())
  mkpath(out)
  for (op, (f, result)) in table
    op in ops || op in always || continue
    n = reps isa AbstractDict ? get(reps, op, 5) : reps
    f()
    best, res, cpu, wall = Inf, nothing, 0.0, 0.0
    for _ in 1:n
      c = cpuTime()
      t = @elapsed r = f()
      cpu += cpuTime() - c
      wall += t
      t < best && ((best, res) = (t, r))
    end
    cores = cpu / wall
    writeProfile(joinpath(out, "profile_julia_$op.json"), algo[op], op, best, profileCall(f; seconds = max(1.0, best)); cores)
    write(joinpath(out, "julia_$op.txt"), "$best\n")
    write(joinpath(out, "julia_$op.bin"), rows(result(res)))
    @printf("%-20s %9.1f ms %5.1f cores\n", op, 1e3best, cores)
    flush(stdout)
  end
end

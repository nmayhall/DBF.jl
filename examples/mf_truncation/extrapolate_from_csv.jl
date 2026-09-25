# =============================================================================
#  V->0 energy extrapolations from saved trajectories (see run_trajectories.jl).
#  Post-processing only — reads logs/<tag>_*.csv and applies DBF.extrapolate_energy
#  (linear/quadratic fit of corrected E vs corrected V over the best low-variance
#  window). No calculation is re-run.
#
#  Run (from DBF/examples):  julia --project=. mf_truncation/extrapolate_from_csv.jl <tag>
# =============================================================================

using DBF
using Printf

function read_csv(path)
    meta = Dict{String,String}()
    energy=Float64[]; variance=Float64[]; acc_err=Float64[]; acc_var=Float64[]
    for (li, line) in enumerate(eachline(path))
        if startswith(line, "#")
            for tok in split(strip(line[2:end]))
                kv = split(tok, "="); length(kv)==2 && (meta[kv[1]] = kv[2])
            end
        elseif li >= 3 && !isempty(strip(line))
            c = split(line, ",")
            push!(energy, parse(Float64, c[2])); push!(variance, parse(Float64, c[3]))
            push!(acc_err, parse(Float64, c[4])); push!(acc_var, parse(Float64, c[5]))
        end
    end
    return meta, energy, variance, acc_err, acc_var
end

# Simple [V,2V] extrapolation: linear intercept over the points whose corrected
# variance lies in [Vlow, 2*Vlow] (Vlow = lowest positive corrected variance).
# Returns (intercept b at V=0, npts in window, standard error of the intercept).
function extrap_v2v(cv, ce)
    idx = sortperm(cv); x = cv[idx]; y = ce[idx]
    pos = x .> 0; x = x[pos]; y = y[pos]
    length(x) >= 2 || return (NaN, 0, NaN)
    Vlow = x[1]
    win = findall(v -> Vlow <= v <= 2*Vlow, x)
    length(win) >= 2 || (win = collect(1:min(3, length(x))))
    xs = x[win]; ys = y[win]; n = length(xs)
    sx=sum(xs); sy=sum(ys); sxx=sum(abs2, xs); sxy=sum(xs.*ys)
    m = (n*sxy - sx*sy) / (n*sxx - sx^2)
    b = (sy - m*sx) / n
    se = NaN
    if n >= 3
        xbar = sx/n
        Sxx  = sxx - sx^2/n                       # Σ(x-x̄)²
        s2   = sum(abs2, ys .- (m .* xs .+ b)) / (n-2)   # residual variance
        se   = sqrt(s2 * (1/n + xbar^2/Sxx))     # SE of the intercept
    end
    return (b, n, se)
end

function main(tag)
    logdir = joinpath(@__DIR__, "logs")
    files = sort(filter(f -> startswith(f, tag*"_") && endswith(f, ".csv"), readdir(logdir)))
    isempty(files) && error("no CSVs matching $(tag)_*.csv — run run_trajectories.jl first")

    eexact = nothing
    ev() = eexact === nothing ? NaN : parse(Float64, eexact)
    ovl = "?"
    @printf("%-20s %13s %11s %13s %14s\n",
            "strategy", "E0[V,2V]", "SE(E0)", "err corr", "err/qubit")
    for f in files
        meta, E, V, ae, av = read_csv(joinpath(logdir, f))
        eexact = get(meta, "eexact", eexact); ovl = get(meta, "overlap", ovl)
        Nq = parse(Float64, get(meta, "N", "1"))
        eC, _, se = extrap_v2v(V .- av, E .- ae)     # corrected variance, corrected energy
        errC = abs(eC - ev())
        @printf("%-20s %13.5f %11.2e %13.2e %14.2e\n", get(meta,"label",f), eC, se, errC, errC/Nq)
    end
    eexact === nothing || @printf("exact GS = %s   |<HF|GS>|^2 = %s\n", eexact, ovl)
end

length(ARGS) >= 1 || error("usage: extrapolate_from_csv.jl <tag>")
main(ARGS[1])

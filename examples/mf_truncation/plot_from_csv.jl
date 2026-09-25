# =============================================================================
#  Plot E-vs-V trajectories from saved CSVs (see run_trajectories.jl).
#  Reads logs/<tag>_*.csv — does NOT run any calculation, so the plot can be
#  edited/regenerated freely without touching the jobs.
#
#  Run (from DBF/examples):  julia --project=. mf_truncation/plot_from_csv.jl <tag> [--corrected|--bare]
# =============================================================================

using Plots
using Printf

function read_csv(path)
    meta = Dict{String,String}()
    igrad=Int[]; energy=Float64[]; variance=Float64[]; acc_err=Float64[]; acc_var=Float64[]
    for (li, line) in enumerate(eachline(path))
        if startswith(line, "#")
            for tok in split(strip(line[2:end]))
                kv = split(tok, "="); length(kv)==2 && (meta[kv[1]] = kv[2])
            end
        elseif li >= 3 && !isempty(strip(line))   # skip the column header (line 2)
            c = split(line, ",")
            push!(igrad, parse(Int, c[1])); push!(energy, parse(Float64, c[2]))
            push!(variance, parse(Float64, c[3])); push!(acc_err, parse(Float64, c[4]))
            push!(acc_var, parse(Float64, c[5]))
        end
    end
    return meta, (; igrad, energy, variance, acc_err, acc_var)
end

function main(tag, corrected)
    logdir = joinpath(@__DIR__, "logs")
    files = sort(filter(f -> startswith(f, tag*"_") && endswith(f, ".csv"), readdir(logdir)))
    isempty(files) && error("no CSVs matching $(tag)_*.csv in $logdir — run run_trajectories.jl first")

    palette = Dict("Coeff"=>:seagreen, "Weight"=>:orange, "MeanField"=>:crimson)
    colorfor(label) = get(palette, first(split(label, "(")), :steelblue)

    Eexact = nothing
    kind = corrected ? "corrected" : "bare"
    plt = plot(xlabel="$kind variance", ylabel="$kind energy  E",
               title="E vs V by truncation  ($tag, $kind)",
               legend=:topleft, framestyle=:box, size=(780,540), dpi=150)

    for f in files
        meta, d = read_csv(joinpath(logdir, f))
        Eexact = get(meta, "eexact", nothing)
        label = get(meta, "label", f)
        V = corrected ? d.variance .- d.acc_var : d.variance
        E = corrected ? d.energy   .- d.acc_err : d.energy
        nt = get(meta, "nterms", "?")
        plot!(plt, V, E, m=:circle, ms=3, lw=1.5, c=colorfor(label), label="$label (n=$nt)")
    end
    if Eexact !== nothing
        ev = parse(Float64, Eexact)
        hline!(plt, [ev], ls=:dash, lc=:black, lw=1.5, label="exact GS = $(round(ev,digits=4))")
        # zoom on the low-variance tail (the extrapolation region) by default
        if !("--full" in ARGS)
            xlims!(plt, -0.15, 0.6)
            ylims!(plt, ev - 0.40, ev + 0.55)
        end
    end
    out = joinpath(logdir, "$(tag)_evsv_$(kind)$("--full" in ARGS ? "_full" : "_zoom").png")
    savefig(plt, out); println("saved: ", out)
end

length(ARGS) >= 1 || error("usage: plot_from_csv.jl <tag> [--bare|--corrected]")
tag = ARGS[1]
corrected = !("--bare" in ARGS)      # corrected by default; pass --bare for raw
main(tag, corrected)

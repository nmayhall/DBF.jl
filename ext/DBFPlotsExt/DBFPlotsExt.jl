module DBFPlotsExt

using DBF
using Plots

"""
    plot_extrapolation(out::Dict; kwargs...)

Plot energy vs variance with linear and quadratic extrapolation to zero variance.

Takes the output dictionary from `dbf_groundstate` and produces a two-panel plot:
- **Left panel**: Full trajectory with fit lines and extrapolated point.
- **Right panel**: Zoomed view of only the data used in the fit, with fit lines.

# Keywords
- `use_per_grad::Bool=false`: use per-macro-iteration data instead of per-rotation
- `min_points::Int=5`: minimum number of data points for fitting
- `r2_thresh::Float64=0.8`: minimum R² for the linear fit
- `verbose::Int=1`: verbosity level
- Additional keyword arguments are passed to `Plots.plot`.
"""
function DBF.plot_extrapolation(out::Dict;
    use_per_grad::Bool=false,
    min_points::Int=5,
    r2_thresh::Float64=0.8,
    verbose::Int=1,
    kwargs...)

    result = DBF.extrapolate_energy(out;
        use_per_grad=use_per_grad,
        min_points=min_points,
        r2_thresh=r2_thresh,
        verbose=verbose)

    ce = result.corrected_energies
    cv = result.corrected_variances
    cutoff = result.cutoff

    e_str = string(round(result.energy, sigdigits=8))
    u_str = string(round(result.uncertainty, sigdigits=2))
    r2_str = string(round(result.r2_linear, digits=4))

    # ── Left panel: full trajectory ──────────────────────────────────────
    p1 = plot(; xlabel="Corrected Variance", ylabel="Corrected Energy",
                title="Full Trajectory", legend=:topright)

    # Excluded points in gray
    if cutoff > 1
        scatter!(p1, cv[1:cutoff-1], ce[1:cutoff-1];
            label="excluded", color=:lightgray,
            markerstrokecolor=:lightgray, markersize=3)
    end

    # Fit window points
    scatter!(p1, cv[cutoff:end], ce[cutoff:end];
        label="fit data", color=:blue, markersize=4)

    # Fit lines from v=0 to max variance in window
    v_max = maximum(cv[cutoff:end])
    v_range = range(0, v_max, length=200)
    plot!(p1, v_range, result.fit_linear.(v_range);
        label="linear", color=:red, linewidth=2)
    plot!(p1, v_range, result.fit_quadratic.(v_range);
        label="quadratic", color=:green, linewidth=2, linestyle=:dash)

    # ── Right panel: fit window zoom ─────────────────────────────────────
    x_fit = cv[cutoff:end]
    y_fit = ce[cutoff:end]

    p2 = plot(; xlabel="Corrected Variance", ylabel="Corrected Energy",
                title="Fit Window ($(length(x_fit)) pts, R²=$r2_str)")

    scatter!(p2, x_fit, y_fit;
        label="fit data", color=:blue, markersize=4)

    v_fit_range = range(0, maximum(x_fit), length=200)
    plot!(p2, v_fit_range, result.fit_linear.(v_fit_range);
        label="linear", color=:red, linewidth=2)
    plot!(p2, v_fit_range, result.fit_quadratic.(v_fit_range);
        label="quadratic", color=:green, linewidth=2, linestyle=:dash)

    # Extrapolated point with error bar
    scatter!(p2, [0.0], [result.energy];
        label="E = $e_str ± $u_str",
        color=:red, markersize=6, yerror=[result.uncertainty])

    # ── Combine into two-panel layout ────────────────────────────────────
    p = plot(p1, p2; layout=(1, 2), size=(1200, 500), kwargs...)

    return p
end

end # module

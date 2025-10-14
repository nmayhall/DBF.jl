using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test
using Plots



function run(; U=U, threshold=1e-3, wmax=nothing, wtype=0)
    # Parameters for Hubbard model
    Lx = 2
    Ly = 2
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    t = 0.10
    H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)
    #H = DBF.fermi_hubbard_2D_snake(Lx, Ly, t, U; snake_ordering=true)
    #H = DBF.hubbard_model_1D(Nsites, t, U)

    println(" Original H:")
    display(H)
    
    Hmat = Matrix(H)
    evals = eigvals(Hmat)
    groundE = minimum(evals)

    #ψ = Ket{N}(0)
    #ψ = Ket([i%2 for i in 1:N])
    kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)
    display(ψ)
    e0 = expectation_value(H,ψ)
   
    @printf(" E0 = %12.8f\n", e0)

    display(norm(H))
    display(norm(diag(H)))
    H, dbfEs, nterms, loss = dbf_groundstate_test(H, ψ, max_iter=200, conv_thresh=1e-6, 
                                evolve_coeff_thresh=threshold,
                                evolve_weight_thresh=wmax, w_type=wtype,
                                search_n_top=10000)
    
    println(" New H:")
    #display(norm(H))
    #display(norm(diag(H)))
    println("Exact Ground State Energy: ", groundE)
    
    e1 = expectation_value(H,ψ)
    error = abs(e1-groundE)
    @printf(" E1 = %12.8f\n", e1)
    @printf("Error |E1-E0| = %12.8f\n", error)

    # DBE estimates
    steps = collect(1:length(dbfEs))
    println(" Number of steps: ", length(steps))
    plt1 = plot(steps, dbfEs,lw=2, color=:black, 
           label="th=$(threshold) , w=$(wmax === nothing ? "none" : string(wmax))")
    plot!(plt1, [1,length(steps)], [groundE, groundE], ls=:dash, lw=2, color=:red, 
    label="Exact Ground State Energy")

    xlabel!(plt1, "DBF Step")
    ylabel!(plt1, "DBF Energy Estimate (a.u.)")

    fname_w = (wmax === nothing) ? "All" : string(wmax)
    wname = (wtype == 0) ? "Pauli" : "Majorana"
    savefig(plt1, "Energies-N=$(N)_th=$(threshold)_w=$(fname_w)_type=$(wname).pdf")

    # Number of terms in Hamiltonian
    plt2 = plot(steps, nterms,lw=2, color=:black, 
           label="th=$(threshold) , w=$(wmax === nothing ? "none" : string(wmax))")
    xlabel!(plt2, "DBF Step")
    ylabel!(plt2, "Number of Terms")
    savefig(plt2, "N-Terms_N=$(N)_th=$(threshold)_w=$(fname_w)_type=$(wname).pdf")   

    # Loss
    plt3 = plot(steps, loss,lw=2, color=:green, 
           label="th=$(threshold) , w=$(wmax === nothing ? "none" : string(wmax))")
    xlabel!(plt3, "DBF Step")
    ylabel!(plt3, "Loss (1 - HS-norm^2)")
    savefig(plt3, "Loss_N=$(N)_th=$(threshold)_w=$(fname_w)_type=$(wname).pdf") 

    return error, dbfEs, nterms, loss, groundE

end


#= 
   Test set performing comparisons of the quality of the DBF-OPT ground state solver for the Hubbard model.
   Here we forculs solely on the 2D Hubbard model on a 2x2 lattice. which can be solved exactly.
   
   - Error comparison and performance with comparison to exact diagonalization
   - Comparison of different weight and coeff thresholding pruning strategies
   - How accurate is this approach with differen choices of the parameters?

   Comparisons in different coupling regimes:
      Weak coupling regime: t=0.1, U=0.001
      Middle coupling regime: t=0.1, U=0.09
      Strong coupling regime: t=0.1, U=0.5   
=#
#run(U=0.001, threshold=1e-3, wmax=4, wtype=1)

#us = [0.001, 0.09, 0.5]
us = [0.5]#, 0.5, 1.0, 1.5]
threshs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
Pweights = [2, 3, 4, 5, 6, 7, 8]
Mweights = [2, 3, 4, 5, 6, 7, 8]

Pauli_errors = Dict{String,Float64}()
Majorana_errors = Dict{String,Float64}()

absolute_errors = Float64[]
absolute_errors_Majorana = Float64[]
dbfEs_list = Vector{Float64}[]
nterms_list = Vector{Int}[]
loss_list = Vector{Float64}[]

for U in us
    println("========================================")
    println(" U = ", U)
    println("========================================")
    println("---- Coefficient Thresholding Only ----")
    for thresh in threshs
        err, dbfEs, nterms, loss, groundE = run(U=U, threshold=thresh, wmax=nothing, wtype=0)

        println("dbfEs type: ", typeof(dbfEs), " length: ", length(dbfEs))
        println(" nterms type: ", typeof(nterms), " length: ", (isa(nterms, AbstractArray) ? length(nterms) : 1))
        println(" loss  type: ", typeof(loss), " length: ", (isa(loss, AbstractArray) ? length(loss) : 1))
        println(" err   type: ", typeof(err))
        println(" groundE: ", groundE)

        
        push!(absolute_errors, err)
        push!(dbfEs_list, dbfEs)
        push!(nterms_list, nterms)
        push!(loss_list, loss)
        @printf(" Error with coeff thresholding only (th=%1.1e): %1.5e\n", thresh, err)

    end
    # Overlay plots
    plt = plot()
    plt1 = plot()
    plt2 = plot()
    for (i, dbfEs) in enumerate(dbfEs_list)
        steps = collect(1:length(dbfEs))
        plot!(plt, steps, dbfEs, lw=2, label="th=$(threshs[i])")#, markershape=:circle)
        plot!(plt1, steps, nterms_list[i], lw=2, label="th=$(threshs[i])")#, markershape=:circle)
        plot!(plt2, steps, loss_list[i], lw=2, label="th=$(threshs[i])")#, markershape=:circle)
    end
    
    
    xlabel!(plt, "DBF Step")
    ylabel!(plt, "DBF Energy Estimate (a.u.)")
    savefig(plt, "Energies_U=$(U)_th=varied_w=None.pdf")

    xlabel!(plt1, "DBF Step")
    ylabel!(plt1, "Number of Terms")
    savefig(plt1, "N-Terms_U=$(U)_th=varied_w=None.pdf")

    xlabel!(plt2, "DBF Step")
    ylabel!(plt2, "Loss (1 - HS-norm^2)")
    savefig(plt2, "Loss_U=$(U)_th=varied_w=None.pdf")

   #- Clean data lists for next round
    empty!(absolute_errors)
    empty!(dbfEs_list)
    empty!(nterms_list)
    empty!(loss_list)   

    println("---- Coefficient + Pauli Weight Thresholding ----")
    labels_list  = String[]

    for thresh in threshs
        for wmax in Pweights
            # run returns errP, dbfEs, nterms, loss, groundE
            errP, dbfEs, nterms, loss, groundE = run(U=U, threshold=thresh, wmax=wmax, wtype=0)

            # push per-run lists (if you need them for overlay plotting)
            push!(absolute_errors, errP)
            push!(dbfEs_list, dbfEs)
            push!(nterms_list, nterms)
            push!(loss_list, loss)
            push!(labels_list, "th=$(thresh), w=$(wmax)")

            # save summary into persistent dict (label -> error)
            lbl = "th=$(thresh), w=$(wmax)"
            Pauli_errors[lbl] = errP
            @printf(" Error with coeff (th=%1.1e) + Pauli weight (w=%d): %1.5e\n", thresh, wmax, errP)
        end
        # Overlay plots (unchanged logic, using dbfEs_list, nterms_list, loss_list)
        plt = plot() 
        plt1 = plot()
        plt2 = plot()
        for (i, dbfEs) in enumerate(dbfEs_list)
            steps = collect(1:length(dbfEs))
            lbl = labels_list[i]
            plot!(plt, steps, dbfEs, lw=2, label=lbl)
            plot!(plt1, steps, nterms_list[i], lw=2, label=lbl)
            plot!(plt2, steps, loss_list[i], lw=2, label=lbl)
        end

        xlabel!(plt, "DBF Step"); ylabel!(plt, "DBF Energy Estimate (a.u.)")
        savefig(plt, "Energies_U=$(U)_th=varied_w=Pauli.pdf")

        xlabel!(plt1, "DBF Step"); ylabel!(plt1, "Number of Terms")
        savefig(plt1, "N-Terms_U=$(U)_th=varied_w=Pauli.pdf")

        xlabel!(plt2, "DBF Step"); ylabel!(plt2, "Loss (1 - HS-norm^2)")
        savefig(plt2, "Loss_U=$(U)_th=varied_w=Pauli.pdf")

        #- Clean transient data lists for next threshold round (keeps Pauli_errors dict)
        empty!(absolute_errors)
        empty!(dbfEs_list)
        empty!(nterms_list)
        empty!(loss_list)
    end

    println("---- Coefficient + Majorana Weight Thresholding ----")
    empty!(labels_list)

    for thresh in threshs
        for wmax in Mweights
            errM, dbfEs, nterms, loss, groundE = run(U=U, threshold=thresh, wmax=wmax, wtype=1)
            push!(absolute_errors_Majorana, errM)
            push!(dbfEs_list, dbfEs)
            push!(nterms_list, nterms)
            push!(loss_list, loss)
            push!(labels_list, "th=$(thresh), w=$(wmax)")

            lbl = "th=$(thresh), w=$(wmax)"
            Majorana_errors[lbl] = errM

            @printf(" Error with coeff (th=%1.1e) + Majorana weight (w=%d): %1.5e\n", thresh, wmax, errM)
        end

        # Overlay plots for Majorana (keep file names different to avoid overwriting)
        plt = plot()
        plt1 = plot()
        plt2 = plot()

        for (i, dbfEs) in enumerate(dbfEs_list)
            steps = collect(1:length(dbfEs))
            lbl = labels_list[i]
            # NOTE: using Mweights / threshs for labels could be more accurate here;
            # keep the pattern similar to the Pauli block.
            plot!(plt, steps, dbfEs, lw=2, label=lbl)
            plot!(plt1, steps, nterms_list[i], lw=2, label=lbl)
            plot!(plt2, steps, loss_list[i], lw=2, label=lbl)
        end

        xlabel!(plt, "DBF Step"); ylabel!(plt, "DBF Energy Estimate (a.u.)")
        savefig(plt, "Energies_U=$(U)_th=varied_w=Majorana.pdf")

        xlabel!(plt1, "DBF Step"); ylabel!(plt1, "Number of Terms")
        savefig(plt1, "N-Terms_U=$(U)_th=varied_w=Majorana.pdf")

        xlabel!(plt2, "DBF Step"); ylabel!(plt2, "Loss (1 - HS-norm^2)")
        savefig(plt2, "Loss_U=$(U)_th=varied_w=Majorana.pdf")

    #- Clean transient lists (persisted summaries remain in Majorana_errors)
        empty!(absolute_errors_Majorana)
        empty!(dbfEs_list)
        empty!(nterms_list)
        empty!(loss_list)
    end

    # --- Collect all unique labels ---
    all_labels = collect(union(keys(Pauli_errors), keys(Majorana_errors)))

    # --- Define a helper to extract numeric th and w from each label safely ---
    function parse_th_w(label::String)
        m = match(r"th=([0-9.eE+-]+), w=([0-9]+)", label)
        if m !== nothing
            th = parse(Float64, m.captures[1])
            w  = parse(Int,    m.captures[2])
            return (th, w)
        else
            # fallback in case the label format is unexpected
            return (Inf, Inf)
        end
    end
    # --- Sort labels numerically by threshold, then weight ---
    all_labels = sort(all_labels, by = l -> parse_th_w(l))

    # --- Now build your arrays using the new sorted order ---
    pauli_vals    = [get(Pauli_errors, l, NaN) for l in all_labels]
    majorana_vals = [get(Majorana_errors, l, NaN) for l in all_labels]
    
    println("All labels: ", all_labels)

    # Plot grouped bars
    bar(
    all_labels,
    [pauli_vals majorana_vals],
    label = ["Pauli" "Majorana"],
    bar_width = 0.6,
    legend = :topright,
    legendfontsize = 30,         # larger legend text
    rotation = 45,
    size = (3500, 3000),
    xlabel = "Threshold / Weight Combination",
    ylabel = "Absolute Error",
    title  = "Pauli vs Majorana Absolute Errors (U=$(U))",
    color = [:white :lightgray], # white and light gray bars
    linecolor = :black,
    linewidth = 0.2,
    # 👇 make everything readable
    titlefont = 50,
    guidefont = 50,              # affects xlabel & ylabel fonts
    tickfont  = 30,              # affects bar tick labels
    left_margin=25*Plots.mm,
    right_margin=25*Plots.mm,
    top_margin=30*Plots.mm,
    bottom_margin=50*Plots.mm,
)

    # Optionally annotate each bar with numeric values (skips NaN)
#    for (i, lbl) in enumerate(all_labels)
#        v1 = pauli_vals[i]
        #v2 = majorana_vals[i]
#        if !isnan(v1)
#            annotate!(i - 0.15, v1, text(@sprintf("%1.2e", v1), 25, :top))
#        end
        #if !isnan(v2)
        #    annotate!(i + 0.15, v2, text(@sprintf("%1.2e", v2), 25, :left))
        #end
#    end
    
    savefig("AbsoluteErrors_Comparison_U=$(U).pdf")
    println("Saved comparison chart to AbsoluteErrors_Comparison_U=$(U).pdf")
    # Clear dicts for next U value
    empty!(Pauli_errors)
    empty!(Majorana_errors)

end
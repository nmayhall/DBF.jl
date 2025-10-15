using DBF
using Base.Threads
using LinearMaps
using KrylovKit

# function Base.Vector(k::OrderedDict{Ket{N},T}) where {N,T}
#     vec = zeros(T,Int128(2)^N)
#     for (k,coeff) in k
#         vec[PauliOperators.index(k)] = T(coeff) 
#     end
#     return vec 
# end

# function Base.Matrix(k::OrderedDict{Ket{N},T}) where {N,T}
#     vec = zeros(T,Int128(2)^N,1)
#     for (k,coeff) in k
#         vec[PauliOperators.index(k),1] = T(coeff) 
#     end
#     return vec 
# end


function matvec(O::XZPauliSum, v::Dict{Ket{N}, T}) where {N,T}
    s = KetSum(N, T)
    sizehint!(s, length(O)) 

    for (vi, ci) in v
        for (x, zs) in O
            b = Ket{N}(vi.v ⊻ x)
            
            val = get(s, b, T(0))
            for (z, c) in zs
                p = PauliBasis{N}(z,x)
                ph,b = p*vi
                val += ph * c * ci
            end
            s[b] = val
        end
    end
    return s
end


function subspace_matvec(O::XZPauliSum, v::KetSum{N,T}) where {N,T}
    s = deepcopy(v)
    # return subspace_matvec!(s, O, v) 
    return subspace_matvec_thread!(s, O, v) 
end

function subspace_matvec!(s::Dict{Ket{N}, T}, O::XZPauliSum, v::Dict{Ket{N}, T}) where {N,T}
    s = deepcopy(v)
    for (sk,sc) in s 
        s[sk] = T(0)
    end

    for (vi, ci) in v
        for (x, zs) in O
            b = Ket{N}(vi.v ⊻ x)
            
            haskey(s,b) || continue

            val = get(s, b, T(0))
            for (z, c) in zs
                p = PauliBasis{N}(z, x)
                ph, b = p*vi
                val += ph * c * ci
            end
            s[b] = val
        end
    end
    return s
end
    
    
function subspace_matvec_thread!(s::KetSum{N,T}, O::XZPauliSum, v::KetSum{N,T}) where {N,T}
    s = KetSum(N, T=ComplexF64) 
    for (sk,sc) in v 
        s[sk] = 0
    end

    function collect_sigma_block!(s)
        #
        #|i>si = \sum_kj hj Pj |k> vk
        #
        @threads for i in collect(keys(s))
            si = s[i]
            for (x, zs) in O
                k = Ket{N}(x ⊻ i.v)

                haskey(v, k) || continue

                vk = get(v, k, T(0))
                for (z, c) in zs
                    p = PauliBasis{N}(z, x)
                    ph, _ = p * k
                    si += ph * c * vk 
                end
            end
            s[i] = si 
        end
    end
    collect_sigma_block!(s)
    return s
end
    
"""
    LinearMaps.LinearMap(O::XZPauliSum{T}, basis::Vector{Ket{N}};
                                ishermitian=true,
                                issymmetric=true) where {N,T}

Build a `LinearMap` that computes the action of a trial vector onto an `XZPauliSum`
"""
function LinearMaps.LinearMap(O::XZPauliSum{T}, basis::Vector{Ket{N}};
                                ishermitian=true,
                                issymmetric=true) where {N,T}
   
    dim = length(basis)
    vin = KetSum(N, T=ComplexF64)
    sizehint!(vin, dim)
    for i in basis
        vin[i] = 0
    end

    function my_matvec(v)
        n = length(v)
        
        n == dim || throw(DimensionMismatch)

        s = zeros(ComplexF64, size(v))
       
        fill!(vin,v,basis)
        # for idx in 1:n
        #     vin[basis[idx]] = v[idx]
        # end
        tmp = DBF.subspace_matvec(O, vin)
        for idx in 1:n
            s[idx] = tmp[basis[idx]]
        end
        flush(stdout)
        flush(stderr)
        return s
    end

    return LinearMap(my_matvec, dim, dim; 
                    issymmetric=issymmetric, 
                    ismutating=false, ishermitian=ishermitian)
end 


function Base.fill!(k::KetSum{N}, v::Vector{T}, basis::Vector{Ket{N}}) where {N,T}
    length(k) == length(v) || throw(DimensionMismatch)
    length(k) == length(basis) || throw(DimensionMismatch)
    for (key,val) in zip(basis,v)
        k[key] = val
    end
    return k
end
function Base.fill!(v::Vector{T}, k::KetSum{N}, basis::Vector{Ket{N}}) where {N,T}
    length(k) == length(v) || throw(DimensionMismatch)
    length(k) == length(basis) || throw(DimensionMismatch)
    for (idx,key) in enumerate(basis) 
        v[idx] = k[key] 
    end
    return v
end
function PauliOperators.KetSum(basis::Vector{Ket{N}}; T=Float64) where N
    return KetSum{N,T}(basis .=> zeros(T,length(basis)))
end

"""
    cepa(H::XZPauliSum, v0::KetSum{N}; thresh=1e-4) where N

PHPc + PHQc = EPc
QHPc + QHQc = EQc

or 

PHPc +  0   + PHRc = EPc
 0     QHQc + QHRc = EQc
RHPc + RHQc + RHRc = ERc

Qc = (EQ-QHQ)^-1 QHRc

RHPc + RHQ (EQ-QHQ)^-1 QHRc + RHRc = ERc

RHPc = (E - RHR - RHQ * (EQ-QHQ)^-1 QHR) Rc
(E - RHR - RHQ * (EQ-QHQ)^-1 QHR)^-1 RHPc = Rc

PHPc + PHR (E - RHR - RHQ * (EQ-QHQ)^-1 QHR)^-1 RHPc = EPc



RHPc + RHQc = (ER-RHR)Rc
(ER-RHR)^-1 RHPc + (ER-RHR)^-1 RHQc = Rc

PHPc + PHR (E-RHR)^-1 RHPc + PHP (ER-RHR)^-1 RHQc = EPc  

(EQ-QHQ)Qc = QHRc
pinv(QHR)(EQ-QHQ)Qc = Rc

P = |v><v|
Q = |wi><wi| + |x><x|
QHPc = (EQ - QHQ)Qc     : b = Ax
Qc = (EQ - QHQ)^-1 QHPc 
cPHPc + cPHQ (EQ-QHQ)^-1 QHPc = E

E0 + cPH(E-EP - H)
"""
function cepa(H::PauliSum, v0::Ket{N}; thresh=1e-4, verbose=4) where N

    ref_basis = [v0]
    vref = KetSum(ref_basis)
    fill!(vref, [1], ref_basis)
    b = DBF.matvec(pack_x_z(H), vref)
    coeff_clip!(b, thresh=thresh)
    delete!(b, v0)
    basis = collect(keys(b))
    e0 = expectation_value(H, v0)
    A = pack_x_z(e0*Pauli(N) - H)
    
    verbose < 1 || @printf(" Size of basis: %i\n", length(basis)) 
    
    # Amat = Matrix(A, basis)
    bvec = Vector(b, basis)

    # e = e0 + bvec'*pinv(diagm(diag(Amat)))*bvec
    # e = e0 + bvec'*pinv(Amat)*bvec
    # @printf(" e0 = %12.8f e(cepa) = %12.8f\n", e0, e)
    
    Amap = LinearMap(A, basis)
    # time = @elapsed x, info = KrylovKit.linsolve(Amap, bvec,
    time = @elapsed x, info = KrylovKit.linsolve(Amap, bvec,
                                            verbosity   = 4,
                                            maxiter     = 10,
                                            issymmetric = true,
                                            ishermitian = true,
                                            tol         = 1e-6)

    e = e0 + x' * bvec

    @printf(" E0 = %12.8f E(cepa) = %12.8f time: %12.8f\n", e0, e, time)
    return e0, e
end

function fois_ci(Hin::PauliSum, v0::Ket{N}; thresh=1e-4, verbose=4) where N

    ref_basis = [v0]
    vref = KetSum(ref_basis)
    fill!(vref, [1], ref_basis)
    H = pack_x_z(Hin)
    fois = DBF.matvec(H, vref)
    coeff_clip!(fois, thresh=thresh)
    basis = collect(keys(fois))
    e0 = expectation_value(H, v0)

    verbose < 1 || @printf(" Size of basis: %i\n", length(basis)) 

    # e = e0 + bvec'*pinv(diagm(diag(Amat)))*bvec
    # e = e0 + bvec'*pinv(Amat)*bvec
    # @printf(" e0 = %12.8f e(cepa) = %12.8f\n", e0, e)
   
    vref = project(vref, basis)
    vguess = Vector(vref, basis)
    Hmap = LinearMap(H, basis)
    time = @elapsed e, v, info = KrylovKit.eigsolve(Hmap, vguess, 1, :SR,
                                                verbosity   = 3,
                                                maxiter     = 10,
                                                issymmetric = true,
                                                ishermitian = true,
                                                eager       = true,
                                                tol         = 1e-8)


    @printf(" E0 = %12.8f E(var)  = %12.8f time: %12.8f\n", e0, e[1], time)
    return e0, e, v
end

"""
    pt2(H::PauliSum{N,T}, k::Ket{N}) where {N,T}

e2 = |<k|Ho|x>|^2 / (e0 - <x|Hd|x>)
"""
function pt2(H::PauliSum{N,T}, ψ::Ket{N}) where {N,T}
    Hd = diag(H)
    e2 = T(0)
    e0 = expectation_value(Hd,ψ)

    h = pack_x_z(H)
    def = Vector{Tuple{Int128, Float64}}()

    for (x,dx) in h

        # make sure p isn't diagonal
        x != 0 || continue       
        
        σHψ = 0
        σ = Ket{N}(0)
        
        hx = get(h, x, def)
        for (z,c) in hx
            pzx = PauliBasis{N}(z,x)
            czx, σ = pzx * ψ
            σHψ += czx * c 
        end
        e2 +=  abs2(σHψ) / (e0 - expectation_value(Hd, σ))
        # c2,k2 = p*k
        
        # k2 != k || error(" k==k2")
        # e2 += (c*c2)'*(c*c2) / (e0 - expectation_value(Hd, k2))
        # e2 += 1 / (e0 - expectation_value(Hd, k2))
        # e2 += 1 / (e0)
        # @show (c*c2)'*(c*c2) / (e0 - expectation_value(Hd, k2))
    end
    return e0, e2
end


function project(k::KetSum{N,T}, basis::Vector{Ket{N}}) where {N,T}
    out = KetSum(basis,T=T)
    for (key,val) in out
        out[key] = get(k, key, T(0))
    end
    return out
end
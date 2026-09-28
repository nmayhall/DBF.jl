#!/usr/bin/env python3
# =============================================================================
#  UHF (AFM broken symmetry) -> UCCSD -> FCI for the 1D Hubbard chain, matching
#  the Julia model in hubbard_hf_basis.jl EXACTLY:
#    OBC, t=1, on-site U, half-filling (na=nb=L/2), mu = U/2 (particle-hole
#    symmetric), AFM broken-symmetry UHF.
#  Reports E_HF, E_CCSD, E_FCI and the % of correlation energy recovered by CCSD:
#     %corr = (E_HF - E_CCSD) / (E_HF - E_FCI) * 100.
#
#  Usage: python3 hubbard_ccsd.py <L> <U> [t]
# =============================================================================
import sys
import numpy as np
from pyscf import gto, scf, cc, fci, ao2mo


def hubbard_ccsd(L, U, t=1.0, do_fci=True):
    na = nb = L // 2
    mu = U / 2.0
    # one-body (OBC hopping) + mu diagonal shift (half-filling PH symmetry)
    h1 = np.zeros((L, L))
    for p in range(L - 1):
        h1[p, p + 1] = -t
        h1[p + 1, p] = -t
    for p in range(L):
        h1[p, p] -= mu
    # on-site U: eri[i,i,i,i] = U (opposite-spin only for Hubbard; same-spin
    # vanishes by Pauli, handled automatically by the antisymmetrized UHF/CC).
    eri = np.zeros((L, L, L, L))
    for i in range(L):
        eri[i, i, i, i] = U

    mol = gto.M()
    mol.nelectron = na + nb
    mol.spin = na - nb          # 0 at half filling
    mol.incore_anyway = True
    mol.build()

    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(L)
    mf._eri = ao2mo.restore(8, eri, L)
    mf.max_cycle = 500
    mf.conv_tol = 1e-11

    # AFM broken-symmetry initial guess (alpha on odd sites, beta on even sites)
    dma = np.zeros((L, L)); dmb = np.zeros((L, L))
    for i in range(L):
        if i % 2 == 0:
            dma[i, i] = 1.0
        else:
            dmb[i, i] = 1.0
    e_hf = mf.kernel((dma, dmb))
    # stability: chase down to the broken-symmetry minimum
    for _ in range(5):
        mo = mf.stability()[0]
        dm = mf.make_rdm1(mo, mf.mo_occ)
        e_new = mf.kernel(dm)
        if abs(e_new - e_hf) < 1e-10:
            e_hf = e_new
            break
        e_hf = e_new

    mycc = cc.UCCSD(mf)
    mycc.max_cycle = 500
    mycc.conv_tol = 1e-10
    e_ccsd = mycc.kernel()[0] + e_hf

    e_fci = np.nan
    if do_fci:
        cisolver = fci.FCI(mf)
        h1_mo_a = mf.mo_coeff[0].T @ h1 @ mf.mo_coeff[0]
        # use direct_uhf with MO-basis integrals
        from pyscf.fci import direct_uhf
        moa, mob = mf.mo_coeff
        h1a = moa.T @ h1 @ moa
        h1b = mob.T @ h1 @ mob
        eri_aa = ao2mo.incore.general(mf._eri, (moa, moa, moa, moa), compact=False).reshape(L, L, L, L)
        eri_ab = ao2mo.incore.general(mf._eri, (moa, moa, mob, mob), compact=False).reshape(L, L, L, L)
        eri_bb = ao2mo.incore.general(mf._eri, (mob, mob, mob, mob), compact=False).reshape(L, L, L, L)
        e_fci, _ = direct_uhf.kernel((h1a, h1b), (eri_aa, eri_ab, eri_bb), L, (na, nb))

    return e_hf, e_ccsd, e_fci


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    U = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    do_fci = (2 * L) <= 20
    e_hf, e_ccsd, e_fci = hubbard_ccsd(L, U, t, do_fci=do_fci)
    print(f"L={L} U={U} t={t}  n={2*L}")
    print(f"  E_HF   = {e_hf:.8f}")
    print(f"  E_CCSD = {e_ccsd:.8f}")
    if do_fci and np.isfinite(e_fci):
        ec = e_hf - e_fci
        pct = (e_hf - e_ccsd) / ec * 100.0 if abs(ec) > 1e-12 else float('nan')
        print(f"  E_FCI  = {e_fci:.8f}")
        print(f"  E_corr(FCI) = {ec:.8f}")
        print(f"  CCSD %corr  = {pct:.2f}%")
    else:
        print(f"  E_FCI  = (skipped, n>20)")

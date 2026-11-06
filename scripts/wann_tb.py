#!/usr/bin/env python
"""Stage A: Wannier _tb.dat machinery for SHG (Sipe-Aversa length gauge).
Parses H(R) + position matrix r(R); builds H(k), velocity (dH/dk), and the interband
Berry connection / dipole r^a_nm via Wang, Yates, Souza, Vanderbilt PRB 74,195118 (2006)."""
import numpy as np


def read_tb(fn):
    with open(fn) as handle:
        L = handle.read().splitlines()
    p=1
    lat=np.array([[float(x) for x in L[p+i].split()] for i in range(3)]); p+=3
    nw=int(L[p]); p+=1; nr=int(L[p]); p+=1
    deg=[]
    while len(deg)<nr: deg+=[int(x) for x in L[p].split()]; p+=1
    deg=np.array(deg,float)
    if np.any(deg <= 0):
        raise ValueError("Wannier degeneracies must be positive")
    # H(R): nr blocks; each: blank, "Rx Ry Rz", nw*nw lines "m n ReH ImH"
    R=np.zeros((nr,3),int); H=np.zeros((nr,nw,nw),complex)
    def skip_blank(p):
        while p<len(L) and L[p].strip()=="" : p+=1
        return p
    for ir in range(nr):
        p=skip_blank(p); R[ir]=[int(x) for x in L[p].split()]; p+=1
        for _ in range(nw*nw):
            t=L[p].split(); m,n=int(t[0])-1,int(t[1])-1; H[ir,m,n]=float(t[2])+1j*float(t[3]); p+=1
    # r(R): nr blocks; each: blank,"Rx Ry Rz", nw*nw lines "m n Rx Ix Ry Iy Rz Iz"
    Rr=np.zeros((nr,3),int); rr_raw=np.zeros((nr,nw,nw,3),complex)
    for ir in range(nr):
        p=skip_blank(p)
        if p>=len(L):
            raise ValueError("truncated position-matrix section in Wannier _tb.dat")
        Rr[ir]=[int(x) for x in L[p].split()]; p+=1
        for _ in range(nw*nw):
            if p>=len(L):
                raise ValueError("truncated position-matrix block in Wannier _tb.dat")
            t=L[p].split(); m,n=int(t[0])-1,int(t[1])-1
            rr_raw[ir,m,n]=[float(t[2])+1j*float(t[3]),float(t[4])+1j*float(t[5]),float(t[6])+1j*float(t[7])]; p+=1
    r_lookup={tuple(r): ir for ir,r in enumerate(Rr)}
    if len(r_lookup)!=nr or any(tuple(r) not in r_lookup for r in R):
        raise ValueError("Hamiltonian and position sections use different R vectors")
    rr=rr_raw[[r_lookup[tuple(r)] for r in R]]
    return dict(lat=lat, nw=nw, R=R, deg=deg, H=H, r=rr, Rcart=R@lat)


def at_k(tb, kf, energy_tol_eV=1e-6, with_generalized_derivative=False):
    """Evaluate the Wannier model at one reduced k point.

    Set ``with_generalized_derivative`` for the mixed-band SHG kernel.  The
    returned ``rd[a,b,n,m]`` is the generalized derivative
    :math:`r^a_{nm;b}` evaluated by differentiating the real-space Wannier
    Hamiltonian and position matrices, following HopTB's orthogonal-basis
    implementation rather than a truncated-band sum rule.
    """
    if energy_tol_eV<=0:
        raise ValueError("energy_tol_eV must be positive")
    ph=np.exp(2j*np.pi*(tb["R"]@kf))/tb["deg"]        # (nr,)
    Hk=np.einsum('r,rmn->mn', ph, tb["H"])
    Hk=(Hk+Hk.conj().T)/2
    dHk=np.einsum('r,ra,rmn->amn', ph, 1j*tb["Rcart"], tb["H"], optimize=True)  # (3,nw,nw) dH/dk (eV*Ang)
    Ak=np.einsum('r,rmna->amn', ph, tb["r"])                     # (3,nw,nw) Wannier Berry conn
    dHk=(dHk+dHk.swapaxes(1,2).conj())/2
    Ak=(Ak+Ak.swapaxes(1,2).conj())/2
    E,U=np.linalg.eigh(Hk)
    Hbar=np.einsum('xi,axy,yj->aij', U.conj(), dHk, U, optimize=True)
    Abar=np.einsum('xi,axy,yj->aij', U.conj(), Ak, U, optimize=True)
    dE=E[:,None]-E[None,:]                       # w_nm = E_n - E_m
    D=np.zeros_like(Hbar)
    interband=np.abs(dE)>=energy_tol_eV
    np.fill_diagonal(interband,False)
    with np.errstate(divide='ignore',invalid='ignore'):
        np.divide(Hbar,-dE[None,:,:],out=D,where=interband[None,:,:])
    # Fix the diagonal band phase so that the total Hamiltonian-gauge Berry
    # connection vanishes there: i*D_nn + Abar_nn = 0.  The off-diagonal
    # dipoles below do not depend on D_nn, but their generalized derivatives do.
    diagonal=np.arange(len(E))
    D[:,diagonal,diagonal]=1j*Abar[:,diagonal,diagonal]
    r_nm=(Abar+1j*D)*interband[None,:,:]          # off-diagonal eigen-gauge dipole
    v=Hbar                                       # velocity (eV*Ang); diagonal v_nn = band velocity
    result = dict(E=E,U=U,w=dE,r=r_nm,v=v,Hbar=Hbar,Abar=Abar,D=D)
    if not with_generalized_derivative:
        return result

    # Derivatives are with respect to Cartesian k (1/Angstrom).  Because
    # exp(i k.R) is represented as exp(2 pi i k_frac.R_frac), each derivative
    # contributes i*R_cart.
    d2Hk=np.einsum(
        'r,ra,rb,rmn->abmn',
        ph,
        1j*tb["Rcart"],
        1j*tb["Rcart"],
        tb["H"],
        optimize=True,
    )
    dAwk=np.einsum(
        'r,rb,rmna->abmn',
        ph,
        1j*tb["Rcart"],
        tb["r"],
        optimize=True,
    )
    d2Hk=(d2Hk+d2Hk.swapaxes(-1,-2).conj())/2
    dAwk=(dAwk+dAwk.swapaxes(-1,-2).conj())/2
    d2Hbar=np.einsum('xi,abxy,yj->abij',U.conj(),d2Hk,U,optimize=True)
    dAwbar=np.einsum('xi,abxy,yj->abij',U.conj(),dAwk,U,optimize=True)

    # D^a_nm=<u_n|partial_a u_m> in the Hamiltonian gauge.  It is already
    # available from the same off-diagonal combination used to construct r.
    band_velocity=np.real(np.diagonal(Hbar,axis1=1,axis2=2))
    denominator=E[None,:]-E[:,None]  # E_m-E_n, indexed [n,m]
    rd=np.zeros((3,3,len(E),len(E)),complex)
    for a in range(3):
        for b in range(3):
            tmpH=(
                Hbar[a]@D[b]
                +d2Hbar[a,b]
                +D[b].conj().T@Hbar[a]
            )
            tmpA=(
                Abar[a]@D[b]
                +dAwbar[a,b]
                +D[b].conj().T@Abar[a]
            )
            delta_b=band_velocity[b][None,:]-band_velocity[b][:,None]
            numerator=1j*tmpH-1j*delta_b*D[a]
            np.divide(
                numerator,
                denominator,
                out=rd[a,b],
                where=interband,
            )
            rd[a,b][interband]+=tmpA[interband]
            np.fill_diagonal(rd[a,b],0.0)
    result["rd"]=rd
    return result

if __name__=="__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/wann_tb.py MODEL_tb.dat")
    tb=read_tb(sys.argv[1])
    print(f"nw={tb['nw']} nr={len(tb['R'])} lat[0]={tb['lat'][0]}")
    # SELF-TEST 1: H(k) eigenvalues at K match DFT (already validated); print gap-ish
    k=np.array([2/3,1/3,0.0]); q=at_k(tb,k); print("E@K (first 6):",np.round(q['E'][:6],3))
    # SELF-TEST 2: velocity vs finite-difference of E(k): dE_n/dk_a = v_nn^a (Hellmann-Feynman)
    h=1e-4; a=0
    kp=k.copy(); kp[a]+=h; km=k.copy(); km[a]-=h
    # need cartesian dk: convert frac step to cartesian via reciprocal lattice
    B=2*np.pi*np.linalg.inv(tb['lat']).T
    dk_cart=np.array([h,0,0])@B
    Ep=at_k(tb,kp)['E']; Em=at_k(tb,km)['E']
    dEdk_fd=(Ep-Em)/(2*np.linalg.norm(dk_cart))
    direction=dk_cart/np.linalg.norm(dk_cart)
    vdiag=np.array([q['v'][a].diagonal().real for a in range(3)])
    vproj=direction@vdiag
    print("FD dE/dk (band0-5):",np.round(dEdk_fd[:6],3))
    print("v_nn proj (band0-5):",np.round(vproj[:6],3))

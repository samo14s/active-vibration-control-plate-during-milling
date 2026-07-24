"""Equivalence + timing check: structured shift-register update vs explicit dense D_i product."""
import time
import _bootstrap  # noqa: F401
import numpy as np
from closed_loop_sld import build_monodromy, discretize_controller, _step_matrices

def dense_reference(omega_n, zeta_n, Dp, H_modal, D_obs, a4_array, dt, m_div, refine=1, ctrl=None):
    n = len(omega_n); n2 = 2*n
    Kp = np.diag(np.asarray(omega_n,float)**2)
    Cp = np.diag(2*np.asarray(zeta_n,float)*np.asarray(omega_n,float))
    Dp = np.asarray(Dp,float).reshape(n); DpDpT = np.outer(Dp,Dp)
    B = np.zeros((n2,1)); B[n:,0] = np.asarray(H_modal,float).reshape(n)
    C = np.zeros((1,n2)); C[0,:n] = np.asarray(D_obs,float).reshape(n)
    closed = ctrl is not None
    N = n2*(m_div+1) + (n2 if closed else 0); s = n2*(m_div+1)
    if closed:
        Ao=np.asarray(ctrl['Ao'],float); Gu=np.asarray(ctrl['Gu'],float).reshape(n2,1)
        Gy=np.asarray(ctrl['Gy'],float).reshape(n2,1); Kf=np.asarray(ctrl['K'],float).reshape(1,n2)
        GyC=Gy@C; Ahat=Ao-Gu@Kf
    Phi_T = np.eye(N)
    for i in range(m_div):
        a=float(a4_array[i])
        A=np.zeros((n2,n2)); A[:n,n:]=np.eye(n); A[n:,:n]=-(Kp+a*DpDpT); A[n:,n:]=-Cp
        Ad=np.zeros((n2,n2)); Ad[n:,:n]=a*DpDpT
        Phi_i,Gam_i,Psi_i=_step_matrices(A,Ad,B,dt)
        D=np.zeros((N,N)); D[:n2,:n2]=Phi_i; D[:n2,n2*m_div:n2*(m_div+1)]=Gam_i
        for j in range(m_div): D[n2*(j+1):n2*(j+2), n2*j:n2*(j+1)]=np.eye(n2)
        if closed:
            D[:n2,s:s+n2]=-Psi_i@Kf
            if i%refine==0: D[s:s+n2,:n2]=GyC; D[s:s+n2,s:s+n2]=Ahat
            else: D[s:s+n2,s:s+n2]=np.eye(n2)
        Phi_T = D@Phi_T
    return Phi_T

rng=np.random.default_rng(0)
n=3; m_div=40
omega=np.array([3264.,6608.,16845.]); zeta=np.array([.0031,.0017,.0027])
Dp=rng.normal(size=n); H=rng.normal(size=n)*0.05; Dobs=rng.normal(size=n)
a4=rng.normal(size=m_div)*1e5
A=np.zeros((6,6)); A[:3,3:]=np.eye(3); A[3:,:3]=-np.diag(omega**2); A[3:,3:]=-np.diag(2*zeta*omega)
B=np.zeros((6,1)); B[3:,0]=H; C=np.zeros((1,6)); C[0,:3]=Dobs
L=rng.normal(size=(6,1))*1e3; K=rng.normal(size=(1,6))*1e2
Ao,Gu,Gy=discretize_controller(A,B,C,L,1e-4)
ctrl=dict(Ao=Ao,Gu=Gu,Gy=Gy,K=K)

for tag,cc,ref in [("open loop",None,1),("closed loop, refine=1",ctrl,1),("closed loop, refine=4",ctrl,4)]:
    md = m_div*ref
    a4r = np.repeat(a4, ref)[:md] if ref>1 else a4
    P1=build_monodromy(omega,zeta,Dp,H,Dobs,a4r,1e-4,md,refine=ref,ctrl=cc)
    P2=dense_reference(omega,zeta,Dp,H,Dobs,a4r,1e-4,md,refine=ref,ctrl=cc)
    err=np.max(np.abs(P1-P2))/max(np.max(np.abs(P2)),1e-30)
    r1=np.max(np.abs(np.linalg.eigvals(P1))); r2=np.max(np.abs(np.linalg.eigvals(P2)))
    print(f"  {tag:<24} max rel matrix err = {err:.3e}   rho: {r1:.10f} vs {r2:.10f}   {'PASS' if err<1e-10 else 'FAIL'}")

# timing at a realistic size
md=82
a4b=np.repeat(a4,3)[:md]
t=time.perf_counter(); [build_monodromy(omega,zeta,Dp,H,Dobs,a4b,1e-4,md,ctrl=ctrl) for _ in range(5)]; t1=(time.perf_counter()-t)/5
t=time.perf_counter(); [dense_reference(omega,zeta,Dp,H,Dobs,a4b,1e-4,md,ctrl=ctrl) for _ in range(5)]; t2=(time.perf_counter()-t)/5
print(f"\n  timing at m_div=82 (N={6*83+6}):  structured {t1*1e3:7.2f} ms   dense {t2*1e3:8.2f} ms   speedup {t2/t1:6.1f}x")

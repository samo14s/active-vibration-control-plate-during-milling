import numpy as np, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
import simulation_base as SB
from simulation_base import SimBase
from plate_model import PlateModel
from kirchhoff_q4 import shape_at_point
from scipy.sparse.linalg import spsolve

# ---------- 1. souplesse statique EXACTE (tous modes) --------------------
p=PlateModel(SB.PLATE_L,SB.PLATE_H,SB.PLATE_T,SB.RHO,SB.YOUNG,SB.POISSON,
             N1=SB.MESH_N1,N2=SB.MESH_N2,n_modes=SB.N_MODES,
             zeta_modes=SB.ZETA_MODES,verbose=False)
p.precompute_Dp(zp_pos=SB.PLATE_H-0.15e-3,n_pos=2001); p.set_observation(*SB.SENSOR_XZ)
p.add_piezo_patch(SB.PATCH['x1'],SB.PATCH['x2'],SB.PATCH['z1'],SB.PATCH['z2'],
                  SB.PATCH['d31'],SB.PATCH['thickness'],SB.PATCH['E'],SB.PATCH['nu'])
Nobs=shape_at_point(*SB.SENSOR_XZ,SB.MESH_N1,SB.MESH_N2,p.lex,p.ley,p.n1)[p.DOFf]
u=spsolve(p.K_free.tocsc(),Nobs)
C_exact=float(Nobs@u)
C_5=float(np.sum(p.D_obs**2/p.omega_n**2))
print('--- souplesse statique au coin sup. droit (100,80) ---')
print(f'  EF exacte (tous ddl)        : {C_exact*1e6:8.3f} um/N')
print(f'  troncature 5 modes          : {C_5*1e6:8.3f} um/N  ({C_5/C_exact*100:.1f} %)')
print(f'  poutre encastree pleine larg: {(0.080**3/(3*SB.YOUNG*0.1*0.004**3/12))*1e6:8.3f} um/N')
print(f'  Fig 12(a) numerisee, f<140Hz:   43.83 um/N   -> facteur {43.83e-6/C_exact:.2f}')

# ---------- 2. lobes de stabilite analytiques (Altintas-Budak, ordre 0) ----
sim=SimBase(); pl=sim.plate
Dt=pl.Dp_array[:,0]; om=pl.omega_n; z=np.array(SB.ZETA_MODES)
k1,k2=sim.k1c,sim.k2c
ae,R,NT=SB.AE_NOM,SB.D_TOOL/2,SB.N_TEETH
phst=np.pi-np.arccos(1-ae/R); phex=np.pi
# coefficient directionnel moyen a_yy  (Fy sur yr) : moyenne de alpha4 sur tau, par mm axial
from milling_force import precompute_alpha_periodic
rpm=4900; tau=60/(NT*rpm); dt=tau/82
a3t,a4t=precompute_alpha_periodic(dt,82,82,2*np.pi*rpm/60,NT,R,SB.HELIX,
                                  phst,phex,SB.PLATE_H-1e-3,SB.PLATE_H,k1,k2,SB.KT)
a4_bar_per_m=np.mean(a4t)/1e-3     # N/m par metre de profondeur axiale
print(f'\n--- coefficient directionnel moyen alpha4/ap = {a4_bar_per_m:.4e} N/m/m ---')
def FRF(f):
    w=2*np.pi*f
    return np.sum(Dt[:,None]**2/((om[:,None]**2-w[None,:]**2)+2j*z[:,None]*om[:,None]*w[None,:]),axis=0)
lim=[];
for rpm_ in [4900]:
    best=1e9; fbest=None
    ff=np.linspace(300,4600,600000); G=FRF(ff); ReG=np.real(G)
    for i in range(len(ff)):
        if ReG[i]>=0: continue
        aplim=-1.0/(2*a4_bar_per_m*ReG[i])     # ap_lim = -1/(2 Kdir Re(G))
        if aplim<best: best=aplim; fbest=ff[i]
    print(f'  ap_lim analytique (ordre 0, toutes vitesses) = {best*1e3:.4f} mm '
          f'  a f_c = {fbest:.0f} Hz')
print(f'  ap_lim Newmark boucle ouverte a 4900 tr/min   = {sim.stability_limit(None,rpm=4900)*1e3:.4f} mm')
print( '  article Fig.13(b) a 4900 tr/min               ~ 0.05 mm')
print( "  article : frequence de broutement simulee     = 1135 Hz (mode 2)")

# ---------- 3. quelle est la frequence de broutement du modele ? ----------
r=sim.run(None,rpm=4900,ap=0.30e-3,T=0.10,keep_signals=True)
y=r['y']; n=len(y); i0=n//3
Y=np.abs(np.fft.rfft(y[i0:]*np.hanning(n-i0))); fr=np.fft.rfftfreq(n-i0,r['dt'])
k=np.argsort(Y)[::-1][:8]
print('\n--- contenu spectral de la simulation a 4900 tr/min, ap=0.30 mm ---')
print('  ft =',NT*rpm/60,'Hz ; pics dominants :',sorted([round(fr[j]) for j in k]))

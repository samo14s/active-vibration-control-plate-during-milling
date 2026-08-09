import numpy as np, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
import simulation_base as SB
from simulation_base import SimBase
from milling_force import precompute_alpha_periodic

sim=SimBase(); pl=sim.plate
Dt=pl.Dp_array[:,0]; om=pl.omega_n; z=np.array(SB.ZETA_MODES)
NT,R=SB.N_TEETH,SB.D_TOOL/2
phst=np.pi-np.arccos(1-SB.AE_NOM/R); phex=np.pi
rpm=4900; tau=60/(NT*rpm); dt=tau/82
_,a4t=precompute_alpha_periodic(dt,82,82,2*np.pi*rpm/60,NT,R,SB.HELIX,
                                phst,phex,SB.PLATE_H-1e-3,SB.PLATE_H,sim.k1c,sim.k2c,SB.KT)
A=np.mean(a4t)/1e-3
def G(f):
    w=2*np.pi*f
    return np.sum(Dt[:,None]**2/((om[:,None]**2-w[None,:]**2)+2j*z[:,None]*om[:,None]*w[None,:]),axis=0)
ff=np.linspace(300,4600,800000); Gr=np.real(G(ff))
ap=np.where(Gr>0,1.0/(-2*A*Gr),np.inf)     # A<0 -> ap>0 requiert Re(G)>0
i=int(np.nanargmin(ap))
print('--- lobes analytiques (ordre 0, Altintas-Budak) ---')
print(f'  alpha4_moyen/ap = {A:.4e} N/m par metre')
print(f'  ap_lim (min sur toutes vitesses) = {ap[i]*1e3:.4f} mm  a f_c = {ff[i]:.0f} Hz')
# par mode isole
for k in range(5):
    Gk=np.real(Dt[k]**2/((om[k]**2-(2*np.pi*ff)**2)+2j*z[k]*om[k]*(2*np.pi*ff)))
    apk=np.where(Gk>0,1.0/(-2*A*Gk),np.inf); j=int(np.nanargmin(apk))
    print(f'   mode {k+1} seul ({om[k]/2/np.pi:6.1f} Hz) : ap_lim = {apk[j]*1e3:7.4f} mm a {ff[j]:6.0f} Hz')
print(f'  Newmark boucle ouverte 4900 tr/min = {sim.stability_limit(None,rpm=4900)*1e3:.4f} mm')
print( '  article Fig.13(b) minimum global   ~ 0.03-0.05 mm')

print('\n--- spectre de broutement (4900 tr/min, ap=0.30 mm, fenetre longue) ---')
r=sim.run(None,rpm=4900,ap=0.30e-3,T=0.25,keep_signals=True)
y=r['y']; n=len(y); seg=y[n//2:]
w=np.hanning(len(seg)); Y=np.abs(np.fft.rfft(seg*w)); fr=np.fft.rfftfreq(len(seg),r['dt'])
sel=fr<2500; Y=Y[sel]; fr=fr[sel]
pk=[(fr[j],Y[j]) for j in range(1,len(Y)-1) if Y[j]>Y[j-1] and Y[j]>Y[j+1]]
pk.sort(key=lambda t:-t[1])
print('  ft = 245.0 Hz, fs = 81.67 Hz')
print('  10 pics dominants :',[(round(f,1),round(a/pk[0][1],3)) for f,a in pk[:10]])
print('  modes calibres    :',np.round(sim.freqs,1))

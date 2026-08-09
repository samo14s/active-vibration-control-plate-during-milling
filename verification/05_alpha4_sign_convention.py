"""F10 — quelle convention de signe des efforts de coupe ?

L'Eq. (13) de Du et al. et l'enchainement de ses Eqs. (1)(2)(5)(10)(A.4)
donnent des signes opposes sur a3 et a4. Une premiere version de ce script
comparait la limite analytique d'ordre 0 MINIMISEE SUR TOUTES LES FREQUENCES,
qui est la meme aux deux signes (0.0464 vs 0.0463 mm) -- d'ou la conclusion
erronee que le choix etait sans consequence.

A vitesse FIXEE, les lobes ne tombent pas au meme endroit et l'ecart atteint un
facteur 10. Ce script mesure donc la limite par bissection sur le simulateur
non lineaire, vitesse par vitesse, et confronte les deux colonnes aux Fig. 13(b),
14(a) et 18 de l'article.
"""
import numpy as np, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
import simulation_base as SB
from simulation_base import SimBase
from milling_force import precompute_alpha_periodic
from newmark_solver import NewmarkSimulator

sim=SimBase(); pl=sim.plate
Dt=pl.Dp_array[:,0]; om=pl.omega_n; z=np.array(SB.ZETA_MODES)
NT,R=SB.N_TEETH,SB.D_TOOL/2
phst=np.pi-np.arccos(1-SB.AE_NOM/R)
rpm=4900; tau=60/(NT*rpm); dt=tau/82

def Gf(ff):
    w=2*np.pi*ff
    return np.sum(Dt[:,None]**2/((om[:,None]**2-w[None,:]**2)+2j*z[:,None]*om[:,None]*w[None,:]),axis=0)
ff=np.linspace(300,4600,800000); Gr=np.real(Gf(ff))
_,a4t=precompute_alpha_periodic(dt,82,82,2*np.pi*rpm/60,NT,R,SB.HELIX,phst,np.pi,
                                SB.PLATE_H-1e-3,SB.PLATE_H,sim.k1c,sim.k2c,SB.KT)
A=np.mean(a4t)/1e-3
print('alpha4 moyen / ap =',f'{A:.4e}','N/m/m  (NEGATIF)')
for lab,Aa in [('Eq.(12) telle qu ecrite  (code)          ',A),
               ('signe issu de Eqs.(1)(2)(5)(10)(A.4)     ',-A)]:
    ap=np.where(-2*Aa*Gr>0,1.0/(-2*Aa*Gr),np.inf); i=int(np.nanargmin(ap))
    print(f'  {lab}: ap_lim={ap[i]*1e3:.4f} mm  f_c={ff[i]:6.1f} Hz  '
          f'({"au-dessus" if ff[i]>536.4 and ff[i]<800 or ff[i]>1071.8 and ff[i]<1500 else "au-dessous"} de la resonance)')

print('\n--- simulation Newmark, meme cas, avec les deux signes ---')
for lab,sgn in [('code (Eq.12)',+1.0),('signe derive',-1.0)]:
    s=NewmarkSimulator(pl,dt=dt,T_end=0.25,ft=SB.FZ_NOM,tau=tau,verbose=False)
    a3,a4=precompute_alpha_periodic(dt,82,s.nstep,2*np.pi*rpm/60,NT,R,SB.HELIX,phst,np.pi,
                                    SB.PLATE_H-0.30e-3,SB.PLATE_H,sim.k1c,sim.k2c,SB.KT)
    r=s.simulate(sgn*a3,sgn*a4,np.zeros(s.nstep,int),controller=None,progress=False,stop_threshold=1e-4)
    i=r['stop_idx']; y=r['y'][:i+1]; seg=y[len(y)//2:]
    Y=np.abs(np.fft.rfft(seg*np.hanning(len(seg)))); fr=np.fft.rfftfreq(len(seg),dt)
    m=fr<2000; j=int(np.argmax(Y[m]))
    print(f'  {lab:14s} : divergence={r["diverged_at"]>0}  pic dominant = {fr[m][j]:7.1f} Hz'
          f'   (f1={sim.freqs[0]:.1f}, f2={sim.freqs[1]:.1f})')

print('\n--- frequences de broutement MESUREES par Du et al. (Fig.20/21) ---')
fn=[540,1068,2787,3351,4122]; fc=[580,1123,2684,3563,4335]
for a,b in zip(fn,fc): print(f'   f_n={a:5d} Hz -> f_c={b:5d} Hz  ({(b/a-1)*100:+5.1f} %)')

print('--- limite de passe en boucle ouverte, vitesse par vitesse (mm) ---')
from simulation_base import SimBase as _SB
_a = _SB(force_sign=+1.0)
_b = _SB(force_sign=-1.0)
print(f"{'tr/min':>8s} {'Eq.(13) publiee':>17s} {'signe derive':>14s}")
for _r in (3000, 3600, 4200, 4900, 5500, 6000, 7200):
    print(f'{_r:8d} {_a.stability_limit(None, rpm=_r, tol=2e-6)*1e3:17.4f} '
          f'{_b.stability_limit(None, rpm=_r, tol=2e-6)*1e3:14.4f}')
print("""
article Fig.13(b) : pics a ~3600 et ~5500 tr/min   -> colonne Eq.(13)
article Fig.14(a) : S (4900 tr/min, 0.30 mm) DIVERGE -> limite < 0.30 mm
                    Eq.(13) 0.0597 OUI ; signe derive 0.6003 NON
article Fig.18    : sans controle < 0.1 mm partout SAUF 5500 tr/min
                    Eq.(13) 4900->0.0597 et 5500->0.3705 OUI ; derive NON
=> l'erreur de signe est dans les equations intermediaires de l'article,
   pas dans son Eq. (13). La base suit l'Eq. (13).""")

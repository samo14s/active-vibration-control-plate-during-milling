"""Balayage des positions de patch : le critere f1 et la signature de la
Fig. 12(b) sont-ils conciliables ?

Pour chaque position et chaque orientation, calcule f1 (patch COLLE, shear lag)
et l'occupation des intervalles inter-mode. Resultat : aucune position ne
satisfait les deux criteres. Voir VERIFICATION.md section 5.
"""
import numpy as np, sys, os
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(_R,'simulation','sim_kit'),os.path.join(_R,'simulation')]
os.chdir(os.path.join(_R,'simulation','sim_kit'))
import simulation_base as SB
from plate_model import PlateModel
from model_v2 import notch_occupancy
L,H=SB.PLATE_L,SB.PLATE_H
OCC=(1,1,0,1); F1=540.0
rows=[]
for orient,(w,h) in [('H 60x20',(0.060,0.020)),('V 20x60',(0.020,0.060))]:
    xs=np.round(np.arange(0.0,L-w+1e-9,0.010),4)
    zs=np.round(np.arange(0.0,H-h+1e-9,0.010),4)
    for x1 in xs:
        for z1 in zs:
            P=dict(SB.PATCH); P.update(x1=x1,x2=x1+w,z1=z1,z2=z1+h)
            p=PlateModel(L,H,SB.PLATE_T,SB.RHO,SB.YOUNG,SB.POISSON,N1=SB.MESH_N1,
                         N2=SB.MESH_N2,n_modes=5,zeta_modes=SB.ZETA_MODES,verbose=False)
            p.precompute_Dp(zp_pos=H-0.15e-3,n_pos=3); p.set_observation(L,H)
            p.add_piezo_patch(P['x1'],P['x2'],P['z1'],P['z2'],P['d31'],P['thickness'],
                              P['E'],P['nu'],G_adh=P['G_adh'],t_adh=P['t_adh'])
            f1=p.freq_n[0]
            p.calibrate_frequencies(SB.F_MEASURED)
            occ=notch_occupancy(p)
            rows.append((orient,x1,z1,f1,occ,occ==OCC,abs(f1/F1-1)*100))
ok=[r for r in rows if r[5]]
print(f'{len(rows)} positions balayees, {len(ok)} reproduisent l occupation (1,1,0,1)\n')
print(f"{'orient':9s} {'x1(mm)':>7s} {'z1(mm)':>7s} {'f1(Hz)':>8s} {'|df1|%':>7s} occupation")
for r in sorted(ok,key=lambda r:r[6]):
    print(f'{r[0]:9s} {r[1]*1e3:7.0f} {r[2]*1e3:7.0f} {r[3]:8.1f} {r[6]:7.2f} {r[4]}')
print('\n--- meilleures positions sur le critere f1 SEUL (occupation ignoree) ---')
for r in sorted(rows,key=lambda r:r[6])[:6]:
    print(f'{r[0]:9s} {r[1]*1e3:7.0f} {r[2]*1e3:7.0f} {r[3]:8.1f} {r[6]:7.2f} {r[4]}  {"OCC OK" if r[5] else ""}')

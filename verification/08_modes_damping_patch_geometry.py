import numpy as np, sys, importlib
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
sys.path.insert(0,'sim_kit'); sys.path.insert(0,'.')
import simulation_base as SB

print('--- A. la limite libre depend-elle du nombre de modes ? (README dit : non) ---')
base_modes, base_f, base_z = SB.N_MODES, list(SB.F_MEASURED), list(SB.ZETA_MODES)
for nm in (1,2,3,5):
    SB.N_MODES=nm; SB.F_MEASURED=base_f[:nm]; SB.ZETA_MODES=base_z[:nm]
    s=SB.SimBase()
    print(f'   {nm} mode(s) : ap_lim(4900) = {s.stability_limit(None,rpm=4900)*1e3:.4f} mm')
SB.N_MODES, SB.F_MEASURED, SB.ZETA_MODES = base_modes, base_f, base_z

print('\n--- B. effet des amortissements Table 4 (zeta4=0.56%, zeta5=0.35%) ---')
for lab,zz in [('base figee 0.31/0.17/0.27/0.30/0.30',[.0031,.0017,.0027,.0030,.0030]),
               ('Table 4    0.31/0.17/0.27/0.56/0.35',[.0031,.0017,.0027,.0056,.0035])]:
    SB.ZETA_MODES=zz; s=SB.SimBase()
    r=s.run(None,rpm=4900,ap=0.05e-3,T=0.20)
    print(f'   {lab} : ap_lim={s.stability_limit(None,rpm=4900)*1e3:.4f} mm  RMS={r["rms_um"]:.4f} um')
SB.ZETA_MODES=base_z

print('\n--- C. frequences EF brutes vs mesures Table 4 : geometrie v1 contre v2 ---')
from plate_model import PlateModel
meas=np.array([540.,1068.,2787.,3351.,4122.])
for lab,P in [('v1 bas-gauche 20x60 (sim_kit)',dict(SB.PATCH)),
              ('v2 bas-droit  60x20 (model_v2)',{**SB.PATCH,'x1':0.04,'x2':0.10,'z1':0.0,'z2':0.02}),
              ('sans patch (plaque nue)       ',None)]:
    p=PlateModel(SB.PLATE_L,SB.PLATE_H,SB.PLATE_T,SB.RHO,SB.YOUNG,SB.POISSON,
                 N1=SB.MESH_N1,N2=SB.MESH_N2,n_modes=5,zeta_modes=base_z,verbose=False)
    if P: p.add_piezo_patch(P['x1'],P['x2'],P['z1'],P['z2'],P['d31'],P['thickness'],P['E'],P['nu'])
    e=(p.freq_n/meas-1)*100
    print(f'   {lab} : {np.round(p.freq_n,1)}  ecarts % {np.round(e,1)}  RMS {np.sqrt(np.mean(e**2)):.2f} %')
print('   article Chebyshev-Ritz         : [ 537.  1101.  2805.  3423.  4254. ]  ecarts % '
      f'{np.round((np.array([537.,1101.,2805.,3423.,4254.])/meas-1)*100,1)}  RMS '
      f'{np.sqrt(np.mean(((np.array([537.,1101.,2805.,3423.,4254.])/meas-1)*100)**2)):.2f} %')

import numpy as np, openpyxl, glob, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
import simulation_base as SB
from plate_model import PlateModel

def load(f):
    ws=openpyxl.load_workbook(f).worksheets[0]
    d=np.array([[r[0],r[1]] for r in ws.iter_rows(values_only=True) if r[0] is not None],float)
    return d[:,0],d[:,1]

fa,Aa=load(glob.glob('experimental*')[0])   # Fig12a  Ref 1 um/N
fb,Ab=load(glob.glob('transfer_functions*')[0])  # Fig12b Ref 0.01 um/V

def build(patch, zeta, structural=True, calib=True):
    p=PlateModel(SB.PLATE_L,SB.PLATE_H,SB.PLATE_T,SB.RHO,SB.YOUNG,SB.POISSON,
                 N1=SB.MESH_N1,N2=SB.MESH_N2,n_modes=SB.N_MODES,
                 zeta_modes=zeta,verbose=False)
    p.precompute_Dp(zp_pos=SB.PLATE_H-0.15e-3,n_pos=2001)
    p.set_observation(*SB.SENSOR_XZ)
    p.add_piezo_patch(patch['x1'],patch['x2'],patch['z1'],patch['z2'],
                      patch['d31'],patch['thickness'],patch['E'],patch['nu'],
                      structural=structural)
    f_raw=p.freq_n.copy()
    if calib: p.calibrate_frequencies(SB.F_MEASURED)
    return p,f_raw

def frf(p,res,ff):
    w=2*np.pi*ff; om=p.omega_n; z=p.zeta_modes
    den=(om[:,None]**2-w[None,:]**2)+2j*z[:,None]*om[:,None]*w[None,:]
    return np.abs(np.sum(res[:,None]/den,axis=0))

def extrema(ff,G):
    lg=np.log(G)
    mn=[ff[i] for i in range(1,len(G)-1) if lg[i]<lg[i-1] and lg[i]<lg[i+1]]
    return mn

ff=np.linspace(60,4900,120000)
P1=dict(SB.PATCH)                                   # v1 : bas-gauche 20x60
P2=dict(SB.PATCH); P2.update(x1=0.04,x2=0.10,z1=0.0,z2=0.02)   # v2 : bas-droit 60x20
ZV1=[0.0031,0.0017,0.0027,0.0030,0.0030]
ZDU=[0.0031,0.0017,0.0027,0.0056,0.0035]

print('Mesures Fig12a (point de frappe = capteur, coin sup droit) :')
print('  resonances   536.4 1071.8 2778.6 3358.7 4122.6')
print('  antires.     734 2161 3001 3805')
print('Mesures Fig12b (tension -> deplacement) :')
print('  antires.     788 1493 3609\n')

for name,P,Z in [('v1  patch bas-GAUCHE 20x60, zeta v1',P1,ZV1),
                 ('v2  patch bas-DROIT  60x20, zeta Du',P2,ZDU)]:
    p,f_raw=build(P,Z)
    H=p.H_Pe_modal; Do=p.D_obs; Dt=p.Dp_array[:,0]
    print('='*78); print(name)
    print('  f EF brutes (avant calibrage) :',np.round(f_raw,1))
    print('  article theorique             : [ 537. 1101. 2805. 3423. 4254.]')
    print('  H_Pe (N/V)   :',np.round(H,5))
    print('  D_obs        :',np.round(Do,3))
    print('  D_tool(x=0)  :',np.round(Dt,3))
    print('  signes H*D_obs :',np.sign(H*Do).astype(int))
    print('  -- antiresonances modele --')
    print('   drive-point  D_obs^2      :',[round(a) for a in extrema(ff,frf(p,Do*Do,ff))],'  <- a comparer a Fig12a')
    print('   transfert    D_obs*D_tool :',[round(a) for a in extrema(ff,frf(p,Do*Dt,ff))],'  <- ce que check_model utilise')
    print('   tension      D_obs*H      :',[round(a) for a in extrema(ff,frf(p,Do*H,ff))],'  <- a comparer a Fig12b')
    # niveaux aux pics
    Gd=frf(p,Do*Do,ff); Gv=frf(p,Do*H,ff)
    print('  -- niveaux aux resonances --')
    for i,fm in enumerate(SB.F_MEASURED):
        j=int(np.argmin(abs(ff-fm)))
        ja=int(np.argmin(abs(fa-fm))); jb=int(np.argmin(abs(fb-fm)))
        print(f'   {fm:7.1f} Hz | Fig12a mes {Aa[ja]:6.1f} dB  modele {20*np.log10(Gd[j]*1e6):6.1f} dB'
              f' | Fig12b mes {Ab[jb]:6.1f} dB  modele {20*np.log10(Gv[j]*1e6/0.01):6.1f} dB')

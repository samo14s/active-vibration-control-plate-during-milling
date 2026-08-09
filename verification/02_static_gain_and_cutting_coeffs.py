import numpy as np, openpyxl, glob, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
import simulation_base as SB
from simulation_base import SimBase

def load(f):
    ws=openpyxl.load_workbook(f).worksheets[0]
    d=np.array([[r[0],r[1]] for r in ws.iter_rows(values_only=True) if r[0] is not None],float)
    return d[:,0],d[:,1]
fa,Aa=load(glob.glob('experimental*')[0]); fb,Ab=load(glob.glob('transfer*')[0])

sim=SimBase(); p=sim.plate
H=p.H_Pe_modal; Do=p.D_obs; om=p.omega_n
print('--- gain statique (le plus fiable : loin des pics, insensible a zeta) ---')
gs_a=np.sum(Do*Do/om**2); gs_b=np.sum(Do*H/om**2)
m_a=np.mean(Aa[fa<140]); m_b=np.mean(Ab[fb<140])
print(f'  Fig12a  f<140Hz  mesure {m_a:6.2f} dB re 1um/N   -> {10**(m_a/20):8.2f} um/N')
print(f'          modele  Sum D^2/w^2                     -> {gs_a*1e6:8.2f} um/N '
      f'  ecart {20*np.log10(gs_a*1e6/10**(m_a/20)):+6.2f} dB')
print(f'  Fig12b  f<140Hz  mesure {m_b:6.2f} dB re 0.01um/V-> {10**(m_b/20)*0.01:8.4f} um/V')
print(f'          modele  Sum D*H/w^2                     -> {abs(gs_b)*1e6:8.4f} um/V '
      f'  ecart {20*np.log10(abs(gs_b)*1e6/(10**(m_b/20)*0.01)):+6.2f} dB')

print('\n--- Eq.14 telle qu ecrite (integrales en coord. NON DIMENSIONNELLES) ---')
lp,hp=SB.PLATE_L,SB.PLATE_H
print(f'  facteur x : lp/hp = {lp/hp:.3f}   facteur z : hp/lp = {hp/lp:.3f}')
print('  -> Eq.14 pondere differemment les deux courbures : non homogene.')
print('     le code utilise le laplacien physique (isotrope) = choix correct.')

print('\n--- verification des coefficients de coupe (Eq.3-4) ---')
from milling_force import cutting_coefficients, milling_force_coeffs
k1,k2=cutting_coefficients(SB.KN,SB.MU_C,SB.HELIX,SB.RAKE)
print(f'  k1 = kn/cos(eta)                     = {k1:.6f}  (attendu {0.26/np.cos(np.deg2rad(35)):.6f})')
print(f'  k2 = 1+muc*tan(eta)*(cos(g)-kn*sin(g))= {k2:.6f}  '
      f'(attendu {1+0.2*np.tan(np.deg2rad(35))*(np.cos(np.deg2rad(15))-0.26*np.sin(np.deg2rad(15))):.6f})')
# integration brute force de reference sur z
rpm=4900; Om=2*np.pi*rpm/60; NT=3; RT=0.005; eta=SB.HELIX
ae=0.1e-3; ap=0.3e-3
phi_st=np.pi-np.arccos(1-ae/RT); phi_ex=np.pi
zl,zh=SB.PLATE_H-ap,SB.PLATE_H
def ref(t,nz=200001):
    z=np.linspace(zl,zh,nz); ss=sc=cc=0.0
    for j in range(NT):
        th=Om*t+2*np.pi*j/NT-z*np.tan(eta)/RT
        thm=np.mod(th,2*np.pi)
        g=((thm>=phi_st)&(thm<=phi_ex)).astype(float)
        ss+=np.trapezoid(g*np.sin(th)**2,z); sc+=np.trapezoid(g*np.sin(th)*np.cos(th),z)
        cc+=np.trapezoid(g*np.cos(th)**2,z)
    return k2*SB.KT*ss-k1*SB.KT*sc, k2*SB.KT*sc-k1*SB.KT*cc
tau=60/(NT*rpm); errs=[]
for t in np.linspace(0,tau,9):
    a3,a4=milling_force_coeffs(t,Om,NT,RT,eta,phi_st,phi_ex,zl,zh,k1,k2,SB.KT)
    r3,r4=ref(t)
    errs.append((a3,r3,a4,r4))
print('   t/tau   alpha3 code / ref quad.        alpha4 code / ref quad.')
for i,(a3,r3,a4,r4) in enumerate(errs):
    print(f'   {i/8:.3f}   {a3:12.4f} / {r3:12.4f}     {a4:12.4f} / {r4:12.4f}')
A3=np.array([e[0] for e in errs]); R3=np.array([e[1] for e in errs])
A4=np.array([e[2] for e in errs]); R4=np.array([e[3] for e in errs])
print(f'   ecart max relatif : alpha3 {np.max(np.abs(A3-R3))/np.max(np.abs(R3))*100:.3f} %,'
      f'  alpha4 {np.max(np.abs(A4-R4))/np.max(np.abs(R4))*100:.3f} %')

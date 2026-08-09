import numpy as np, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
import simulation_base as SB
from plate_model import PlateModel
import kirchhoff_q4 as KQ

print('--- 1. quadrature de Gauss 5 pts de mass_matrix_K ---')
xg,wg=np.polynomial.legendre.leggauss(5)
print('  exact  abscisses',np.round(xg,6),' poids',np.round(wg,6),' somme',wg.sum())
print('  code   abscisses [-0.906 -0.538 0. 0.538 0.906]  poids [0.237 0.479 0.569 0.479 0.237] somme',
      0.237*2+0.479*2+0.569)
print(f'  -> masse surestimee de {(0.237*2+0.479*2+0.569)**2/4-1:+.4%} (produit tensoriel)')
Me=KQ.mass_matrix_K(SB.RHO,0.1/36,0.08/30,0.004)
print(f'  masse elementaire sum(Me)={Me.sum():.6e} vs rho*h*A={SB.RHO*0.004*(0.1/36)*(0.08/30):.6e}'
      f'  ecart {Me.sum()/(SB.RHO*0.004*(0.1/36)*(0.08/30))-1:+.4%}')

print('\n--- 2. laplace_n_patch : theoreme de la divergence ---')
N1,N2=SB.MESH_N1,SB.MESH_N2; lex,ley=SB.PLATE_L/N1,SB.PLATE_H/N2; n1=N1+1
ndof=3*n1*(N2+1)
g=KQ.laplace_n_patch(0.0,0.020,0.0,0.060,N1,N2,lex,ley,n1,ndof)
# reference : flux normal sur le contour, par differences finies sur w(x,y)
def w_of(dof,x,y):
    return KQ.shape_at_point(x,y,N1,N2,lex,ley,n1)[dof]
h=1e-6; tot=[]
for dof in [3*(15)*n1+3*8+0, 3*(10)*n1+3*3+1, 3*(5)*n1+3*2+2]:
    ny=400
    ys=np.linspace(0,0.060,ny); xs=np.linspace(0,0.020,ny)
    fx2=np.array([(w_of(dof,0.020+h,y)-w_of(dof,0.020-h,y))/(2*h) for y in ys])
    fx1=np.array([(w_of(dof,0.0+h,y)-w_of(dof,max(0.0-h,0.0),y))/(h if True else 2*h) for y in ys])
    fx1=np.array([(w_of(dof,0.0+2*h,y)-w_of(dof,0.0,y))/(2*h) for y in ys])
    fz2=np.array([(w_of(dof,x,0.060+h)-w_of(dof,x,0.060-h))/(2*h) for x in xs])
    fz1=np.array([(w_of(dof,x,0.0+2*h)-w_of(dof,x,0.0))/(2*h) for x in xs])
    ref=np.trapezoid(fx2,ys)-np.trapezoid(fx1,ys)+np.trapezoid(fz2,xs)-np.trapezoid(fz1,xs)
    tot.append((dof,g[dof],ref))
for dof,a,b in tot:
    print(f'  ddl {dof:5d} : g={a:+.6e}   flux contour={b:+.6e}   ecart rel {abs(a-b)/max(abs(b),1e-12):.2e}')

print('\n--- 3. convergence du maillage (frequences EF brutes, sans patch) ---')
for (a,b) in [(18,15),(24,20),(30,24),(36,30),(48,40),(60,50)]:
    p=PlateModel(SB.PLATE_L,SB.PLATE_H,SB.PLATE_T,SB.RHO,SB.YOUNG,SB.POISSON,
                 N1=a,N2=b,n_modes=5,zeta_modes=SB.ZETA_MODES,verbose=False)
    print(f'  {a}x{b:3d} : ',np.round(p.freq_n,2))
print('  article theorique (Chebyshev-Ritz) : [ 537. 1101. 2805. 3423. 4254.]')
print('  mesure Table 4                     : [ 540. 1068. 2787. 3351. 4122.]')

print('\n--- 4. inertie de rotation (terme I2 de Eq. A.1, absent du code) ---')
h=SB.PLATE_T; I0=SB.RHO*h; I2=SB.RHO*h**3/12
print(f'  I0={I0:.4f} kg/m2, I2={I2:.4e} kg.m')
for f in [536.4,1071.8,2778.6,3358.7,4122.6]:
    lam=np.sqrt(2*np.pi*f)*0  # indicatif
print('  correction relative ~ (I2/I0)*k^2 = (h^2/12)*k^2 ; pour le mode 5,')
print('  demi-longueur d onde ~ 20 mm -> k~157 rad/m -> (0.004^2/12)*157^2 =',
      f'{(0.004**2/12)*157**2:.4f}', ' soit -1.6 % sur omega^2 => -0.8 % sur f')

print('\n--- 5. patch structurel : quantification par elements ---')
lex,ley=SB.PLATE_L/SB.MESH_N1,SB.PLATE_H/SB.MESH_N2
nx=sum(1 for I in range(1,SB.MESH_N1+1) if 0-1e-9<=(I-0.5)*lex<=0.020+1e-9)
nz=sum(1 for J in range(1,SB.MESH_N2+1) if 0-1e-9<=(J-0.5)*ley<=0.060+1e-9)
print(f'  patch demande 20.0 x 60.0 mm ; couvert par {nx} x {nz} elements = '
      f'{nx*lex*1e3:.2f} x {nz*ley*1e3:.2f} mm  (aire {nx*lex*nz*ley/(0.020*0.060)-1:+.2%})')
print('  (le couplage H_Pe, lui, integre exactement sur le rectangle 20x60 :')
print('   raideur/masse ajoutees et couplage piezo ne portent pas sur la meme aire)')

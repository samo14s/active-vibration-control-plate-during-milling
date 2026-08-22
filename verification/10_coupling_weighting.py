"""La ponderation anisotrope de l'Eq. (14) reconcilie-t-elle le conflit ?

Le code utilise le laplacien physique (isotrope) ; l'Eq. (14) de l'article,
ecrite en coordonnees non dimensionnelles, pondere differemment les deux
courbures. Ce script balaie le rapport rho entre les deux contributions et
verifie l'occupation des creux. Resultat : non, pour aucun rho dans [0, 1].
"""
import numpy as np, sys
import os
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(_R,'simulation','sim_kit'),os.path.join(_R,'simulation')]
os.chdir(os.path.join(_R,'simulation','sim_kit'))
import simulation_base as SB
from plate_model import PlateModel
from kirchhoff_q4 import matrix_der_K
L,H=SB.PLATE_L,SB.PLATE_H
OCC=(1,1,0,1)

def gxx_gzz(xP1,xP2,zP1,zP2,N1,N2,lex,ley,n1,ndof):
    gx=np.zeros(ndof); gz=np.zeros(ndof); xg=np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    for J in range(max(1,int(np.floor(zP1/ley))+1),min(N2,int(np.ceil(zP2/ley)))+1):
        for I in range(max(1,int(np.floor(xP1/lex))+1),min(N1,int(np.ceil(xP2/lex)))+1):
            xl,xh=(I-1)*lex,I*lex; yl,yh=(J-1)*ley,J*ley
            a,b=max(xl,xP1),min(xh,xP2); c,d=max(yl,zP1),min(yh,zP2)
            if b<=a or d<=c: continue
            xm,xr=(a+b)/2,(b-a)/2; ym,yr=(c+d)/2,(d-c)/2
            Dof=np.r_[3*(J-1)*n1+3*(I-1)+np.arange(6),3*J*n1+3*(I-1)+np.arange(3,6),
                      3*J*n1+3*(I-1)+np.arange(3)]
            for ig in range(2):
                for jg in range(2):
                    xi=2*((xm+xr*xg[ig])-xl)/lex-1; et=2*((ym+yr*xg[jg])-yl)/ley-1
                    B=matrix_der_K(xi,et,lex,ley)
                    gx[Dof]+=-B[0,:]*(xr*yr); gz[Dof]+=-B[1,:]*(xr*yr)
    return gx,gz

def zeros_occ(r,om):
    n=len(r); s2=om**2; P=np.zeros(n)
    for i in range(n): P+=r[i]*np.poly(np.delete(s2,i))*((-1)**(n-1))
    z=np.roots(P); z=z[np.abs(z.imag)<=1e-6*max(np.abs(z.real).max(),1e-30)].real
    z=np.sort(np.sqrt(z[z>0]))/(2*np.pi); f=om/2/np.pi
    return tuple(int(((z>f[k])&(z<f[k+1])).sum()) for k in range(4)), z[(z>f[0])&(z<f[-1])]

print('H_Pe ∝ ∫∫(∂²N/∂x² + rho·∂²N/∂z²) dA    rho=1 : laplacien physique (code)')
print(f'                                        rho={(H/L)**2:.2f} : Eq.(14) telle qu ecrite\n')
print(f"{'patch':22s} {'f1(Hz)':>7s} {'rho':>5s} {'occupation':12s} {'ok':>4s} zeros en bande")
best=[]
for lab,(x1,w,z1,h) in [('V 20x60 x1=0mm  ',(0.00,0.020,0.0,0.060)),
                        ('V 20x60 x1=20mm ',(0.02,0.020,0.0,0.060)),
                        ('V 20x60 x1=80mm ',(0.08,0.020,0.0,0.060)),
                        ('H 60x20 x1=40mm ',(0.04,0.060,0.0,0.020))]:
    P=dict(SB.PATCH); P.update(x1=x1,x2=x1+w,z1=z1,z2=z1+h)
    p=PlateModel(L,H,SB.PLATE_T,SB.RHO,SB.YOUNG,SB.POISSON,N1=SB.MESH_N1,N2=SB.MESH_N2,
                 n_modes=5,zeta_modes=SB.ZETA_MODES,verbose=False)
    p.precompute_Dp(zp_pos=H-0.15e-3,n_pos=3); p.set_observation(L,H)
    p.add_piezo_patch(P['x1'],P['x2'],P['z1'],P['z2'],P['d31'],P['thickness'],P['E'],
                      P['nu'],G_adh=P['G_adh'],t_adh=P['t_adh'])
    f1=p.freq_n[0]
    gx,gz=gxx_gzz(P['x1'],P['x2'],P['z1'],P['z2'],SB.MESH_N1,SB.MESH_N2,p.lex,p.ley,p.n1,p.ndof)
    Hx=p.V.T@gx[p.DOFf]; Hz=p.V.T@gz[p.DOFf]
    p.calibrate_frequencies(SB.F_MEASURED); Do=p.D_obs
    for rho in [1.0,(H/L)**2,0.8,0.5,0.3,0.0]:
        occ,z=zeros_occ(Do*(Hx+rho*Hz),p.omega_n)
        ok = occ==OCC
        if ok: best.append((lab,f1,rho,z))
        print(f'{lab:22s} {f1:7.1f} {rho:5.2f} {str(occ):12s} {"OUI" if ok else "non":>4s} {np.round(z,0)}')
print('\nmesure : f1=540.0, occupation=(1,1,0,1), zeros=[788, 1493, 3609]')

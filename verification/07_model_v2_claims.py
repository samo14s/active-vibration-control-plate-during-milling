import numpy as np, sys
import os,sys
_R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_K=os.path.join(_R,'simulation','sim_kit')
sys.path[:0]=[_K,os.path.join(_R,'simulation')]
os.chdir(_K)
sys.path.insert(0,'sim_kit'); sys.path.insert(0,'.')
from model_v2 import make_sim
import simulation_base as SB

sim=make_sim(); p=sim.plate
H=np.asarray(p.H_Pe_modal).ravel(); Do=np.asarray(p.D_obs).ravel()
Dt=np.asarray(p.Dp_array[:,0]).ravel(); om=p.omega_n; z=p.zeta_modes
print('v2 : ||H|| =',round(np.linalg.norm(H),4),' (docstring annonce 0.5546)')
print('     signes H*D_obs =',np.sign(H*Do).astype(int))
ff=np.linspace(60,4900,200000)
def anti(res):
    w=2*np.pi*ff
    G=np.abs(np.sum(res[:,None]/((om[:,None]**2-w[None,:]**2)+2j*z[:,None]*om[:,None]*w[None,:]),axis=0))
    lg=np.log(G)
    return [round(ff[i]) for i in range(1,len(G)-1) if lg[i]<lg[i-1] and lg[i]<lg[i+1]]
print('     antires. tension  D_obs*H  =',anti(Do*H))
print('     docstring v2 annonce        = [787, 1493, 3609]  (motif 4/4)')
print('     mesure numerisee Fig.12(b)  = [788, 1493, 3609]')

print('\n--- check_model : quelle FRF comparer aux antiresonances de Fig.12(a) ? ---')
import importlib; importlib.reload(SB)
s1=SB.SimBase(); q=s1.plate
Do1=np.asarray(q.D_obs).ravel(); Dt1=np.asarray(q.Dp_array[:,0]).ravel()
om=q.omega_n; z=q.zeta_modes
def anti2(res):
    w=2*np.pi*ff
    G=np.abs(np.sum(res[:,None]/((om[:,None]**2-w[None,:]**2)+2j*z[:,None]*om[:,None]*w[None,:]),axis=0))
    lg=np.log(G); return np.array([ff[i] for i in range(1,len(G)-1) if lg[i]<lg[i-1] and lg[i]<lg[i+1]])
tgt=np.array([734.,2161.,3001.,3805.])
for lab,res in [('D_obs*D_tool (ce que fait check_model)',Do1*Dt1),
                ('D_obs*D_obs (point de frappe = Fig.12a)',Do1*Do1)]:
    a=anti2(res)
    e=[(min(abs(a-t))/t*100) for t in tgt]
    print(f'  {lab:42s} {np.round(a).astype(int)}  ecarts % {np.round(e,1)}  RMS {np.sqrt(np.mean(np.square(e))):.2f} %')

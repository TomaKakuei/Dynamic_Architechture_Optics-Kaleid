from camera import *
from sampling import sample
torch.set_num_threads(1);fit=WaveFit();cam=Camera();cal=cam.calibrate(fit.fit(np.zeros(40)),1024)
for case in (37,41):
 q=sample((case-37)%3,case)[0];rep=fit.fit(q)[4];print('CASE',case,'carrier',rep['carrier'],flush=True)
 images={n:cam.integrated_irradiance(rep,n) for n in (512,1024,2048)}
 start=clock();ref=cam.reference_irradiance(rep,n=4096,carrier_shift=True);print('REFERENCE SECONDS',clock()-start,flush=True)
 for n,im in images.items():
  sigma=np.sqrt(np.maximum(ref*cam.gain_e,0)+1.2**2);err=np.sqrt(np.mean(((im-ref)*cam.gain_e/sigma)**2))
  print('GRID',n,'ERR',float(err),flush=True)

from camera import *
from sampling import sample
def run():
 torch.set_num_threads(1);start=clock();fit=WaveFit();cam=Camera();q0=np.zeros(40);reps=fit.fit(q0)
 calibration=cam.calibrate(reps,1024);atomic(HERE/'camera_calibration_candidate.json',calibration)
 rows=[]
 states=[q0]+[sample(i%2,i+37)[0] for i in range(4)]
 for si,q in enumerate(states):
  t=clock();rep=fit.fit(q);fields=[0,2,4] if si else [0,1,2]
  for fi in fields:
   a=cam.integrated_irradiance(rep[fi],512);b=cam.integrated_irradiance(rep[fi],1024);c=cam.integrated_irradiance(rep[fi],2048)
   # Numerical error relative to shot+read noise for the fixed exposure.
   sigma=np.sqrt(np.maximum(c*cam.gain_e,0)+1.2**2)
   err512=float(np.sqrt(np.mean(((a-c)*cam.gain_e/sigma)**2)));err1024=float(np.sqrt(np.mean(((b-c)*cam.gain_e/sigma)**2)))
   rows.append(dict(state=si,field=fi,rms_noise_units_512=err512,rms_noise_units_1024=err1024,fit=rep[fi]['independent_fit_rms_waves']))
  print('CAMERA',si,clock()-t,rows[-3:],flush=True)
 # Fixed-map noise: different temporal draws with invariant exposure and PRNU.
 image=cam.integrated_irradiance(reps[0],1024);x=cam.noisy(image,1);y=cam.noisy(image,2)
 assert np.mean(x!=y)>.1
 grid=512 if max(r['rms_noise_units_512'] for r in rows)<.20 else 1024
 assert max(r['rms_noise_units_1024'] for r in rows)<.20,rows
 calibration=cam.calibrate(reps,grid);atomic(HERE/'camera_calibration.json',calibration)
 # Initial-score distribution using exactly the sampler law, no WFE rejection.
 model=Optics(grid=65,device='cuda');qs=np.array([sample(i%2,i)[0] for i in range(128)]);values=[];supports=[]
 with torch.no_grad():
  for a in range(0,128,8):
   w,s=model.metrics(torch.tensor(qs[a:a+8],dtype=torch.float64,device='cuda'));values.extend(w[:,0].cpu().tolist());supports.extend(s.min(1).values.cpu().tolist())
 distribution=dict(count=128,initial_passes=sum(v<.07 and s>=.98 for v,s in zip(values,supports)),mean=float(np.mean(values)),p50=float(np.median(values)),p90=float(np.quantile(values,.9)),min_support=min(supports))
 result=dict(status='pass',grid=grid,convergence=rows,initial_distribution=distribution,seconds=clock()-start)
 atomic(HERE/'camera_check.json',result);print('PASS',result,flush=True)
if __name__=='__main__':run()

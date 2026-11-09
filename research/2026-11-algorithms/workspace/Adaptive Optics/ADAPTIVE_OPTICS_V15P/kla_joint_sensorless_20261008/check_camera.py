from camera import *
from sampling import sample
def run():
 torch.set_num_threads(1);start=clock();fit=WaveFit();cam=Camera();reps=fit.fit(np.zeros(40));print('NOMINAL FIT',max(r['independent_fit_rms_waves'] for r in reps),flush=True)
 cal=cam.calibrate(reps,2048);rows=[]
 states=[np.zeros(40)]+[sample(i%3,i+37)[0] for i in range(6)]
 for si,q in enumerate(states):
  rep=fit.fit(q)
  for fi in (0,4,8):
   a=cam.integrated_irradiance(rep[fi],1024);b=cam.integrated_irradiance(rep[fi],2048);c=cam.reference_irradiance(rep[fi],4096,carrier_shift=True)
   sigma=np.sqrt(np.maximum(c*cam.gain_e,0)+1.2**2)
   rows.append(dict(state=si,field=fi,rms_noise_units_1024=float(np.sqrt(np.mean(((a-c)*cam.gain_e/sigma)**2))),rms_noise_units_2048=float(np.sqrt(np.mean(((b-c)*cam.gain_e/sigma)**2))),fit=rep[fi]['independent_fit_rms_waves']))
  print('CAMERA',si,rows[-3:],flush=True)
 atomic(HERE/'camera_convergence.json',dict(rows=rows))
 assert max(r['rms_noise_units_2048'] for r in rows)<.20,rows
 grid=2048
 cal=cam.calibrate(reps,grid);atomic(HERE/'camera_calibration.json',cal)
 image=cam.integrated_irradiance(reps[4],grid);x=cam.noisy(image,1);y=cam.noisy(image,2);assert np.mean(x!=y)>.1
 # Fresh temporal realizations verify the noise variance in electron units.
 stack=np.array([cam.noisy(image,k).astype(float)/4 for k in range(128)])
 expect=image*cam.gain_e*cam.prnu+1.2**2;active=expect>20
 ratio=float(np.mean(stack.var(0,ddof=1)[active]/expect[active]));assert .95<ratio<1.05,ratio
 # Include cases never used for phase fitting, not just the nominal camera.
 fits=[]
 for i in range(48):fits.extend(r['independent_fit_rms_waves'] for r in fit.fit(sample(i%3,300+i)[0]))
 result=dict(status='pass',grid=grid,convergence=rows,independent_phase_fit_max_waves=max(fits),noise_variance_ratio=ratio,seconds=clock()-start)
 atomic(HERE/'camera_check.json',result);print('PASS',json.dumps(result),flush=True)
if __name__=='__main__':run()

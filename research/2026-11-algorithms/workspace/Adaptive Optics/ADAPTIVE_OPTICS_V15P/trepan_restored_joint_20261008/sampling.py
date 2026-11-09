from common import *
from geometry import Geometry
def proposal(rng,cohort,index):
 # Four pre-registered amplitude patterns in every split. No optical merit
 # filtering, easy-case quotas or identical body amplitudes.
 mode=index%4;q=rng.uniform(-1,1,(8,5));a=rng.uniform(.2,1,8)
 if mode==0:
  a=rng.uniform(.03,.15,8);ids=rng.choice(8,int(rng.integers(1,4)),replace=False);a[ids]=rng.uniform(.4,1,len(ids))
 elif mode==1:a=rng.uniform(.4,1,8)
 elif mode==2:
  q=np.clip(.6*rng.uniform(-1,1,(1,5))+.4*q,-1,1);a=rng.uniform(.35,1,8)
 else:a=np.exp(rng.uniform(np.log(.08),0,8))
 # Front/rear cohorts differ in which branch contains one forced dominant body;
 # all eight bodies remain perturbed and are seen in the same acquisition.
 dominant=int(rng.integers(0,4))+4*cohort;a[dominant]=rng.uniform(.65,1)
 q*=a[:,None];axis=int(rng.integers(0,5));q[dominant,axis]=rng.choice([-1,1])*rng.uniform(.5,1)
 return q.ravel()
def sample(cohort,index):
 rng=np.random.default_rng(np.random.SeedSequence([20261008,cohort,index]));geo=Geometry()
 for tries in range(1000):
  q=proposal(rng,cohort,index)
  if geo.path(q):return q,tries
 raise RuntimeError('Mechanical sampler exhausted without a feasible pose')

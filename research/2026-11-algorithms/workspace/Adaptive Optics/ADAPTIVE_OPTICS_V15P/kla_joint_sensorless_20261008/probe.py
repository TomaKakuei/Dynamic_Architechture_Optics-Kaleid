from common import *
from optics import *
from geometry import Geometry
from sampling import sample
torch.set_num_threads(1)
print('geometry',Geometry().clearances.tolist(),flush=True)
print('samples',[(i,sample(i%3,i)[1]) for i in range(12)],flush=True)
qs=np.vstack([np.zeros(40)]+[sample(i%3,i)[0] for i in range(12)])
for g in (33,65,129):
 start=clock();o=Optics(grid=g,device='cuda');rows=[]
 with torch.no_grad():
  for q in qs:rows.append({k:v.cpu().tolist() for k,v in o.metrics(torch.tensor(q[None],dtype=torch.float64,device='cuda')).items()})
 print('GRID',g,'SECONDS',clock()-start,'ROWS',json.dumps(rows),flush=True)

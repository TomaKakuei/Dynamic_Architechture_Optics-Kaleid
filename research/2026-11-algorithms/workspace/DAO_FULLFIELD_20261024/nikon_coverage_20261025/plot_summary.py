"""Plot saved coverage statistics only; no further model computation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'coverage.json').read_text())
c=json.loads((HERE/'conditional_summary.json').read_text())
with np.load(HERE/'pose_coverage_arrays.npz') as z:
    arrays={k:z[k].copy() for k in z.files}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                    'axes.spines.right':False,'axes.titleweight':'bold'})
blue,orange,gray='#176b9a','#d67732','#8d99a6'
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
x=np.arange(2,8)
ax=axes[0,0]
for offset,key,total,color,label in [(-.24,'original_train',24000,gray,'Original train (24k)'),
    (0,'all_train',28000,blue,'Train + 4k (28k)'),(.24,'validation',6000,orange,'Validation (6k)')]:
    ax.bar(x+offset,[100*row[key]/total for row in r['activity']],width=.23,color=color,label=label)
ax.set(title='A  Active-body distributions',xlabel='Simultaneously perturbed rigid bodies',ylabel='Samples (%)',xticks=x)
ax.legend(frameon=False,fontsize=8)
ax=axes[0,1]
for key,color,label in [('calibration_distance',blue,'Train queries, self excluded'),('distance',orange,'Validation queries')]:
    values=np.sort(arrays[key]);ax.plot(values,np.arange(1,len(values)+1)/len(values),color=color,lw=2,label=label)
ax.set(title='B  Matched-pattern pose coverage',xlabel='Nearest-neighbor RMS distance per active coordinate',ylabel='Cumulative fraction')
ax.legend(frameon=False,fontsize=8)
ax.text(.04,.72,'Medians: train 0.517 / validation 0.516\nSame training reference; activity masks matched',
        transform=ax.transAxes,fontsize=9)
ax=axes[1,0]
for offset,key,color,label in [(-.18,'old_passes',gray,'Previous E3'),(.18,'passes',blue,'Additive E4')]:
    heights=[100*row[key]/row['validation'] for row in r['activity']]
    ax.bar(x+offset,heights,width=.35,color=color,label=label)
for n,row in zip(x,r['activity']):
    ax.text(n+.18,100*row['rate']+1.5,f"{100*row['rate']:.1f}%",ha='center',fontsize=9)
ax.set(title='C  Recovery drops with active-body count',xlabel='Simultaneously perturbed rigid bodies',
       ylabel='Original three-field recovery (%)',xticks=x,ylim=(0,100))
ax.legend(frameon=False,fontsize=8)
ax=axes[1,1]
selected=[row for row in c if row['active']>=4]
xx=[row['active'] for row in selected];yy=[100*row['common_rms_rate'] for row in selected]
ax.bar(xx,yy,color=blue,width=.6)
for n,y,row in zip(xx,yy,selected):
    ax.text(n,y+1,f"{y:.1f}%\nn={row['common_rms_count']}",ha='center',fontsize=9)
ax.set(title='D  Same pose-amplitude band',xlabel='Simultaneously perturbed rigid bodies',
       ylabel='Recovery (%)',xticks=xx,ylim=(0,42))
ax.text(.04,.93,'Normalized whole-pose RMS: 0.45–0.55',transform=ax.transAxes,fontsize=9)
fig.suptitle('Nikon coverage: no large train–validation shift detected',fontsize=16,fontweight='bold')
fig.savefig(HERE/'coverage_summary.png',dpi=180)
fig.savefig(HERE/'coverage_summary.pdf')
plt.close(fig)

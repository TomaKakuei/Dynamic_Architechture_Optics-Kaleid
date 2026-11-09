from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'evaluation6000_4plus4/result.json').read_text())
a=json.loads((HERE/'ablation.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[1,1.15]})
x=np.arange(6);keys=[str(i) for i in range(2,8)]
old=[100*r['active_bodies'][k]['baseline']['rate'] for k in keys]
new=[100*r['active_bodies'][k]['candidate']['rate'] for k in keys]
ax=axes[0];ax.bar(x-.18,old,.36,label='Original 4 steps',color='#7488a3')
ax.bar(x+.18,new,.36,label='4 native + 4 full packet',color='#138477')
ax.set(xticks=x,xticklabels=keys,ylim=(0,111),ylabel='Recovery (%)',xlabel='Active Nikon bodies',title='Original 6,000 validation cases')
for i,v in enumerate(new):ax.text(i+.18,v+1.5,f'{v:.1f}',ha='center',fontsize=8)
ax.legend(frameon=False,loc='lower left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axes[1]
names=['Original 4 steps','Native only, 16 steps','+ center ABS, 8 steps','+ outer DIFF, 8 steps','Full packet, 8 steps']
counts=[8,10,a['routes']['add_centerABS']['passes'],a['routes']['add_diff_0p1_0p15']['passes'],48]
y=np.arange(len(names));ax.barh(y,counts,color=['#7488a3','#7488a3','#5bb1a5','#5bb1a5','#138477'])
for i,v in enumerate(counts):ax.text(v+.6,i,f'{v}/48',va='center',fontsize=9)
ax.set(yticks=y,yticklabels=names,xlim=(0,57),xlabel='Recovered cases',title='Fixed 48 seven-body diagnostic cases');ax.invert_yaxis()
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Nikon bottleneck: use the complete observation packet in the inverse solve',fontsize=14,x=.51)
fig.text(.5,.015,'Frozen network; one acquisition; no new training. Center ABS + edge DIFF, all three < 0.07 waves.\nThe 8-step candidate changes model computation, not the number of measured observation packets.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,.93));fig.savefig(HERE/'bottleneck_summary.png',dpi=180);fig.savefig(HERE/'bottleneck_summary.pdf');plt.close(fig)

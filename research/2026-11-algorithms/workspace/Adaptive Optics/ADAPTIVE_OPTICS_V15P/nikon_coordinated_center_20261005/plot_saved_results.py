"""Plot archived metrics only; no model loading, tracing, or training."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
current = json.loads((HERE / 'curves.json').read_text(encoding='utf-8'))['epochs']
old_data = json.loads((HERE.parent / 'expanded_differentiable_20261005/nikon/curves.json').read_text(encoding='utf-8'))
old = old_data['epochs']
selected = json.loads((HERE / 'selected.json').read_text(encoding='utf-8'))['epoch']
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 11,
                     'axes.unicode_minus': False, 'pdf.fonttype': 42})
blue, orange, green = '#2864B5', '#DA7228', '#27916A'
ep = [r['epoch'] for r in current]
metric = lambda rows, section, key: [r[section]['overall'][key] for r in rows]

fig, ax = plt.subplots(2, 2, figsize=(13.3, 8.6))
fig.suptitle('Nikon：50 轮训练与单次修正恢复率', fontsize=18, y=.98)
ax[0, 0].plot(ep[1:], [r['train_loss'] for r in current[1:]], color=blue, lw=2, label='总损失')
ax[0, 0].plot(ep[1:], [r['physical_loss'] for r in current[1:]], color=orange, lw=1.5, ls='--', label='中心光学损失')
ax[0, 0].set(title='训练目标（每轮全部 24000 个训练样本）', ylabel='损失（无量纲）')
ax[0, 0].legend()
ax[0, 1].plot(ep, [100*v for v in metric(current, 'train_probe', 'recovery_rate')], color=blue, lw=2, label='固定训练抽查：1500 个')
ax[0, 1].plot(ep, [100*v for v in metric(current, 'validation', 'recovery_rate')], color=orange, lw=2, label='验证：6000 个')
ax[0, 1].axhline(90, color='#666666', ls=':', label='90% 目标')
chosen = next(r for r in current if r['epoch'] == selected)
ax[0, 1].scatter([selected], [100*chosen['validation']['overall']['recovery_rate']], color=orange, zorder=4)
ax[0, 1].annotate('第 49 轮：88.03%', xy=(49, 88.0333), xytext=(26, 50), arrowprops={'arrowstyle':'->','color':orange}, color=orange)
ax[0, 1].set(title='单次修正：中心 WRMS < 0.07λ', ylabel='达标率（%）', ylim=(0, 101))
ax[0, 1].legend(loc='lower right', fontsize=9)
for g, c in zip(('G1', 'G2', 'G3'), (blue, orange, green)):
    rate = [100*r['validation']['groups'][g]['recovery_rate'] for r in current]
    val = 100*chosen['validation']['groups'][g]['recovery_rate']
    ax[1, 0].plot(ep, rate, color=c, lw=2, label=f'{g} 来源组：第49轮 {val:.2f}%')
ax[1, 0].axhline(90, color='#666666', ls=':')
ax[1, 0].set(title='按采样来源分组（每组 2000 个验证样本）', ylabel='达标率（%）', ylim=(0, 101))
ax[1, 0].legend(fontsize=9, loc='lower right')
for key, label, c in [('center_mean','验证均值',blue), ('center_p90','验证 P90',orange)]:
    ax[1, 1].plot(ep, metric(current, 'validation', key), color=c, lw=2, label=label)
ax[1, 1].axhline(.07, color='#666666', ls=':', label='0.07λ 门槛')
ax[1, 1].set(title='单次修正后的中心波前误差', ylabel='WRMS（λ；去 piston / tilt / defocus）')
ax[1, 1].legend()
for a in ax.flat:
    a.set_xlabel('本轮训练 epoch')
    a.grid(alpha=.18)
    a.set_xlim(0, 50)
    a.spines[['top','right']].set_visible(False)
fig.text(.05, .02, '原始带噪输入（σ=0.015λ）；同一验证集逐轮评分；epoch 0 继承上一轮 epoch 29。\nG1/G2/G3 是不同难度的采样来源，不是三个网络各自的准确率；本图仅重绘已保存记录。', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .07, 1, .95))
fig.savefig(HERE / 'curves_readable.png', dpi=180)
fig.savefig(HERE / 'curves_readable.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6.3))
ax.plot(ep, [100*v for v in metric(current,'validation','recovery_rate')], color=orange, lw=2.4, label='本轮：协调模块 + 原始节点 + 中心损失（50轮）')
ax.plot([r['epoch'] for r in old], [100*v for v in metric(old,'validation','recovery_rate')], color=blue, lw=2.2, label='上一轮可微方案（30轮；原联合目标）')
ax.axhline(90, color='#666666', ls=':', label='90% 目标')
ordinary_rate = old_data['baselines']['ordinary']['overall']['recovery_rate'] * 100
ax.axhline(ordinary_rate, color='#777777', ls='--', lw=1.3, label=f'普通版冻结 checkpoint：{ordinary_rate:.2f}%（参考点）')
ax.scatter([49,29],[88.0333333333,31.6833333333],color=[orange,blue],zorder=4)
ax.annotate('88.03%',(49,88.0333),xytext=(42,74),arrowprops={'arrowstyle':'->','color':orange},color=orange)
ax.annotate('31.68%',(29,31.6833),xytext=(34,38),arrowprops={'arrowstyle':'->','color':blue},color=blue)
ax.set(title='同一 6000 个带噪验证样本：历史单次恢复率', xlabel='各阶段自身的训练 epoch', ylabel='中心 WRMS < 0.07λ 的比例（%）', xlim=(0,50), ylim=(0,101))
ax.grid(alpha=.18)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower right',fontsize=9)
fig.text(.09,.03,'橙线从蓝线第29轮继续训练；起点、目标、架构和预算均不同，不能作为等预算消融。\n普通版仅有本次同集冻结评估值，虚线不是普通版逐轮训练曲线；不同损失不叠画比较。',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.10,1,1))
fig.savefig(HERE / 'historical_recovery_comparison.png', dpi=180)
fig.savefig(HERE / 'historical_recovery_comparison.pdf')
plt.close(fig)
print(json.dumps({'status':'saved_metrics_replotted','training_updates':0,'outputs':['curves_readable.png','curves_readable.pdf','historical_recovery_comparison.png','historical_recovery_comparison.pdf']}))

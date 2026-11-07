import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/d8-v5'); a=json.loads((p/'audit/audit.json').read_text()); c=json.loads((p/'artifacts/comparison.json').read_text()); t=a['totals']; rate=4.43/3600
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
labels=['Training incl. interim eval','Final validation + test','Inference benchmark','Other overhead']
projected=[t['projected_training_seconds'],t['projected_final_evaluation_seconds'],t['projected_benchmark_seconds'],t['projected_overhead_seconds']]
actual=[t['training_seconds'],t['final_evaluation_seconds'],t['benchmark_seconds'],t['actual_other_seconds']]
for y,vals in enumerate([projected,actual]):
 left=0
 for label,value,color in zip(labels,vals,['#345a84','#739dce','#bc8c36','#cccccc']):
  cost=value*rate;ax.barh(y,cost,left=left,color=color,label=label if y==0 else None)
  if cost>.5:ax.text(left+cost/2,y,f'${cost:.2f}',ha='center',va='center',color='white' if color!='#cccccc' else '#222')
  left+=cost
 ax.text(left+.15,y,f'${left:.2f}',va='center',weight='bold')
ax.set(yticks=[0,1],yticklabels=['Pilot projection','Measured run pipelines'],xlabel='USD at the configured $4.43 per node-hour',xlim=(0,21),title='All 14 runs completed; the cost estimator overestimated overhead')
ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False);fig.savefig(p/'audit/cost_reconciliation.png',dpi=180);plt.close(fig)
results={r['run_id']:r for r in c['runs']}; groups=[('B0 (3 seeds)',['d8-b0-s1','d8-b0-s2','d8-b0-s3']),('GQA-2 (3 seeds)',['d8-gqa2-s1','d8-gqa2-s2','d8-gqa2-s3']),('GQA-2 R75',['d8-gqa2-r75']),('GQA-2 R25',['d8-gqa2-r25']),('GQA-2 parameter matched',['d8-gqa2-pm']),('GQA-4',['d8-gqa4']),('B0 without value embeddings',['d8-b0-nove']),('GQA-2 without value embeddings',['d8-gqa2-nove']),('GQA-2 matched tokens',['d8-gqa2-tok']),('GQA-4 matched tokens',['d8-gqa4-tok'])]
fig,ax=plt.subplots(figsize=(10,6.5),layout='constrained')
for i,(label,ids) in enumerate(groups):
 vals=[results[k]['test_bpb'] for k in ids]; mean=st.mean(vals)
 ax.scatter(vals,[i]*len(vals),color='#9db4ce',s=28,zorder=3);ax.scatter([mean],[i],marker='D',color='#345a84',s=60,zorder=4);ax.text(mean+.0008,i,f'{mean:.5f}',va='center')
ax.axvline(st.mean(results[k]['test_bpb'] for k in groups[0][1]),color='#888',ls=':',lw=1)
ax.set(yticks=range(len(groups)),yticklabels=[g[0] for g in groups],xlabel='Final held-out test bits per byte — lower is better',xlim=(.958,.986),title='GQA trades quality for lower KV-cache memory at d8')
ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.savefig(p/'audit/quality_summary.png',dpi=180);plt.close(fig)
base=json.loads((p/'artifacts/d8-b0-s1/inference.json').read_text())['rows'];fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
for ax,prompt in zip(axes,[128,1792,8192]):
 for rid,label,color in [('d8-gqa2-s1','GQA-2 (seed 1)','#bc8c36'),('d8-gqa4','GQA-4 (seed 1)','#735197')]:
  inf=json.loads((p/'artifacts'/rid/'inference.json').read_text())['rows'];xs=[];ys=[]
  for x,b in zip(inf,base):
   if x['prompt_tokens']==prompt:xs.append(x['batch_size']);ys.append(x['decode_tokens_per_second_aggregate']/b['decode_tokens_per_second_aggregate'])
  ax.plot(xs,ys,'o-',label=label,color=color)
 ax.axhline(1,color='#888',ls=':');ax.set(xscale='log',xticks=[1,4,16,64],xticklabels=['1','4','16','64'],ylim=(.8,1.05),xlabel='Batch size',title=f'{prompt:,}-token prompt');ax.grid(alpha=.15)
axes[0].set_ylabel('Decode speed / B0 seed 1');axes[2].legend(frameon=False,loc='lower left');fig.suptitle('KV-cache savings did not produce faster eager decoding');fig.savefig(p/'audit/decode_speed.png',dpi=180);plt.close(fig)

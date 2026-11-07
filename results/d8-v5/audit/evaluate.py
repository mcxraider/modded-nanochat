"""Read-only audit of exported v5 evidence. Run from repository root with python3."""
import sys, json, math, statistics as st, hashlib, csv, re, tarfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.config import RUNS, NODE_HOUR_USD
from experiments.report import comparison_eligibility, check_consistency, tables, seed_noise, paired_analysis, pooled_noise
from experiments.resume import project_seconds, pilot_tokens_per_second
root=Path('results/d8-v5'); art=root/'artifacts'; out=root/'audit'
def read(p): return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def check(condition, message):
 if not condition: failures.append(message)
failures=[]; loaded={}; summaries=[]; warnings=[]
for rid in RUNS:
 d=art/rid
 item={k:read(d/(k+'.json')) for k in ['config','result','launch','hardware','data_manifest','inference']}; loaded[rid]=item
 c,r,h,inf=item['config'],item['result'],item['hardware'],item['inference']; t=c['training']
 rows=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()]; tr=[x for x in rows if x['kind']=='train']; n=t['target_tokens']//t['total_batch_size']
 check(comparison_eligibility(r,c)['eligible'],rid+': eligibility')
 check(c==read(Path('experiments/configs')/(rid+'.json')),rid+': config differs locally')
 check([x['step'] for x in tr]==list(range(1,n+1)),rid+': non-contiguous steps')
 check(all(x['tokens']==x['step']*t['total_batch_size'] and x['estimated_flops']==x['tokens']*r['flops_per_token'] for x in tr),rid+': metric token/FLOP arithmetic')
 check(all(math.isfinite(v) for x in rows for v in x.values() if isinstance(v,(int,float))),rid+': nonfinite metric')
 check(r['step']==r['checkpoint_step']==n,rid+': final step')
 check(r['estimated_flops']==r['tokens']*r['flops_per_token'],rid+': final FLOPs')
 interim=[x for x in rows if x['kind']=='validation' and not x.get('final')]
 check([x['step'] for x in interim]==list(range(t['eval_every'],n,t['eval_every'])),rid+': interim evaluations')
 ev=read(d/f'evaluation_{n:06d}.json')
 for kind,field in [('validation','val_bpb'),('test','test_bpb')]:
  finals=[x for x in rows if x['kind']==kind and x.get('final')]
  check(len(finals)==1 and finals[0][field]==r[field],rid+': final '+kind)
 check(ev['val']['bpb']==r['val_bpb'] and ev['test']['bpb']==r['test_bpb'],rid+': evaluation ledger')
 check(h['attention_backend']==inf['attention_backend']=='FA3' and h['world_size']==1 and 'H100' in h['gpu'],rid+': hardware')
 speed=st.median(x['tokens_per_second'] for x in tr if x['step']>20); wait=st.median(x['data_wait_seconds']/x['step_seconds'] for x in tr if x['step']>20)
 check(speed==r['steady_tokens_per_second'] and wait==r['steady_data_wait_fraction'],rid+': steady summaries')
 check(wait<.02,rid+': loader wait')
 check(max(r['peak_vram_bytes'],r['eval_peak_vram_bytes'])<.9*80e9,rid+': memory')
 check({(x['batch_size'],x['prompt_tokens']) for x in inf['rows']}=={(b,p) for b in c['benchmark']['batch_sizes'] for p in c['benchmark']['prompt_lengths']} and len(inf['rows'])==12,rid+': benchmark grid')
 check(inf['warmup']==2 and inf['repeats']==5,rid+': benchmark repeats')
 for x in inf['rows']:
  check(x['status']=='ok',rid+': benchmark failure')
  if x['status']!='ok':continue
  analytic=2*c['model']['n_layer']*x['batch_size']*(x['prompt_tokens']+256)*c['model']['n_kv_head']*(c['model']['n_embd']//c['model']['n_head'])*2
  check(x['kv_cache_bytes_allocated']==x['kv_cache_bytes_analytic']==analytic,rid+': cache size')
  check(x['decode_working_set_bytes']<=.5*analytic+64*2**20,rid+': expanded decode cache')
  check(x['generated_tokens_per_request']==256 and x['decode_timed_tokens_per_request']==255,rid+': decode length')
 for p,sha in item['launch']['overlay_sha256'].items():
  
  if (d/'source'/p).exists(): check(digest(d/'source'/p)==sha,rid+': saved source hash '+p)
  else:
   recovered=root/'export/recovered-source'/p
   check(recovered.exists() and digest(recovered)==sha,rid+': recovered source hash '+p)
   warnings.append({'run_id':rid,'missing_source_snapshot':p,'recovered_hash_matches':recovered.exists() and digest(recovered)==sha})
 order=read(d/'data_order.json'); expected=[f'shard_{i:05d}.parquet' for i in range(48)]; rotation=(c['seed']-1)*16
 check(order['train_order']==expected[rotation:]+expected[:rotation],rid+': seed data order')
 logs=(d/'train.log').read_text()+'\n'+(d/'benchmark.log').read_text()
 hits=[l for l in logs.splitlines() if re.search(r'Traceback|\b(?:WARNING|ERROR|Error|Exception|nan|NaN|OOM)\b',l)]
 if hits:warnings.append({'run_id':rid,'lines':hits})
 accounting=read(d/'rental_accounting.json')
 check(len(accounting['launches'])==1 and accounting['launches'][0]['status']=='complete',rid+': retries/failure')
 pilot=next(p for p in (root if (root/'pilots').exists() else Path('results/d8-v5-pilots')).glob('pilots/*/pilot-*') if read(p/'config.json')['model']==c['model'])
 pr=read(pilot/'result.json'); pi=read(pilot/'inference.json'); ps=pilot_tokens_per_second(pilot/'metrics.jsonl')
 proj=project_seconds(c,ps,262144/pr['evaluation_seconds'],pi['elapsed_seconds']*3.5)
 actual=accounting['total_pipeline_node_seconds']
 summaries.append(dict(run_id=rid,steps=n,tokens=r['tokens'],test_bpb=r['test_bpb'],val_bpb=r['val_bpb'],steady_tokens_per_second=speed,data_wait_fraction=wait,training_seconds=r['training_seconds'],final_evaluation_seconds=r['evaluation_seconds'],benchmark_seconds=inf['elapsed_seconds'],pipeline_seconds=actual,pipeline_cost_usd=accounting['cost_usd'],projected_seconds=proj['total'],projected_training_seconds=proj['training'],projected_final_evaluation_seconds=(c['evaluation']['final_tokens']+c['evaluation']['test_tokens'])/(262144/pr['evaluation_seconds']),projected_benchmark_seconds=pi['elapsed_seconds']*3.5*1.25,projected_overhead_seconds=420,actual_other_seconds=actual-r['training_seconds']-r['evaluation_seconds']-inf['elapsed_seconds'],loss_first=tr[0]['loss'],loss_last=tr[-1]['loss']))
tokenizer_manifest=next(iter(loaded.values()))['data_manifest']['tokenizer']
for name,sha in tokenizer_manifest.items(): check(digest(root/'export/tokenizer'/name)==sha,'exported tokenizer '+name)
preflight=read(root/'pilots/9a34115f44463cc7/preflight.json')
check(preflight['status']=='passed' and len(preflight['variants'])==8 and all(v['status']=='passed' for v in preflight['variants']),'numerical preflight')
check_consistency(loaded)
comparison=read(art/'comparison.json'); results={k:v['result'] for k,v in loaded.items()}
check(comparison['missing_runs']==[], 'missing runs')
check(comparison['noise']==seed_noise([results[f'd8-b0-s{i}']['test_bpb'] for i in (1,2,3)]),'baseline noise')
check(comparison['paired']==paired_analysis(results,comparison['noise']),'paired statistics')
check(comparison['pooled_noise']==pooled_noise(results),'pooled statistics')
check((art/'tables.md').read_text()==tables(comparison),'tables regeneration')
check({x['run_id']:x for x in comparison['runs']}==results,'comparison run records')
for cp in comparison['comparisons']:
 a=st.mean(results[k]['test_bpb'] for k in cp['members']); b=st.mean(results[k]['test_bpb'] for k in cp['reference_members'])
 check(math.isclose(cp['test_bpb_delta'],a-b,abs_tol=1e-15),cp['run_id']+': comparison arithmetic')
# Capture all final checkpoint member hashes without deserializing pickle-based tensors.
archive=(root/'export/checkpoints.tar' if (root/'export/checkpoints.tar').exists() else Path('results/d8-v5-checkpoints.tar')); checkpoint_checks=[]
if archive.exists() and archive.stat().st_size==9700812800:
 with tarfile.open(archive) as tf:
  ms=tf.getmembers(); byname={m.name:m for m in ms}
  for rid in RUNS:
   n=results[rid]['step']; prefix=f'artifacts/{rid}/checkpoints/'
   timing=json.load(tf.extractfile(prefix+f'timing_{n:06d}.json'))
   meta=json.load(tf.extractfile(prefix+f'meta_{n:06d}.json'))
   check(meta['step']==n and meta['tokens']==results[rid]['tokens'] and meta['experiment_config']==loaded[rid]['config'],rid+': checkpoint metadata')
   for name,sha in timing['sha256'].items():
    stream=tf.extractfile(prefix+name); hh=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024**2),b''):hh.update(block)
    ok=hh.hexdigest()==sha;check(ok,rid+': checkpoint checksum '+name)
    checkpoint_checks.append(dict(run_id=rid,file=name,bytes=byname[prefix+name].size,sha256=hh.hexdigest(),valid=ok))
else:print('Checkpoint download incomplete; rerun after download.')
with (out/'run_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
payload=dict(failures=failures,log_warnings=warnings,run_count=len(summaries),total_updates=sum(x['steps'] for x in summaries),total_tokens=sum(x['tokens'] for x in summaries),runs=summaries,checkpoint_checks=checkpoint_checks,totals={k:sum(x[k] for x in summaries) for k in summaries[0] if k.endswith('_seconds') or k=='pipeline_cost_usd'})
(out/'audit.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({k:v for k,v in payload.items() if k not in ['runs','checkpoint_checks']},indent=2))

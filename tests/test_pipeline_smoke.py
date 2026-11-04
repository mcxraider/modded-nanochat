"""End-to-end CPU smoke test of the actual pinned training runtime."""
import json, os, random, shutil, string, subprocess, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from experiments.run import prepare_runtime, ROOT
from experiments.make_configs import build
from experiments.checkpoints import digest, valid
from nanochat.tokenizer import RustBPETokenizer

import pytest

@pytest.mark.parametrize('value_embeds', [True, False])
def test_pinned_training_completed_resume_preserves_saved_evidence(tmp_path, value_embeds):
    root=tmp_path; runtime=root/'runtime'
    try:
        _, hashes=prepare_runtime(runtime)
        _, hashes2=prepare_runtime(runtime)
        assert hashes == hashes2
        base=root/'data'; data=base/'base_data_climbmix'; data.mkdir(parents=True)
        rng=random.Random(14)
        docs=[' '.join(''.join(rng.choices(string.ascii_lowercase,k=8)) for _ in range(15)) for _ in range(1200)]
        tok=RustBPETokenizer.train_from_iterator(iter(docs),512)
        tok.save(str(base/'tokenizer'))
        torch.save(torch.ones(512,dtype=torch.int32),base/'tokenizer/token_bytes.pt')
        for index in (0,1,2,6541,6542):
            pq.write_table(pa.table({'text':docs}),data/f'shard_{index:05d}.parquet',row_group_size=1200)
        cfg=build('d8-b0-s1'); cfg['run_id']='cpu-smoke'
        cfg['model'].update(sequence_len=16,vocab_size=512,n_layer=2,n_embd=64,n_head=2,n_kv_head=1,value_embeds=value_embeds)
        cfg['training'].update(target_tokens=64,total_batch_size=32,device_batch_size=1,save_every=1,eval_every=1,compile=False)
        cfg['evaluation'].update(interim_tokens=32,final_tokens=32,test_tokens=32)
        cfg['benchmark'].update(batch_sizes=[1],prompt_lengths=[16],decode_tokens=2)
        output=root/'result';output.mkdir();path=output/'config.json';path.write_text(json.dumps(cfg))
        env=dict(os.environ,NANOCHAT_BASE_DIR=str(base),TORCH_COMPILE_DISABLE='1',OMP_NUM_THREADS='1')
        cmd=[sys.executable,'-m','experiments.train','--config',str(path),'--output',str(output),'--device-type','cpu']
        first=subprocess.run(cmd,cwd=runtime,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if first.returncode: raise RuntimeError(first.stdout)
        assert valid(output/'checkpoints',2,1)
        before={p.name:digest(p) for p in (output/'checkpoints').iterdir() if p.is_file()}
        evaluation=(output/'evaluation_000002.json').read_bytes()
        second=subprocess.run(cmd+['--resume','2'],cwd=runtime,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if second.returncode: raise RuntimeError(second.stdout)
        assert before=={p.name:digest(p) for p in (output/'checkpoints').iterdir() if p.is_file()}
        assert evaluation==(output/'evaluation_000002.json').read_bytes()
        resumed_hardware=json.loads((output/'hardware_resume_000002.json').read_text())
        assert resumed_hardware['resume_replay_batches'] == 0
        result=json.loads((output/'result.json').read_text())
        assert result['steady_tokens_per_second'] is None and result['eval_peak_vram_bytes'] == 0
        # Mid-run resume from the retained step-1 fallback replays data and must reproduce step 2 exactly.
        mid=root/'mid'; shutil.copytree(output,mid,ignore=shutil.ignore_patterns('*000002*','result.json','training_complete.json'))
        third=subprocess.run(cmd[:-4]+['--output',str(mid),'--device-type','cpu','--resume','1'],cwd=runtime,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if third.returncode: raise RuntimeError(third.stdout)
        assert json.loads((mid/'hardware_resume_000001.json').read_text())['resume_replay_batches'] == 2  # step 1 * accum 2
        a,b=(torch.load(d/'checkpoints/model_000002.pt',map_location='cpu') for d in (output,mid))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    finally:
        if runtime.exists():
            subprocess.run(['git','worktree','remove','--force',str(runtime)],cwd=ROOT,check=True)

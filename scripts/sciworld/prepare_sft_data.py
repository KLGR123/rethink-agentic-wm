"""Prepare audited whole-episode, action-only SFT data and optional-Thought control."""
import collections
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[2]/'datasets/sciworld/sft',help='Published dataset; raw demonstrations remain in --data-dir')
p.add_argument('--model',default='/fs_mol/liujiarun/models/qwen2.5-3b-instruct')
p.add_argument('--seed',type=int,default=20261029)
a=p.parse_args();D=a.data_dir.resolve()
OUT=a.output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
assert not any((OUT/name).exists() for name in ('train.parquet','train.pending.parquet','metadata.json','splits.json')), 'Use a fresh output directory; never overwrite a published dataset'
assert json.loads((D/'collection.json').read_text())['status']=='collection_complete'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sciworld.sft_prompts import build_system_message,build_obs_user_message,nearby_commands
from transformers import AutoTokenizer
import pandas as pd
from omegaconf import OmegaConf
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

tok=AutoTokenizer.from_pretrained(a.model)
manifest=json.loads((D/'manifest.json').read_text())
allowed={(r['task_name'],r['variation_idx']) for r in manifest['candidates']}
rows=[];audit=[]
for p in sorted((D/'raw').glob('*.json')):
    d=json.loads(p.read_text());g=d['scenario']
    assert not d.get('accepted') or d['split']=='train'
    assert (g['task_name'],g['variation_idx']) in allowed
    if not d['accepted']:
        audit.append(dict(task=g['task_name'],variation=g['variation_idx'],included=False,reason=d['reason'],score=d.get('score'),turns=len(d.get('turns',[]))))
        continue
    previous=d['initial']
    def obs_message(state,previous=None):
        return build_obs_user_message(state['obs'],state['inventory'],look=state['look'],objects=state['objects'],admissible_commands=state['admissible_commands'],previous=previous)
    messages=[build_system_message(d['instruction'],d['action_templates']),obs_message(previous)]
    for i,t in enumerate(d['turns']):
        messages.append(dict(role='assistant',content='Action: '+t['action']))
        if i+1<len(d['turns']):
            state=t['response'];m=obs_message(state,previous)
            candidates=nearby_commands(state['obs'],t['action'],state['admissible_commands'])
            # Accepted replay has no parser errors: therefore no conditional hint.
            assert not candidates
            messages.append(m);previous=state
    n=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_dict=False))
    initial=len(tok.apply_chat_template(messages[:2],tokenize=True,add_generation_prompt=True,return_dict=False))
    included=initial<=4096 and n-initial<=16384 and n<=20480
    audit.append(dict(task=g['task_name'],variation=g['variation_idx'],included=included,reason='accepted' if included else 'exceeds_online_context_budget_no_truncation',tokens=n,initial_tokens=initial,turns=len(d['turns']),raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if included:rows.append(dict(messages=messages,task_name=g['task_name'],variation_idx=g['variation_idx'],raw_path=str(p),duplicate_for_batch=False))
assert rows
unique=len(rows)
# Native trainer drops incomplete distributed batches. Explicitly repeat <=7
# seeded episodes to retain every unique scenario; record the exact repeats.
rng=random.Random(a.seed)
extras=rng.sample(range(unique),(-unique)%8)
rows += [{**rows[i],'duplicate_for_batch':True} for i in extras]
pd.DataFrame(rows).to_parquet(OUT/'train.pending.parquet',index=False)
config=OmegaConf.create(dict(messages_key='messages',pad_mode='no_padding',max_length=20480,truncation='error',enable_thinking_default=None,ignore_input_ids_mismatch=False))
dataset=MultiTurnSFTDataset(str(OUT/'train.pending.parquet'),tok,config)
mask_tokens=[]
for i,row in enumerate(rows):
    data=dataset[i]  # includes native complete-chat tokenization sanity check
    ids=data['input_ids'];mask=data['loss_mask'];messages=row['messages']
    assert len(ids)<=20480
    end=0
    for j,m in enumerate(messages):
        next_end=len(tok.apply_chat_template(messages[:j+1],tokenize=True,add_generation_prompt=False,return_dict=False))
        if m['role']!='assistant':assert mask[end:next_end].sum().item()==0
        else:
            start=len(tok.apply_chat_template(messages[:j],tokenize=True,add_generation_prompt=True,return_dict=False))
            assert mask[end:start].sum().item()==0
            assert mask[start:next_end].sum().item()==next_end-start
            assert m['content'].startswith('Action: ')
        end=next_end
    assert end==len(ids) and mask.sum()>0
    mask_tokens.append(int(mask.sum()))
summary=dict(unique_episodes=unique,training_rows=len(rows),updates=len(rows)//8,duplicates_for_complete_batch=[dict(task=rows[i]['task_name'],variation=rows[i]['variation_idx']) for i in extras],
    class_counts=dict(collections.Counter(r['task_name'] for r in rows[:unique])),candidate_count=len(allowed),initially_accepted=sum(x.get("accepted",False) for x in [json.loads(p.read_text()) for p in (D/"raw").glob("*.json")]),episode_audit=audit,
    max_tokens=max(x['tokens'] for x in audit if x['included']),total_supervised_tokens=sum(mask_tokens),mask_check='all system/user/header tokens excluded; assistant Action labels only; native full-chat sanity passed for every row',
    source=str(ROOT),prompt_change='Thought optional, no fabricated rationale; matched base-model control uses same prompt',
    train_parquet_sha256=hashlib.sha256((OUT/'train.pending.parquet').read_bytes()).hexdigest(),prompt_sha256=hashlib.sha256((Path(__file__).parent/'sft_prompts.py').read_bytes()).hexdigest(),
    limitation='Small success-selected training set may not cover all 30 task classes. RL/evaluation scope remains all 30. Teacher uses privileged information for labels, never added to observation inputs. No held-out scene used for SFT.')
summary['raw_data_dir']=str(D)
(OUT/'metadata.json').write_text(json.dumps(summary,indent=2))
(OUT/'splits.json').write_text(json.dumps(manifest,indent=2))
(OUT/'train.pending.parquet').rename(OUT/'train.parquet')
print(json.dumps({k:v for k,v in summary.items() if k!='episode_audit'},indent=2))

#!/usr/bin/env python3
"""Audit complete train demonstrations and build native MultiTurnSFTDataset parquet."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.alfworld.prompts import build_system_message, build_obs_user_message


def messages_from_episode(episode):
    assert episode['split'] == 'train' and episode['accepted'] and episode['replay_matches']
    steps = episode['steps']
    assert steps and steps[-1]['done'] and steps[-1]['won']
    messages = [build_system_message(episode['instruction']),
                build_obs_user_message(episode['initial_observation'], episode['initial_admissible_commands'])]
    for i, step in enumerate(steps):
        assert step['action'] in step['admissible_before']
        messages.append(dict(role='assistant', content='Action: ' + step['action']))
        if i + 1 < len(steps):
            assert not step['done']
            messages.append(build_obs_user_message(step['observation'], step['admissible_after']))
    return messages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2]/'datasets/alfworld/sft', help='Published dataset; raw demonstrations remain in --data-dir')
    p.add_argument('--model', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--prompt-length', type=int, default=4096)
    p.add_argument('--response-length', type=int, default=12288)
    a = p.parse_args(); d = a.data_dir.resolve()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not any((out/name).exists() for name in ('train.parquet', 'train.pending.parquet', 'metadata.json', 'splits.json')), 'Use a fresh output directory; never overwrite a published dataset'
    manifest = json.loads((d/'manifest.json').read_text())
    assert manifest['split'] == 'train'
    assert len(list((d/'raw').glob('*.json'))) == len(manifest['scenarios'])
    from transformers import AutoTokenizer
    import pandas as pd
    from omegaconf import OmegaConf
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    tok = AutoTokenizer.from_pretrained(a.model)
    def encode(messages, generation=False):
        return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, return_dict=False)
    rows, audit = [], []
    for i, scenario in enumerate(manifest['scenarios']):
        raw = d/'raw'/f'{i:03d}.json'; ep = json.loads(raw.read_text())
        assert all(ep[k] == scenario[k] for k in ('game_file', 'game_sha256', 'goal_idx', 'task_type'))
        item = dict(index=i, task_type=ep['task_type'], goal_idx=ep['goal_idx'], included=False,
                    raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), reason=ep['reason'], turns=len(ep.get('steps', [])))
        if ep['accepted']:
            assert len(ep['steps']) <= manifest['max_actions']
            messages = messages_from_episode(ep)
            n, initial = len(encode(messages)), len(encode(messages[:2], True))
            include = initial <= a.prompt_length and n-initial <= a.response_length and n <= a.prompt_length+a.response_length
            item.update(tokens=n, initial_tokens=initial, response_tokens=n-initial, included=include,
                        reason='accepted' if include else 'whole_episode_excluded_online_token_budget')
            if include:
                rows.append(dict(messages=messages, task_type=ep['task_type'], goal_idx=ep['goal_idx'],
                                 raw_path=str(raw), duplicate_for_batch=False))
        audit.append(item)
    assert rows
    unique = len(rows)
    assert set(r['task_type'] for r in rows) == set(s['task_type'] for s in manifest['scenarios']), 'Missing task class: review before training'
    extras = random.Random(a.seed).choices(range(unique), k=(-unique) % a.batch_size)
    rows += [{**rows[i], 'duplicate_for_batch': True} for i in extras]
    candidate = out/'train.pending.parquet'
    pd.DataFrame(rows).to_parquet(candidate, index=False)
    config = OmegaConf.create(dict(messages_key='messages', pad_mode='no_padding', max_length=a.prompt_length+a.response_length,
                                   truncation='error', enable_thinking_default=None, ignore_input_ids_mismatch=False))
    dataset = MultiTurnSFTDataset(str(candidate), tok, config)
    supervised = []
    for i, row in enumerate(rows):
        data = dataset[i]  # Native whole-chat sanity check is mandatory.
        ids, mask, messages = data['input_ids'], data['loss_mask'], row['messages']
        assert ids.tolist() == encode(messages)
        end = 0
        for j, message in enumerate(messages):
            next_end = len(encode(messages[:j+1]))
            if message['role'] != 'assistant':
                assert mask[end:next_end].sum().item() == 0
            else:
                start = len(encode(messages[:j], True))
                assert mask[end:start].sum().item() == 0
                assert mask[start:next_end].sum().item() == next_end-start
                assert message['content'].startswith('Action: ')
            end = next_end
        assert end == len(ids) and mask.sum() > 0
        supervised.append(int(mask.sum()))
    summary = dict(unique_episodes=unique, training_rows=len(rows), updates=len(rows)//a.batch_size,
                   candidate_count=len(manifest['scenarios']), class_counts=dict(Counter(r['task_type'] for r in rows[:unique])),
                   repeats=[dict(task_type=rows[i]['task_type'], goal_idx=rows[i]['goal_idx']) for i in extras],
                   episode_audit=audit, max_tokens=max(x['tokens'] for x in audit if x['included']),
                   total_supervised_tokens=sum(supervised), seed=a.seed, prompt_length=a.prompt_length, response_length=a.response_length,
                   train_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                   prompt_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'scripts/alfworld/prompts.py').read_bytes()).hexdigest(),
                   mask_check='Native full-chat and independent prefix masks passed; only Action content and end delimiters supervised.',
                   limitations='Success-selected official train expert labels use privileged teacher state. No invented reasoning. No heldout training. Whole episodes only; no truncation. Seeded repeats only to fill distributed batches.')
    summary['raw_data_dir'] = str(d)
    (out/'metadata.json').write_text(json.dumps(summary, indent=2))
    (out/'splits.json').write_text(json.dumps(manifest, indent=2))
    candidate.rename(out/'train.parquet')
    print(json.dumps({k:v for k,v in summary.items() if k != 'episode_audit'}, indent=2))

if __name__ == '__main__':
    main()

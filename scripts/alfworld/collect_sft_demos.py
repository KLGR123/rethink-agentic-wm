#!/usr/bin/env python3
"""Fixed stratified train-only expert collection, followed by fresh non-expert replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time

TASKS = ('look_at_obj_in_light', 'pick_and_place_simple', 'pick_clean_then_place_in_recep',
         'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_two_obj_and_place')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def make_env(expert):
    from textworld import EnvInfos
    from textworld.envs.wrappers import Filter, GenericEnvironment, Limit
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert
    infos = EnvInfos(won=True, admissible_commands=True, extras=['gamefile'] + (['expert_plan'] if expert else []))
    env = Limit(GenericEnvironment(infos), max_episode_steps=50)
    env = AlfredDemangler()(env)
    env = AlfredInfos(env)
    if expert: env = AlfredExpert(env=env, expert_type='handcoded')
    return Filter(env)

def collect_one(manifest, index):
    from scripts.alfworld.game_files import read_instruction
    row = manifest['scenarios'][index]
    path = Path(row['game_file'])
    assert Path(manifest['data_root']) / 'json_2.1.1/train' in path.parents
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['game_sha256']
    result = dict(row, split='train', accepted=False, instruction=read_instruction(str(path)))
    actions, reference = [], []
    env = make_env(True)
    try:
        env.load(str(path)); obs, infos = env.reset()
        result['initial_observation'] = obs
        result['initial_admissible_commands'] = list(infos['admissible_commands'])
        for turn in range(manifest['max_actions']):
            plan = infos.get('extra.expert_plan', [])
            if not plan: break
            action = plan[0]
            before = obs
            obs, score, done, infos = env.step(action)
            actions.append(action)
            reference.append({'action': action, 'before': before, 'observation': obs,
                              'score': float(score), 'done': bool(done), 'won': bool(infos.get('won', False))})
            if done: break
        result['expert_steps'] = reference
        result['expert_won'] = bool(infos.get('won', False))
    finally: env.close()
    env = make_env(False)
    replay = []
    try:
        env.load(str(path)); obs, infos = env.reset()
        matched = obs == result['initial_observation'] and list(infos['admissible_commands']) == result['initial_admissible_commands']
        for action in actions:
            before, commands = obs, list(infos['admissible_commands'])
            obs, score, done, infos = env.step(action)
            replay.append({'action': action, 'before': before, 'admissible_before': commands,
                           'observation': obs, 'admissible_after': list(infos['admissible_commands']),
                           'score': float(score), 'done': bool(done), 'won': bool(infos.get('won', False))})
            if done: break
        matched = matched and len(replay) == len(reference) and all(
            a['observation'] == b['observation'] and a['won'] == b['won']
            for a, b in zip(reference, replay))
        result.update(steps=replay, replay_matches=matched, final_won=bool(infos.get('won', False)))
        result['accepted'] = bool(result['expert_won'] and result['final_won'] and matched and replay
                                  and all(x['action'] in x['admissible_before'] for x in replay))
        result['reason'] = 'accepted' if result['accepted'] else 'expert_or_replay_failed_or_nonadmissible'
    finally: env.close()
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-root', default='/fs_mol/liujiarun/alfworld/data')
    p.add_argument('--per-class', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--max-actions', type=int, default=30)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--worker', type=int)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--limit', type=int)
    a = p.parse_args(); d = a.output_dir
    d.mkdir(parents=True, exist_ok=True)
    mp = d / 'manifest.json'
    if not mp.exists():
        from scripts.alfworld.game_files import collect_game_files
        games = collect_game_files(a.data_root, 'train')
        groups = {task: [] for task in TASKS}
        for i, path in enumerate(games): groups[Path(path).parent.parent.name.split('-')[0]].append((i, path))
        rng = random.Random(a.seed); chosen = []
        for task in TASKS:
            for goal, path in sorted(rng.sample(groups[task], a.per_class)):
                chosen.append({'task_type': task, 'goal_idx': goal, 'game_file': path,
                               'game_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        save(mp, {'split': 'train', 'data_root': a.data_root, 'seed': a.seed,
                  'max_actions': a.max_actions, 'per_class': a.per_class, 'scenarios': chosen,
                  'train_count': len(games), 'heldout_policy': 'valid_unseen never collected or evaluated; valid_seen only for development',
                  'teacher': 'official AlfredExpert handcoded; privileged expert labels, not policy inputs'})
    m = json.loads(mp.read_text())
    assert m['split'] == 'train' and m['seed'] == a.seed and m['max_actions'] == a.max_actions
    if a.plan_only: print(mp); return
    if a.worker is not None:
        out = d / 'raw' / f'{a.worker:03d}.json'
        assert not out.exists()
        try: result = collect_one(m, a.worker)
        except Exception as e:
            import traceback
            result = dict(m['scenarios'][a.worker], accepted=False, reason=repr(e), traceback=traceback.format_exc())
        save(out, result); return
    import fcntl
    with (d / 'collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = list(range(len(m['scenarios'])))[:a.limit]
        def launch(i):
            out = d / 'raw' / f'{i:03d}.json'
            if out.exists(): return json.loads(out.read_text())['accepted']
            (d / 'logs').mkdir(exist_ok=True)
            with (d / 'logs' / f'{i:03d}.log').open('w') as f:
                proc = subprocess.Popen([sys.executable, __file__, '--output-dir', str(d), '--worker', str(i),
                                         '--seed', str(a.seed), '--max-actions', str(a.max_actions)],
                                        stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                try: proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                if not out.exists(): save(out, dict(m['scenarios'][i], accepted=False, reason=f'worker_exit_{proc.returncode}'))
            return json.loads(out.read_text())['accepted']
        started = time.time()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(launch, i) for i in selected]
            for n, f in enumerate(as_completed(futures), 1):
                f.result()
                raw = [json.loads(x.read_text()) for x in (d/'raw').glob('*.json')]
                status = {'status': 'completed' if n == len(selected) else 'running', 'processed': len(raw),
                          'accepted': sum(x['accepted'] for x in raw), 'expected': len(m['scenarios']),
                          'elapsed_seconds': time.time()-started}
                save(d/'collection.json', status); print(json.dumps(status), flush=True)

if __name__ == '__main__': main()

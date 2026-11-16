"""Replay official train-only expert paths; raw data, never a model evaluator."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

P=sys.executable
# Direct interpreter invocation must also find the environment's Java runtime.
os.environ['PATH']=str(Path(P).parent)+os.pathsep+os.environ.get('PATH','')
D=None
JAR=None
MAX_ACTIONS=100


def one(item):
    from scienceworld import ScienceWorldEnv
    from scripts.sciworld.command_candidates import accessible_commands
    env=ScienceWorldEnv(serverPath=JAR,envStepLimit=MAX_ACTIONS)
    record=dict(scenario=item,split='train',simplification='',turns=[],accepted=False)
    def state(obs,infos):
        return dict(obs=obs,inventory=infos.get('inv',''),look=infos.get('look',''),objects=env.get_possible_objects(),admissible_commands=accessible_commands(env))
    try:
        env.load(item['task_name'],item['variation_idx'],'',False)
        assert item['variation_idx'] in env.get_variations_train()
        env.load(item['task_name'],item['variation_idx'],'',True)
        actions=env.get_gold_action_sequence()
        record['reference_actions']=actions
        assert actions and not actions[0].startswith('ERROR')
        # Regenerate the initial state before replay; do not use teacher state.
        env.load(item['task_name'],item['variation_idx'],'',False)
        obs,infos=env.reset()
        record['initial']=state(obs,infos)
        record['instruction']=infos['taskDesc'].removeprefix('Task Description:').strip() if sys.version_info>=(3,9) else infos['taskDesc'].replace('Task Description:','',1).strip()
        record['action_templates']=env.get_possible_actions()
        for action in actions[:MAX_ACTIONS]:
            obs,reward,done,infos=env.step(action)
            record['turns'].append(dict(action=action,response=state(obs,infos),score=infos['score'],done=done))
            if done:break
        record['score']=infos['score']
        record['accepted']=bool(record['turns'] and done and infos['score']==100 and not any('No known action matches' in t['response']['obs'] for t in record['turns']))
        record['reason']='successful_replay' if record['accepted'] else 'reference_failed_or_exceeded_100_actions_or_parser_error'
    except Exception as exc:
        record['reason']=repr(exc)
    finally:
        env.close()
    (D/'raw'/f"{item['task_name']}-{item['variation_idx']}.json").write_text(json.dumps(record,indent=2))

def launch(item):
    path=D/'raw'/f"{item['task_name']}-{item['variation_idx']}.json"
    if path.exists():return json.loads(path.read_text())['accepted']
    env={**os.environ,'PATH':str(Path(P).parent)+':'+os.environ['PATH']}
    with (D/'logs'/f"{item['task_name']}-{item['variation_idx']}.log").open('w') as log:
        proc=subprocess.Popen([P,__file__,'--output-dir',str(D),'--jar',JAR,'--max-actions',str(MAX_ACTIONS),'--one',json.dumps(item)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=-1
    if not path.exists():
        path.write_text(json.dumps(dict(scenario=item,accepted=False,reason='subprocess_timeout_or_failure',exit_code=code),indent=2))
    return json.loads(path.read_text())['accepted']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--jar',default='/fs_mol/liujiarun/ScienceWorld/scienceworld/scienceworld.jar')
    parser.add_argument('--per-class',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--max-actions',type=int,default=100)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--one')
    args=parser.parse_args();D=args.output_dir.resolve();JAR=args.jar;MAX_ACTIONS=args.max_actions
    D.mkdir(parents=True,exist_ok=True)
    (D/'raw').mkdir(exist_ok=True);(D/'logs').mkdir(exist_ok=True)
    if args.one:
        one(json.loads(args.one))
    else:
        import fcntl,random,hashlib
        with (D/'collection.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            path=D/'manifest.json'
            if not path.exists():
                from scienceworld import ScienceWorldEnv
                env=ScienceWorldEnv(serverPath=JAR,envStepLimit=MAX_ACTIONS)
                rng=random.Random(args.seed);candidates=[]
                try:
                    for task in env.get_task_names():
                        env.load(task,0,'',False)
                        for v in sorted(rng.sample(list(env.get_variations_train()),args.per_class)):
                            candidates.append(dict(task_name=task,variation_idx=v))
                finally:env.close()
                path.write_text(json.dumps(dict(split='train',seed=args.seed,per_class=args.per_class,
                    max_actions=MAX_ACTIONS,candidates=candidates,jar_sha256=hashlib.sha256(Path(JAR).read_bytes()).hexdigest(),
                    profile='sft_prompts.py; accessible candidate filter; privileged expert labels only'),indent=2))
            manifest=json.loads(path.read_text())
            assert manifest.get('split')=='train' and manifest['max_actions']==MAX_ACTIONS
            if args.plan_only:
                print(path);raise SystemExit(0)
            (D/'collection.json').write_text(json.dumps(dict(status='running',pid=os.getpid(),planned_candidates=len(manifest['candidates']))))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for index,ok in enumerate(pool.map(launch,manifest['candidates']),1):
                    print(index,'/',len(manifest['candidates']),'accepted',ok,flush=True)
            records=[json.loads(p.read_text()) for p in (D/'raw').glob('*.json')]
            summary=dict(status='collection_complete',candidates=len(records),accepted=sum(r['accepted'] for r in records),
                next='Review task process and coverage, then prepare complete action-only SFT messages. No model training started.')
            (D/'collection.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

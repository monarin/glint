"""Bounded alternating A/B trials on one allocation; fresh processes, shared caches."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import time
from glint_psana_common import atomic_json
from glint_psana import inside_rank,stop_launcher
from psana_cpu_config import load_config,positive
from validate_psana_parity import survivors
from qspool import file_digest


def trial_order(pairs):
    return [('warm-A','A',None),('warm-B','B',None)]+[
        (f'pair-{i}-{arm}',arm,i) for i in range(pairs) for arm in ('AB' if i%2==0 else 'BA')]


def compare_outputs(a,b):
    if (a/'merged.stream').read_bytes()!=(b/'merged.stream').read_bytes():
        raise ValueError('A/B merged stream mismatch')
    frames=[]
    for group in range(2):
        left=a/f'merged.stream.rank{group:04d}';right=b/f'group-{group}.stream'
        if left.read_bytes()!=right.read_bytes():raise ValueError('A/B group stream mismatch')
        x=json.loads(Path(str(left)+'.bench.json').read_text())
        y=json.loads(Path(str(right)+'.bench.json').read_text())
        for key in ('q_peaks_sha256','events','raw_indexed','strict_indexed','frames','seed','strict_gate','torch','numpy','matmul_precision','tf32'):
            if x[key]!=y[key]:raise ValueError('A/B index metadata mismatch: '+key)
        frames.extend(x['events'])
    if len(frames)!=len(set(frames)):raise ValueError('duplicate hit event')
    return dict(hits=len(frames),exact_stream_and_q=True)


def collect(output,mode,events):
    prefix='reader' if mode=='A' else 'native'
    reports=[json.loads(p.read_text()) for p in sorted(output.glob(prefix+'-*.json'))]
    if len(reports)!=(2 if mode=='A' else 4):raise ValueError('missing process metrics')
    if sum(r['calibration_calls'] for r in reports)!=events:raise ValueError('calibration call count mismatch')
    indices=[json.loads(p.read_text()) for p in sorted(output.glob('*.bench.json'))]
    if len(indices)!=2:raise ValueError('missing index metrics')
    if mode=='B':
        complete=json.loads((output/'mpi-complete.json').read_text())
        if complete['status']!='complete' or complete['harness'] is not False:raise ValueError('native incomplete')
        if any(not r['child_reaped'] for r in complete['ranks']):raise ValueError('native child not reaped')
    return dict(read_bytes=sum(r['read_bytes'] for r in reports),
        calibration_calls=sum(r['calibration_calls'] for r in reports),
        index_seconds_max=max(r['index_seconds'] for r in indices),
        raw_indexed=sum(r['raw_indexed'] for r in indices),
        strict_indexed=sum(len(r['strict_indexed']) for r in indices),
        process_metrics=reports,index_metrics=indices)


def paired_summary(trials):
    paired={}
    for t in trials:
        if t['pair'] is not None:paired.setdefault(t['pair'],{})[t['arm']]=t
    if not paired or any(set(p)!=set('AB') for p in paired.values()):raise ValueError('incomplete A/B pairs')
    ratios=[p['A']['wall_seconds']/p['B']['wall_seconds'] for p in paired.values()]
    return dict(pairs=len(paired),median_A_seconds=statistics.median(p['A']['wall_seconds'] for p in paired.values()),
        median_B_seconds=statistics.median(p['B']['wall_seconds'] for p in paired.values()),
        paired_A_over_B=ratios,median_paired_A_over_B=statistics.median(ratios))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--revision',required=True);parser.add_argument('--pairs',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=2100);args=parser.parse_args(argv)
    positive(args.pairs,'pairs',integer=True);positive(args.timeout,'timeout')
    if inside_rank(os.environ):raise ValueError('launch coordinator outside MPI')
    config=load_config(args.config)
    if config['num_groups']!=2:raise ValueError('two consensus groups required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if config['processor']['geom']:
        geom=root/'reference.geom';geom.write_bytes(Path(config['processor']['geom']).read_bytes())
        config['processor']['geom']=str(geom)
    atomic_json(root/'config.json',config)
    bridge=Path(__file__).resolve().parent;deadline=time.monotonic()+args.timeout
    atomic_json(root/'source.json',dict(revision=args.revision,hostname=os.uname().nodename,
        affinity=sorted(os.sched_getaffinity(0)),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        files={str(p.relative_to(bridge.parents[1])):file_digest(p)
               for d in (bridge,bridge.parents[1]/'glint') for p in d.rglob('*.py')}))
    def launch(command,prelude,log,limit):
        argv=['bash','-c',prelude+' && exec '+shlex.join(command)]
        with log.open('w') as handle:
            started=time.perf_counter();p=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=p.wait(timeout=min(limit,max(.01,deadline-time.monotonic())))
                elapsed=time.perf_counter()-started
                if code:raise RuntimeError(f'{log.name}: exit {code}')
                return elapsed,argv
            except BaseException:stop_launcher(p);raise
    trials=[];by_name={}
    try:
        check=("import sys,json;sys.path.insert(0,"+repr(str(bridge))+");"
               "from glint_psana_common import gpu_identity;g=[gpu_identity(i) for i in range(2)];"
               "assert len({d['pci'] for d in g})==2;print(json.dumps(g))")
        launch(['python','-c',check],config['psana_prelude'],root/'gpu-check.log',120)
        devices=json.loads((root/'gpu-check.log').read_text().strip().splitlines()[-1])
        for name,arm,pair in trial_order(args.pairs):
            output=root/name;output.mkdir();path=output/'config.json';atomic_json(path,config)
            monitor=None;gpu_log=(output/'gpu.csv').open('w')
            try:
                monitor=subprocess.Popen(['nvidia-smi','--id='+','.join(d['pci'] for d in devices),
                    '--query-gpu=timestamp,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits','--loop-ms=1000'],
                    stdout=gpu_log,stderr=subprocess.STDOUT)
                command=['mpirun','--bind-to','none','-n','2' if arm=='A' else '4',
                         'python',str(bridge/'psana_ab_trial.py'),arm,'--config',str(path),'--output',str(output)]
                wall,command=launch(command,config['glint_prelude'] if arm=='A' else config['psana_prelude'],
                                    output/'run.log',config['job_timeout'])
            finally:
                if monitor is not None:
                    monitor.terminate();monitor.wait(timeout=10)
                gpu_log.close()
            live=survivors(root)
            if live:raise RuntimeError(f'surviving benchmark processes: {live}')
            result=dict(name=name,arm=arm,pair=pair,wall_seconds=wall,events_per_second=len(config['reference']['timestamps'])/wall,
                        argv=command,output=str(output),survivors=live,**collect(output,arm,len(config['reference']['timestamps'])))
            atomic_json(output/'trial.json',result);trials.append(result);by_name[name]=output
            atomic_json(root/'trials.json',trials)
            if arm=='B' and pair is None:compare_outputs(by_name['warm-A'],by_name['warm-B'])
            if pair is not None and all(f'pair-{pair}-{a}' in by_name for a in 'AB'):
                compare_outputs(by_name[f'pair-{pair}-A'],by_name[f'pair-{pair}-B'])
            print(name,round(wall,3),'seconds',result['read_bytes'],'bytes',flush=True)
        summary=paired_summary(trials)
        summary.update(status='complete',revision=args.revision,exact_output_parity=True,
            limitations=['203-event sample, few hits; no claim of indexing-heavy throughput',
                         'fresh processes, startup included, shared warm caches; no physical IO claim',
                         'scalar counters and host timers enabled; no image hashing or per-read trace',
                         'three pairs give descriptive variability, not statistical significance'],
            trials_sha256=file_digest(root/'trials.json'))
        atomic_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as exc:
        atomic_json(root/'failed.json',dict(error=f'{type(exc).__name__}: {exc}'))
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid in survivors(root):
                try:os.kill(pid,sig)
                except ProcessLookupError:pass
            if sig==signal.SIGTERM:time.sleep(1)
        raise


if __name__=='__main__':main()

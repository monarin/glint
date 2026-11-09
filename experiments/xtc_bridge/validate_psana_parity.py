"""Bounded standalone acceptance matrix; run once outside MPI on an allocated node."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from glint_psana_common import atomic_json
from glint_psana import inside_rank, stop_launcher
from psana_cpu_config import load_config, positive
from psana_parity_compare import compare
from qspool import file_digest


def survivors(root):
    result=[]
    prefix=('GLINT_PSANA_PROBE_ID='+str(root)+'/').encode()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            tags=(proc/'environ').read_bytes().split(b'\0')
            if any(t.startswith(prefix) for t in tags) and (proc/'stat').read_text().split()[2]!='Z':
                result.append(int(proc.name))
        except OSError: pass
    return result


def matrix_gates(results,event_count,min_indexed,live):
    if event_count%5==0: raise ValueError('matrix did not exercise a partial batch tail')
    owners=results[0]['owners']
    changes=[sum(r['owners'][t]!=owners[t] for t in owners) for r in results[1:]]
    if not any(changes): raise ValueError('matrix did not change event-to-BD ownership')
    if live: raise RuntimeError(f'surviving validation processes: {live}')
    covered=all(len(r['indexed_events'])>=min_indexed for r in results)
    return changes,covered


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--revision',required=True,help='recorded source commit; source hashes also recorded')
    parser.add_argument('--timeout',type=float,default=1800)
    parser.add_argument('--min-indexed',type=int,default=1,help='real indexed-event coverage gate, frozen before running')
    args=parser.parse_args(argv)
    if inside_rank(os.environ): raise ValueError('run once outside MPI')
    positive(args.timeout,'timeout')
    if args.min_indexed<1: raise ValueError('positive deadline and indexed coverage required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    bridge=Path(__file__).resolve().parent
    raw=json.loads(args.config.read_text());source=raw['source']
    # Same default/validation as load_config, also before reference creation.
    raw.setdefault('job_timeout',1200);positive(raw['job_timeout'],'job_timeout')
    if raw['processor'].get('geom'):
        geom=(args.config.parent/raw['processor']['geom']).resolve()
        frozen=root/'reference.geom';frozen.write_bytes(geom.read_bytes())
        raw['processor']['geom']=str(frozen)
    reference=root/'reference.json'
    raw['reference']=str(reference)
    started=time.monotonic();deadline=started+args.timeout;commands=[]
    atomic_json(root/'source.json',dict(revision=args.revision,
        files={str(p.relative_to(bridge.parents[1])):file_digest(p)
               for directory in (bridge,bridge.parents[1]/'glint') for p in directory.rglob('*.py')}))
    def launch(name,command,prelude):
        remaining=deadline-time.monotonic()
        if remaining<=0: raise TimeoutError('acceptance matrix deadline')
        argv=['bash','-c',prelude+' && exec '+shlex.join(command)]
        t=time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=min(remaining,raw['job_timeout']))
            except BaseException:
                stop_launcher(process);raise
        commands.append(dict(name=name,argv=argv,returncode=code,seconds=time.monotonic()-t))
        atomic_json(root/'commands.json',commands)
        if code: raise RuntimeError(f'{name} exited {code}; see {name}.log')
    try:
        gpu_check=("import sys,json; sys.path.insert(0,"+repr(str(bridge))+"); "
                   "from glint_psana_common import gpu_identity; import cupy as cp; "
                   "devices=[gpu_identity(i) for i in range(2)]; "
                   "assert len({d['pci'] for d in devices})==2; "
                   "print(json.dumps(devices),flush=True)")
        launch('gpu-check',['python','-c',gpu_check],raw['psana_prelude'])
        launch('freeze',['python',str(bridge/'freeze_psana_reference.py'),'--exp',source['exp'],
            '--run',str(source['run']),'--dir',source['dir'],'--det',raw['processor']['det'],
            '--max-events',str(source['max_events']),'--output',str(reference)],raw['psana_prelude'])
        path=root/'input.json';atomic_json(path,raw)
        config=load_config(path)
        baseline=root/'baseline';baseline.mkdir();atomic_json(baseline/'config.json',config)
        script=str(bridge/'psana_parity_run.py')
        for group in range(config['num_groups']):
            launch(f'baseline-{group}',['python',script,'baseline','--config',str(baseline/'config.json'),
                   '--output',str(baseline/f'group-{group}'),'--group',str(group)],config['psana_prelude'])
        results=[]
        for name,batch,swapped in [('batch1',1,False),('batch5',5,False),('batch5-swapped',5,True)]:
            trial=copy.deepcopy(config);trial['source']['batch_size']=batch
            if swapped: trial['devices']={k:1-v for k,v in trial['devices'].items()}
            output=root/name;output.mkdir();atomic_json(output/'config.json',trial)
            launch(name,['mpirun','-n','4','python',script,'native','--config',str(output/'config.json'),
                         '--output',str(output)],config['psana_prelude'])
            results.append(dict(case=name,**compare(baseline,output)))
        live=survivors(root)
        changes,covered=matrix_gates(results,source['max_events'],args.min_indexed,live)
        atomic_json(root/'comparison.json',dict(parity_pass=True,indexed_coverage=covered,
            min_indexed=args.min_indexed,cases=results,ownership_changes=changes,survivors=live,
            wall_seconds=time.monotonic()-started,
            limitations=['diagnostic hash/counter hooks add overhead; not a performance benchmark',
                         'read counts cover EventManager bulk reads, not SMD/metadata or physical storage',
                         'stream equality gate is stricter than crystallographic basis equivalence']))
        if not covered: raise ValueError('numerical parity passed but real indexed-crystal coverage is incomplete')
        atomic_json(root/'acceptance-complete.json',dict(status='complete',revision=args.revision,
                    comparison_sha256=file_digest(root/'comparison.json')))
        print('GLINT_PSANA_PARITY_OK',root,flush=True)
        return 0
    except BaseException as exc:
        atomic_json(root/'failed.json',dict(error=f'{type(exc).__name__}: {exc}'))
        # Only processes tagged with this newly-created validation directory are owned here.
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid in survivors(root):
                try: os.kill(pid,sig)
                except ProcessLookupError: pass
            if sig==signal.SIGTERM: time.sleep(1)
        raise


if __name__=='__main__': raise SystemExit(main())

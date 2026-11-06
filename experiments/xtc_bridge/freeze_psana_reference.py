"""Explicit OFFLINE BASELINE utility, outside the measured native MPI execution."""
import argparse
import os
from pathlib import Path
from glint_psana import inside_rank
from glint_probe_common import atomic_json
from psana_reference import Reference, calibration_digest, event_timestamp


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp',required=True)
    p.add_argument('--run',type=int,required=True)
    p.add_argument('--dir',required=True)
    p.add_argument('--det',default='jungfrau')
    p.add_argument('--max-events',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if inside_rank(os.environ) or args.max_events<=0:
        raise ValueError('freeze the bounded reference once, outside MPI')
    if args.output.exists():
        raise ValueError('reference output already exists')
    os.environ['PS_PARALLEL']='none'
    from psana import DataSource
    ds=DataSource(exp=args.exp,run=args.run,dir=args.dir,max_events=args.max_events)
    reference=None
    for run in ds.runs():
        if reference is not None or run.expt!=args.exp or run.runnum!=args.run:
            raise ValueError('expected exactly one configured run')
        digest=calibration_digest(run.Detector(args.det).calibconst)
        stamps=[event_timestamp(evt) for evt in run.events()]
        if len(stamps)!=args.max_events:
            raise ValueError('baseline did not yield the requested prefix length')
        reference=dict(schema_version=1,experiment=args.exp,run=args.run,
                       calibration_digest=digest,timestamps=stamps)
        Reference(reference)
        close=getattr(run,'close_shared_memory',None)
        if close:
            close()
    if reference is None:
        raise ValueError('no reference run')
    atomic_json(args.output,reference)


if __name__=='__main__':
    main()

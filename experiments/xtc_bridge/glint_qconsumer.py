"""Ordinary GLINT child consuming one complete group; never reads XTC or initializes MPI."""
from pathlib import Path
from types import SimpleNamespace
import os
import qrecords
from psana_reference import Reference
from qspool import records_digest, file_digest
from glint_child_runtime import initialize as initialize_runtime, isolated

_ready=None


def initialize(parent, config):
    global _ready
    _ready=initialize_runtime(parent, dict(check_gpu=True, allow_same_environment=False,
                                         expected_child_prefix=config['expected_child_prefix']))
    return _ready


def index_group(records, config, group, config_digest, output):
    isolated()
    if _ready is None or _ready['gpu'] is None:
        raise RuntimeError('initialize and verify the child GPU before indexing')
    if 'integrate' in config or 'integrate' in config.get('processor',{}):
        raise ValueError('integration is unsupported: the child must not read XTC')
    reference=Reference(config['reference'])
    for record in records:
        reference.check_record(record)
    source=config['source']; processor=config['processor']
    if processor.get('geom') and file_digest(processor['geom'])!=config['geometry_digest']:
        raise ValueError('geometry changed before indexing')
    out=qrecords.to_legacy(records,experiment=source['exp'],run=source['run'],
                          config_digest=config_digest,group_id=group,num_groups=config['num_groups'],
                          expected_ordinals=range(group,len(reference.ordinals),config['num_groups']),
                          max_bytes=config['max_group_bytes'])
    # Construct an explicit allowlist; never forward a legacy unrestricted Namespace.
    args=SimpleNamespace(exp=source['exp'],run=source['run'],geom=processor['geom'],
                         zdist=processor['zdist'],wavelength=processor['wavelength'],
                         cell=[str(v) for v in config['cell']] if config['cell'] else None,
                         nbest=config['nbest'],integrate=False)
    from glint_xtc import index_and_write
    _,_,count=index_and_write(out,args,output,report=False)
    path=Path(output)
    if not path.is_file() or path.stat().st_size==0:
        raise RuntimeError('GLINT did not write a valid group output')
    isolated()
    return dict(group=group,config_digest=config_digest,digest=records_digest(records),
                pid=os.getpid(),n_frames=len(out['events']),n_indexed=int(count),
                output=str(path),output_bytes=path.stat().st_size,sha256=file_digest(path))

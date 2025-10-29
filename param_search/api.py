from typing import Optional, Iterable, Dict, Any
from pathlib import Path
import pandas as pd

from . import types, utils, queues


VALID_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890-_')
COLUMN_ORDER = ['job_name', 'job_state', 'n_submits', 'job_id', 'node_id', 'runtime', 'work_dir',  'script_path', 'stdout_path', 'stderr_path', 'stdout_raw', 'stderr_raw',]

_QUEUE = None


def get_queue():
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = queues.slurm.SlurmQueue()
    return _QUEUE


def set_queue(queue):
    global _QUEUE
    _QUEUE = queue


def set_backend(backend):
    if backend == 'local':
        set_queue(queue.local.LocalQueue())
    elif backend == 'slurm':
        set_queue(queues.slurm.SlurmQueue())
    elif backend == 'torque':
        set_queue(queues.torque.TorqueQueue())
    raise ValueError(f'invalid backend: {backend}')


def grid(**dims) -> Iterable[Dict[str, Any]]:
    from itertools import product
    from collections import OrderedDict
    keys = list(dims.keys())
    dims = [utils.as_list(v) for v in dims.values()]
    return [OrderedDict(zip(keys, vals)) for vals in product(*dims)]


def setup(
    template: str,
    name_format: str,
    param_space: Iterable[Dict[str, Any]],
    base_dir: Path='.',
    script_name: str='run.sh',
    overwrite: bool=False,
    write: bool=True
) -> pd.DataFrame:
    import json

    base_dir = Path(base_dir).resolve()
    job_names = set()
    rows = []

    for p in param_space:
        params_json = json.dumps(p, sort_keys=True)
        params_hash = utils.hash_func(params_json.encode())
        job_name = name_format.format(params_hash=params_hash, **p)

        if set(job_name) - VALID_CHARS:
            raise ValueError(f'invalid job name: {job_name}')

        if job_name in job_names:
            raise RuntimeError(f'name not unique: {job_name}')

        job_names.add(job_name)
        script_body = template.format(job_name=job_name, params_hash=params_hash, **p)

        work_dir = base_dir / job_name
        if write:
            work_dir.mkdir(parents=True, exist_ok=overwrite)
            script_path = work_dir / script_name
            if script_path.exists() and not overwrite:
                raise IOError(f'{script_path} already exists')
            script_path.write_text(script_body)

        rows.append({
            'job_name': job_name,
            'job_state': 'NEW',
            'n_submits': 0,
            'job_id': pd.NA,
            'node_id': pd.NA,
            'runtime': pd.NA,
            'work_dir': Path(work_dir),
            'script_path': Path(script_path),
            'stdout_path': pd.NA,
            'stderr_path': pd.NA,
            'stdout_raw': pd.NA,
            'stderr_raw': pd.NA,
            'params_json': params_json,
            'params_hash': params_hash,
            **utils.namespace(p, name='params'),
        })

    jobs = pd.DataFrame(rows)

    # reorder columns for readability
    param_cols = [c for c in jobs.columns if c.startswith('params')]

    return jobs[COLUMN_ORDER + param_cols]


def submit(jobs: pd.DataFrame, update: bool=False, **queue_kws):

    # idempotence: select rows ready to be submit
    sel = jobs['job_id'].isna()
    if not sel.any():
        return sel, [], [], []

    scripts = jobs.loc[sel, 'script_path'].map(Path).tolist()
    work_dirs = jobs.loc[sel, 'work_dir'].map(Path).tolist()
    log_dirs = [wd / 'logs' for wd in work_dirs]

    job_ids = get_queue().submit(scripts, **queue_kws)

    if len(job_ids) != len(scripts):
        raise RuntimeError(f'num job_ids mismatch: {len(job_ids)} vs. {len(scripts)}')

    job_ids      = [str(j) for j in job_ids]
    stdout_paths = [(d / f'{j}.out').as_posix() for d, j in zip(log_dirs, job_ids)]
    stderr_paths = [(d / f'{j}.err').as_posix() for d, j in zip(log_dirs, job_ids)]

    if update:
        jobs.loc[sel, 'job_id']      = job_ids
        jobs.loc[sel, 'job_state']   = 'SUBMITTED'
        jobs.loc[sel, 'n_submits']   = jobs.loc[sel, 'n_submits'].fillna(0).astype(int) + 1
        jobs.loc[sel, 'stdout_path'] = stdout_paths
        jobs.loc[sel, 'stderr_path'] = stderr_paths
        return jobs

    return sel, job_ids, stdout_paths, stderr_paths


def status(jobs: pd.DataFrame, update=False):

    status = get_queue().status(jobs['job_id'])
    stdout = jobs['stdout_path'].map(_read_file)
    stderr = jobs['stderr_path'].map(_read_file)
    if update:
        jobs = jobs.merge(status, on='job_id', how='left', suffixes=('', '_new'))
        for col in ['job_state', 'node_id', 'runtime']:
            jobs[col] = jobs[f'{col}_new'].fillna(jobs[col])

        jobs.drop(columns=[c for c in jobs.columns if c.endswith('_new')], inplace=True)
        jobs['stdout'] = stdout
        jobs['stderr'] = stderr
        return jobs

    return status, stdout, stderr


def collect(status):
    raise NotImplementedError


def plot(*args, **kwargs):
    raise NotImplementedError


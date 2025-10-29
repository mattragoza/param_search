from typing import Iterable, Dict, Any
from pathlib import Path
import pandas as pd

from . import utils, queues

from .utils import set_verbose


VALID_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890.-_')
COLUMN_ORDER = [
    'job_name', 'job_state', 'n_submits', 
    'job_id', 'node_id', 'runtime',
    'stdout', 'stderr',
    'work_dir', 'script_path',
    'log_dir', 'stdout_path', 'stderr_path',
]
TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'}
QUEUE = None


def get_queue():
    global QUEUE
    if QUEUE is None:
        QUEUE = queues.slurm.SlurmQueue()
    return QUEUE


def set_queue(queue):
    global QUEUE
    QUEUE = queue


def set_backend(backend):
    backend = backend.lower()
    if backend == 'local':
        set_queue(queues.local.LocalQueue())
    elif backend == 'slurm':
        set_queue(queues.slurm.SlurmQueue())
    elif backend == 'torque':
        set_queue(queues.torque.TorqueQueue())
    else:
        raise ValueError(f'invalid backend: {backend}')


def param_grid(**dims) -> Iterable[Dict[str, Any]]:
    from itertools import product
    keys = list(dims.keys())
    dims = [utils.as_list(v) for v in dims.values()]
    return [dict(zip(keys, vals)) for vals in product(*dims)]


def setup(
    template: str,
    name_format: str,
    param_space: Iterable[Dict[str, Any]],
    base_dir: Path='.',
    script_name: str='run.sh',
    overwrite: bool=False,
    write: bool=True,
) -> pd.DataFrame:
    import json

    base_dir = Path(base_dir).resolve()
    used_names = set()
    rows = []

    for p in param_space:
        params_json = json.dumps(p, sort_keys=True)
        params_hash = utils.hash_params(p)

        job_name = name_format.format(params_hash=params_hash, **p)

        if not (set(job_name) <= VALID_CHARS):
            raise ValueError(f'invalid job name: {job_name}')

        if job_name in used_names:
            raise RuntimeError(f'name not unique: {job_name}')

        used_names.add(job_name)

        work_dir    = base_dir / job_name
        log_dir     = work_dir / 'logs'
        script_path = work_dir / script_name
        script_body = template.format(job_name=job_name, params_hash=params_hash, **p)

        if write:
            if script_path.exists() and not overwrite:
                raise IOError(f'{script_path} already exists')

            if not work_dir.is_dir():
                utils.log(f'mkdir {work_dir}')
                work_dir.mkdir(parents=True)

            if not log_dir.is_dir():
                utils.log(f'mkdir {log_dir}')
                log_dir.mkdir(parents=True, exist_ok=True)

            utils.log(f'write {script_path}')
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
            'log_dir': Path(log_dir),
            'stdout_path': pd.NA,
            'stderr_path': pd.NA,
            'stdout': pd.NA,
            'stderr': pd.NA,
            'params_json': params_json,
            'params_hash': params_hash,
            **utils.namespace(p, name='params'),
        })

    jobs = pd.DataFrame(rows)
    param_cols = [c for c in jobs.columns if c.startswith('params')]

    # reorder columns for readability
    return jobs[COLUMN_ORDER + param_cols]


def submit(jobs: pd.DataFrame, queue=None, **queue_kws):
    queue = queue or get_queue()

    sel = jobs['job_id'].isna()
    if not sel.any():
        return jobs.copy()

    scripts   = jobs.loc[sel, 'script_path'].map(Path).tolist()
    work_dirs = jobs.loc[sel, 'work_dir'].map(Path).tolist()
    log_dirs  = jobs.loc[sel, 'log_dir'].map(Path).tolist()

    job_ids = [str(j) for j in queue.submit(scripts, **queue_kws)]

    if len(job_ids) != len(scripts):
        raise RuntimeError(f'num job_ids mismatch: {len(job_ids)} vs. {len(scripts)}')

    stdout_paths = [(d / f'{j}.out').as_posix() for d, j in zip(log_dirs, job_ids)]
    stderr_paths = [(d / f'{j}.err').as_posix() for d, j in zip(log_dirs, job_ids)]

    out = jobs.copy()
    out.loc[sel, 'job_id']      = job_ids
    out.loc[sel, 'job_state']   = 'SUBMITTED'
    out.loc[sel, 'n_submits']   = out.loc[sel, 'n_submits'].fillna(0).astype(int) + 1
    out.loc[sel, 'stdout_path'] = stdout_paths
    out.loc[sel, 'stderr_path'] = stderr_paths

    return out


def status(jobs: pd.DataFrame, queue=None, ret_stat=False):
    queue = queue or get_queue()

    sel = jobs['job_id'].notna()
    if not sel.any():
        if ret_stat:
            return pd.DataFrame(columns=['job_id', 'job_state', 'node_id', 'runtime'])
        return jobs.copy()

    job_ids = jobs.loc[sel, 'job_id'].astype(str).tolist()

    stat = queue.status(job_ids)

    if ret_stat:
        return stat

    if stat.empty:
        missing = sel & ~jobs['job_state'].isin(TERMINAL_STATES)
        jobs = jobs.copy()
        jobs.loc[missing, 'job_state'] = jobs.loc[missing, 'job_state'].fillna('MISSING')
        return jobs

    # merge and carefully update job states
    merged = jobs.merge(stat, on='job_id', how='left', suffixes=('', '_new'))

    has_new = merged['job_state_new'].notna()
    merged.loc[has_new, 'job_state'] = merged.loc[has_new, 'job_state_new']

    was_terminal = merged['job_state'].isin(TERMINAL_STATES)
    missing = ~was_terminal & ~has_new
    merged.loc[missing, 'job_state'] = 'MISSING'

    merged.loc[has_new, 'state_source'] = 'status'
    merged.loc[missing, 'state_source'] = merged.loc[missing, 'state_source'].fillna('squeue(missing)')

    for col in ['node_id', 'runtime']:
        new_col = f'{col}_new'
        if new_col in merged:
            has_new = merged[new_col].notna()
            merged.loc[has_new, col] = merged.loc[has_new, new_col]

    keep = [c for c in merged.columns if not c.endswith('_new')]
    out = merged[keep]

    return out


def collect(jobs: pd.DataFrame, tail=20):
    from . import text

    sel = jobs['stdout_path'].notna() & jobs['stderr_path'].notna()
    if not sel.any():
        return jobs.copy()

    stdout = jobs.loc[sel, 'stdout_path'].map(text.read_tail)
    stderr = jobs.loc[sel, 'stderr_path'].map(text.read_tail)

    out = jobs.copy()
    out.loc[sel, 'stdout'] = stdout
    out.loc[sel, 'stderr'] = stderr

    return out




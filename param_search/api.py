from typing import Iterable, Callable, Dict, Any
from pathlib import Path
import pandas as pd

from . import utils, queues

from .utils import set_verbose


VALID_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567890.-_')
COLUMN_ORDER = [
    'job_name', 'job_state', 'n_submits', 
    'job_id', 'node_id', 'runtime',
    'stdout', 'stderr',
    'base_dir',
    'work_dir', 'script_path', 'output_path',
    'log_dir', 'stdout_path', 'stderr_path',
]
LIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'}
TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'}
AUTOSAVE = True
QUEUE = None


def get_queue():
    global QUEUE
    if QUEUE is None:
        QUEUE = queues.slurm.SlurmQueue()
    return QUEUE


def set_queue(queue):
    global QUEUE
    QUEUE = queue


def set_autosave(val):
    global AUTOSAVE
    AUTOSAVE = val


def _autosave(jobs):
    if AUTOSAVE:
        save(jobs)


def save(jobs: pd.DataFrame, base_dir: Path=None):
    base_dir = base_dir or utils.get_base_dir(jobs)
    path = Path(base_dir) / 'jobs.parquet'
    utils.log(f'Saving {path}')
    utils.atomic_write(path, jobs)
    return path


def load(base_dir) -> pd.DataFrame:
    path = Path(base_dir) / 'jobs.parquet'
    if not path.is_file():
        raise FileNotFoundError(f'No jobs found at {path}')
    utils.log(f'Loading {path}')
    return pd.read_parquet(path)


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
    base_dir: str|Path,
    template: str,
    name_format: str,
    param_space: Iterable[Dict[str, Any]],
    script_name: str='run.sh',
    output_name: str='output.csv',
    overwrite: bool=False,
    write: bool=True
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
        script_path = work_dir / script_name
        output_path = work_dir / output_name
        log_dir     = work_dir / 'logs'

        script_body = template.format(
            params_hash=params_hash, 
            job_name=job_name,
            output_path=output_path,
            **p
        )
        if write:
            if script_path.exists() and not overwrite:
                raise IOError(f'{script_path} already exists')

            utils.make_dirs(work_dir)
            utils.make_dirs(script_path.parent)
            utils.make_dirs(output_path.parent)
            utils.make_dirs(log_dir)

            utils.log(f'write {script_path}')
            script_path.write_text(script_body)

        rows.append({
            'job_name': job_name,
            'job_state': 'NEW',
            'n_submits': 0,
            'job_id': pd.NA,
            'node_id': pd.NA,
            'runtime': pd.NA,
            'work_dir': str(work_dir),
            'script_path': str(script_path),
            'output_path': str(output_path),
            'log_dir': str(log_dir),
            'stdout_path': pd.NA,
            'stderr_path': pd.NA,
            'stdout': pd.NA,
            'stderr': pd.NA,
            'params_json': params_json,
            'params_hash': params_hash,
            **utils.namespace(p, name='params'),
        })

    jobs = pd.DataFrame(rows)
    jobs['base_dir'] = str(base_dir) # store for saves/loads

    # reorder columns for readability
    param_cols = [c for c in jobs.columns if c.startswith('params')]
    out = jobs[COLUMN_ORDER + param_cols]
    if write:
        _autosave(out)
    return out


def submit(jobs: pd.DataFrame, queue=None, queue_kws=None) -> pd.DataFrame:
    queue = queue or get_queue()

    sel = jobs['job_id'].isna()
    if not sel.any():
        utils.log('no jobs require submit')
        return jobs.copy()

    scripts   = jobs.loc[sel, 'script_path'].map(Path).tolist()
    work_dirs = jobs.loc[sel, 'work_dir'].map(Path).tolist()
    log_dirs  = jobs.loc[sel, 'log_dir'].map(Path).tolist()

    job_ids = [str(j) for j in queue.submit(scripts, **(queue_kws or {}))]

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
    _autosave(out)
    return out


def status(jobs: pd.DataFrame, queue=None, ret_stat=False) -> pd.DataFrame:
    from datetime import datetime
    queue = queue or get_queue()
    stat_cols = ['job_id', 'job_state', 'runtime', 'node_id']

    sel = jobs['job_id'].notna()
    if not sel.any():
        utils.log('no jobs require status')
        return pd.DataFrame(columns=stat_cols) if ret_stat else jobs.copy()

    job_ids = jobs.loc[sel, 'job_id'].astype(str).tolist()
    stat = queue.status(job_ids)
    if ret_stat:
        return stat

    merged = jobs.merge(stat, on='job_id', how='left', suffixes=('', '_new'))
    has_new = merged['job_state_new'].notna()

    for col in ['job_state', 'node_id', 'runtime']:
        new_col = f'{col}_new'
        if new_col in merged:
            merged.loc[has_new, col] = merged.loc[has_new, new_col]

    now = datetime.now().isoformat(timespec='seconds')
    merged.loc[has_new, 'last_live_at'] = now
    merged.loc[has_new, 'state_source'] = 'status'

    keep_cols = [c for c in merged.columns if not c.endswith('_new')]
    out = merged[keep_cols]
    _autosave(out)
    return out


def history(jobs: pd.DataFrame, queue=None, ret_hist=False) -> pd.DataFrame:
    from datetime import datetime
    queue = queue or get_queue()
    hist_cols = ['job_id', 'job_state', 'runtime', 'node_id']

    sel = jobs['job_id'].notna() & ~jobs['job_state'].isin(TERMINAL_STATES)
    if not sel.any():
        utils.log('no jobs require history')
        return pd.DataFrame(columns=hist_cols) if ret_hist else jobs.copy()

    job_ids = jobs.loc[sel, 'job_id'].astype(str).tolist()
    hist = queue.history(job_ids)
    if ret_hist:
        return hist

    if not hist.empty and 'job_state' in hist:
        hist = hist[hist['job_state'].isin(TERMINAL_STATES)]

    merged = jobs.merge(hist, on='job_id', how='left', suffixes=('', '_new'), sort=False)
    to_final = merged['job_state_new'].isin(TERMINAL_STATES) & ~merged['job_state'].isin(TERMINAL_STATES)

    for col in ['job_state', 'node_id', 'runtime']:
        new_col = f'{col}_new'
        if new_col in merged:
            merged.loc[to_final, col] = merged.loc[to_final, new_col]

    now = datetime.now().isoformat(timespec='seconds')
    merged.loc[to_final, 'finalized'] = True
    merged.loc[to_final, 'finalized_at'] = now
    merged.loc[to_final, 'state_source'] = 'history'

    keep_cols = [c for c in merged.columns if not c.endswith('_new')]
    out = merged[keep_cols]
    _autosave(out)
    return out


def collect(jobs: pd.DataFrame, tail: int=20) -> pd.DataFrame:
    from . import text
    out = jobs.copy()

    has_logs = jobs['stdout_path'].notna() & jobs['stderr_path'].notna()
    if has_logs.any():
        utils.log('parsing log files')
        read_tail = lambda p: text.read_tail(p, max_lines=tail)
        stdout = jobs.loc[has_logs, 'stdout_path'].map(read_tail)
        stderr = jobs.loc[has_logs, 'stderr_path'].map(read_tail)
        error  = [text.parse_error_line(s) for s in stderr]

        out.loc[has_logs, 'stdout'] = stdout
        out.loc[has_logs, 'stderr'] = stderr
        out.loc[has_logs, 'error_line']  = error

    utils.log('checking output files')

    # output validation
    exist_vals, fsize_vals, mtime_vals = [], [], []

    for p in out['output_path'].astype(str).fillna(''):
        output_path = Path(p)
        if output_path.is_file():
            exist_vals.append(True)
            stat = output_path.stat()
            fsize_vals.append(stat.st_size)
            mtime_vals.append(stat.st_mtime)
        else:
            exist_vals.append(False)
            fsize_vals.append(pd.NA)
            mtime_vals.append(pd.NA)

    out['output_exists'] = exist_vals
    out['output_fsize']  = fsize_vals
    out['output_mtime']  = mtime_vals
    _autosave(out)
    return out


def recover(jobs: pd.DataFrame) -> pd.DataFrame:

    sel = jobs['job_id'].isna()
    if not sel.any():
        return jobs.copy()

    rows = []
    for idx in jobs.index[sel]:
        work_dir = Path(jobs.at[idx, 'work_dir'])
        log_dir = work_dir / 'logs'
        job_ids = utils.find_job_ids(log_dir)
        if not job_ids:
            continue
        last_job_id = str(job_ids[-1])
        stdout_path = (log_dir / f'{last_job_id}.out').as_posix()
        stderr_path = (log_dir / f'{last_job_id}.err').as_posix()
        n_submits = len(job_ids)
        rows.append({
            'index': idx,
            'job_id': last_job_id,
            'stdout_path': stdout_path,
            'stderr_path': stderr_path,
            'n_submits': n_submits,
        })

    if not rows:
        return jobs.copy()

    update = pd.DataFrame(rows).set_index('index')
    idx = update.index # only assign recovered rows

    out = jobs.copy()
    out.loc[idx, 'job_id'] = update['job_id'].astype(str)
    out.loc[idx, 'stdout_path'] = update['stdout_path']
    out.loc[idx, 'stderr_path'] = update['stderr_path']

    prev_submits = out.loc[idx, 'n_submits'].fillna(0).astype(int)
    out.loc[idx, 'n_submits'] = prev_submits.combine(update['n_submits'], max)

    needs_state = out.loc[idx, 'job_state'].isna() | out.loc[idx, 'job_state'].eq('NEW')
    out.loc[idx[needs_state], 'job_state'] = 'RECOVERED'
    _autosave(out)
    return out


def outputs(
    jobs: pd.DataFrame,
    meta_cols: Iterable[str]=('job_name', 'job_id', 'params_hash'),
    include_params: bool=True,
    read_csv_kws: Dict[str, Any]=None,
    skip_errors: bool=False
) -> pd.DataFrame:

    meta_cols = list(meta_cols)
    if include_params:
        meta_cols += [c for c in jobs.columns if c.startswith('params.')]

    data = []
    for idx, row in jobs.iterrows():
        job_name = row.get('job_name')
        out_path = row.get('output_path')

        df, error = utils.safe_load(out_path, **(read_csv_kws or {}))
        if error and not skip_errors:
            raise RuntimeError(f'Job {job_name}: {error}')
        elif error:
            utils.warn(f'Job {job_name}: {error}')
            continue

        df = df.copy()
        df['source_path'] = str(out_path)
        meta = {c: row.get(c, pd.NA) for c in meta_cols}
        for k, v in meta.items():
            df[k] = v

        data.append(df)

    if not data:
        return pd.DataFrame(columns=meta_cols + ['source_path', 'load_error'])

    return pd.concat(data, ignore_index=True)


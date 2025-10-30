from typing import List
import os, re
import pandas as pd
from . import base
from .. import utils, shell


class SlurmQueue(base.BaseQueue):

    @staticmethod
    def _submit_cmd(path, *args, **kwargs):

        abs_path = os.path.abspath(path)
        work_dir = os.path.dirname(abs_path)
        logs_dir = os.path.join(work_dir, 'logs')

        if 'array' in kwargs:
            stdout_fmt = os.path.join(logs_dir, '%A_%a.out')
            stderr_fmt = os.path.join(logs_dir, '%A_%a.err')
        else:
            stdout_fmt = os.path.join(logs_dir, '%j.out')
            stderr_fmt = os.path.join(logs_dir, '%j.err')

        # NOTE: sbatch options must come BEFORE the script path
        return shell.as_command(
            'sbatch',
            *args,
            output=stdout_fmt,
            error=stderr_fmt,
            **kwargs
        ) + ' ' + shell._as_arg_value(abs_path)

    @staticmethod 
    def _status_cmd(job_ids, *args, **kwargs):

        # slurm errors when a single unknown job is queried; pad with dummy
        job_ids = [str(j) for j in job_ids]
        if len(job_ids) == 1:
            job_ids.append('0')

        # columns: JOBID STATE TIME NODELIST(REASON)
        fmt = '%i %T %M %R'

        return shell.as_command(
            'squeue',
            *args,
            format=fmt,
            noheader=True,
            job=job_ids,
            **kwargs
        )

    @staticmethod
    def _history_cmd(job_ids, *args, **kwargs):
        return shell.as_command(
            'sacct',
            *args,
            job=[str(j) for j in job_ids],
            format='JobIDRaw,State,Elapsed,NodeList',
            parsable2=True,
            noheader=True,
            allocations=True,
            **kwargs
        )

    @staticmethod
    def _cancel_cmd(*args, **kwargs):
        return shell.as_command('scancel', *args, **kwargs)

    @staticmethod
    def _parse_submit(stdout: str) -> List[str]:
        pat = r'^Submitted batch job (\d+)( on cluster .+)?\n$'
        match = re.match(pat, stdout)
        if match:
            return int(match.group(1))
        raise RuntimeError(f'failed to parse: {stdout:r}')

    @staticmethod
    def _parse_status(stdout: str) -> pd.DataFrame:
        columns = ['job_id', 'job_state', 'runtime', 'node_id']
        df = _parse_status(stdout, columns=columns)
        if not df.empty: # split off array_idx from job_id
            parts = df['job_id'].astype(str).str.split('_', n=1, expand=False)
            df['job_id'] = [p[0] for p in parts]
            array_inds = [p[1] if len(p) == 2 else None for p in parts]
            df['array_idx'] = pd.to_numeric(array_inds, errors='coerce')
        else:
            df['array_idx'] = pd.Series([], dtype='float')
        return df

    @staticmethod
    def _parse_history(stdout: str) -> pd.DataFrame:
        columns = ['job_id', 'job_state', 'runtime', 'node_id']
        df = _parse_history(stdout, columns=columns)
        return df


def _parse_status(stdout: str, columns: List[str], sep=' ') -> pd.DataFrame:
    from .. import text
    if not stdout:
        return pd.DataFrame(columns=columns)

    rows = []
    for line in filter(len, stdout.splitlines()):
        tokens = text.paren_split(line, sep=sep)
        if len(tokens) != len(columns):
            raise ValueError(f'failed to parse: {len(tokens)} vs. {len(columns)}')
        rows.append(dict(zip(columns, tokens)))

    return pd.DataFrame(rows, columns=columns)


def _parse_history(stdout: str, columns: List[str], sep='|') -> pd.DataFrame:
    if not stdout:
        return pd.DataFrame(columns=columns)

    rows = []
    for line in filter(len, stdout.splitlines()):
        tokens = line.split(sep)
        if len(tokens) != len(columns):
            raise ValueError(f'failed to parse: {len(tokens)} vs. {len(columns)}')
        rows.append(dict(zip(columns, tokens)))

    return pd.DataFrame(rows, columns=columns)

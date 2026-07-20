from typing import List
import os, re

import pandas as pd
import datetime as dt

from . import base
from .. import utils, shell


class SGEQueue(base.BaseQueue):
    '''
    Sun Grid Engine queuing system.
    '''
    @staticmethod
    def _submit_cmd(path, *args, **kwargs):

        abs_path = os.path.abspath(path)
        work_dir = os.path.dirname(abs_path)
        logs_dir = os.path.join(work_dir, 'logs')

        kwargs = dict(kwargs)
        array = kwargs.pop('array', None)

        if array:
            stdout_fmt = os.path.join(logs_dir, '$JOB_ID_$SGE_TASK_ID.out')
            stderr_fmt = os.path.join(logs_dir, '$JOB_ID_$SGE_TASK_ID.err')
            kwargs['t'] = array # number of tasks
        else:
            stdout_fmt = os.path.join(logs_dir, '$JOB_ID.out')
            stderr_fmt = os.path.join(logs_dir, '$JOB_ID.err')

        cmd = _as_command(
            'qsub',
            *args,
            terse=True,
            o=stdout_fmt,
            e=stderr_fmt,
            **kwargs
        )

        # qsub options must precede the script path
        return cmd + ' ' + shell.as_arg_value(abs_path)

    @staticmethod
    def _status_cmd(job_ids, *args, **kwargs):
        
        # qstat does not support querying by job id,
        #   so instead we query all user jobs

        user = os.environ.get('USER', '*')

        return _as_command(
            'qstat',
            *args,
            u=user,
            **kwargs
        )

    @staticmethod
    def _history_cmd(job_ids, *args, **kwrags):

        # qacct does not support querying by job id,
        #   so instead we query recent user jobs
        
        kwargs = dict(kwargs)
        user = os.environ.get('USER')
        days = kwargs.pop('days', 30)

        return _as_command(
            'qacct',
            *args,
            o=user,
            d=days,
            **kwargs
        )

    @staticmethod
    def _cancel_cmd(*args, **kwargs):
        return _as_command('qdel', *args, **kwargs)

    @staticmethod
    def _parse_submit(stdout: str) -> List[str]:
        
        # terse output: 12345 or 12435.1-20:1 for arrays
        match = re.match(r'^\s*(\d+)(?:\.\S+)?\s*$', stdout)
        if match:
            return int(match.group(1))

        # fallback for non-terse output
        match = re.search(r'Your job(?:-array)?\s+(\d+)', stdout)
        if match:
            return int(match.group(1))

        raise RuntimeError(f'Failed to parse: {stdout!r}')

    @staticmethod
    def _parse_status(stdout: str) -> pd.DataFrame:
        
        columns = [
            'job_id',
            'job_state',
            'runtime',
            'node_id',
            'array_idx'
        ]

        rows = []
        for line in stdout.splitlines():
            line = line.rstrip()

            # ignore headings and separators
            if not re.match(r'^\d+\s', line):
                continue

            tokens = line.split()
            if len(tokens) < 8:
                raise ValueError(f'Failed to parse qstat line: {line!r}')

            job_id = str(int(tokens[0]))
            job_state = _parse_job_state(tokens[4])

            if job_state == 'RUNNING':
                runtime = _compute_runtime(tokens[5], tokens[6])
                node_id = _parse_node_id(tokens[7])
            else:
                runtime = None
                node_id = None

            try:
                array_idx = int(tokens[9])
            except IndexError:
                array_idx = None

            rows.append({
                'job_id': job_id,
                'job_state': job_state,
                'runtime': runtime,
                'node_id': node_id,
                'array_idx': array_idx
            })

        df = pd.DataFrame(rows, columns=columns)
        df['array_idx'] = pd.to_numeric(df['array_idx'], errors='coerce')
        return df

    @staticmethod
    def _parse_history(stdout: str) -> pd.DataFrame:
        
        columns = [
            'job_id',
            'job_state',
            'runtime',
            'node_id',
            'array_idx'
        ]

        rows = []
        for block in re.split(r'^=+\s*$', stdout, flags=re.MULTILINE):

            record = {}
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    record[parts[0]] = parts[1]

            job_id = record.get('jobnumber')
            if job_id is None:
                continue

            failed = _parse_int(record.get('failed'))
            status = _parse_int(record.get('exit_status'))
            job_state = (
                'COMPLETE' if failed == status == 0 else 'FAILED'
            )

            runtime = record.get('ru_wallclock')
            if runtime is not None:
                runtime = _format_runtime(runtime)

            node_id = record.get('hostname')
            if node_id is not None:
                node_id = _parse_node_id(node_id)

            array_idx = record.get('taskid')
            if array_idx in {'undefined', 'NONE', '0'}:
                array_idx = None

            rows.append({
                'job_id': str(int(job_id)),
                'job_state': job_state,
                'runtime': runtime,
                'node_id': node_id,
                'array_idx': array_idx
            })

        df = pd.DataFrame(rows, columns=columns)
        df['array_idx'] = pd.to_numeric(df['array_idx'], errors='coerce')

        if not df.empty:
            df = df.drop_duplicates(['job_id', 'array_idx'], keep='last')

        return df


def _as_command(cmd, *args, **kwargs) -> str:
    argv = [cmd]

    for arg in args:
        argv.append(shell.as_arg_value(arg))

    for key, value in kwargs.items():
        option = f'-{key}'
        if value is False:
            continue
        elif value is True:
            argv.append(option)
        else:
            argv.extend([option, shell.as_arg_value(value)])

    return ' '.join(argv)


def _parse_job_state(state: str) -> str:
    if 'E' in state:
        return 'FAILED'
    if 'r' in state or 't' in state:
        return 'RUNNING'
    return 'PENDING'


def _compute_runtime(date: str, time: str):
    try:
        dt_init = dt.datetime.strptime(
            f'{date} {time}', '%m/%d/%Y %H:%M:%S'
        )
    except ValueError:
        return None

    dt_curr = dt.datetime.now()
    seconds = (dt_curr - dt_init).total_seconds()
    return _format_runtime(seconds)


def _format_runtime(seconds: str):
    try:
        seconds = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return None

    days, seconds = divmod(seconds, 60 * 60 * 24)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)

    result = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{days}-{result}' if days > 0 else result


def _parse_node_id(hostname: str) -> str:
    try:
        return hostname.split('@', 1)[1].split('.', 1)[0]
    except IndexError:
        raise ValueError(f'Failed to parse host: {hostname!r}')


def _parse_int(value, default=-1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


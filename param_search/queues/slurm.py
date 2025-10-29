import os, shlex
from . import base
from .. import utils, shell


class SlurmQueue(base.BaseQueue):

    @staticmethod
    def _submit_cmd(path, *args, **kwargs):
        abs_path = os.path.abspath(path)
        work_dir = os.path.dirname(abs_path)
        logs_dir = os.path.join(work_dir, 'logs')

        if 'array' in kwargs:
            stdout_pat = os.path.join(logs_dir, '%A_%a.out')
            stderr_pat = os.path.join(logs_dir, '%A_%a.err')
        else:
            stdout_pat = os.path.join(logs_dir, '%j.out')
            stderr_pat = os.path.join(logs_dir, '%j.err')

        return shell.as_command(
            'sbatch',
            *args,
            output=shlex.quote(stdout_pat),
            error=shlex.quote(stderr_pat),
            **kwargs
        ) + ' ' + shlex.quote(abs_path)

    @staticmethod 
    def _status_cmd(job_ids, *args, **kwargs):

        # slurm throws an error if you try to check the
        #   status of a single job that's not in the queue
        if len(job_ids) == 1:
            job_ids.append('1')

        return shell.as_command(
            'squeue',
            *args,
            format=shlex.quote('%j %i %P %T %R %M %Z'),
            job=job_ids,
            **kwargs
        )

    @staticmethod
    def _cancel_cmd(*args, **kwargs):
        return 'scancel ' + _as_command_args(*args, **kwargs)

    @staticmethod
    def _parse_submit(stdout):
        import re
        pat = r'^Submitted batch job (\d+)( on cluster .+)?\n$'
        match = re.match(pat, stdout)
        if match:
            return int(match.group(1))
        raise RuntimeError(f'failed to parse: {stdout:r}')

    @staticmethod
    def _parse_status(stdout):
        import pandas as pd
        data = _parse_status_data(stdout)
        df = pd.DataFrame(data).rename(columns={
            'NAME': 'job_name',
            'JOBID': 'job_id',
            'PARTITION': 'partition',
            'STATE': 'job_state',
            'NODELIST(REASON)': 'node_id',
            'TIME': 'runtime',
            'WORK_DIR': 'work_dir'
        })

        # parse array idx from job id
        if len(df) > 0:
            df['job_id'] = df['job_id'].astype(str) + '_'
            df[['job_id', 'array_idx']] = \
                df['job_id'].str.split('_', n=1, expand=True)
        else:
            df['array_idx'] = []

        df['job_id'] = df['job_id'].astype(str)
        df['array_idx'] = df['array_idx'].replace('', float('nan')).map(pd.to_numeric)

        # parse reason from node id
        #node_re = re.compile(r'^(.*)\((.+)\)?$')
        #matches = [node_re.match(x) for x in df['node_id']]
        #for i, m in enumerate(matches):
        #    if m is None:
        #        print(df.iloc[i])
        #df['node_id'] = [m.group(1) for m in matches]
        #df['reason'] = [m.group(2) for m in matches]
        return df


def _parse_status_data(stdout):
    from .. import text
    start = stdout.index('NAME')
    lines = stdout[start:].split('\n')
    columns = lines[0].split(' ')
    data = {c: [] for c in columns}
    for line in filter(len, lines[1:]):
        fields = text.paren_split(line, sep=' ')
        if len(fields) != len(columns):
            raise ValueError(f'failed to parse: {stdout}')
        for i, field in enumerate(fields):
            data[columns[i]].append(field)
    return data

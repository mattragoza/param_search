import sys, os, re, shlex, tqdm
from contextlib import contextmanager
import numpy as np
import pandas as pd
from subprocess import Popen, PIPE
from multiprocessing import Pool
from functools import partial

from .common import read_file, write_file, non_string_iterable
from . import job_output

verbose = False


def as_cmd_arg_value(val):
    if non_string_iterable(val):
        return ','.join(map(str, val))
    elif not isinstance(val, str):
        return str(val)
    else:
        return val


def as_positional_cmd_arg(val):
    return ' ' + as_cmd_arg_value(val)


def as_optional_cmd_arg(key, val):
    val = as_cmd_arg_value(val)
    if len(key) == 1:
        return ' -{} {}'.format(key, val)
    else:
        return ' --{}={}'.format(key, val)


def as_cmd_args(*args, **kwargs):
    cmd = ''
    for v in args:
        cmd += as_positional_cmd_arg(v)
    for k, v in kwargs.items():
        cmd += as_optional_cmd_arg(k, v)
    return cmd


def run_subprocess(cmd, stdin=None, work_dir=None, verbose=False):
    '''
    Run cmd as a subprocess with the given stdin,
    from the given work_dir, and return (stdout, stderr).
    '''
    if verbose:
        print(repr(work_dir), repr(cmd))

    if sys.platform == 'win32':
        args = cmd
    else:
        args = shlex.split(cmd)

    proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=work_dir)
    stdout, stderr = proc.communicate(stdin)

    if isinstance(stdout, bytes):
        stdout = stdout.decode()

    if isinstance(stderr, bytes):
        stderr = stderr.decode()

    return stdout, stderr


def call_subprocess(cmd, stdin=None, work_dir=None, verbose=False):
    '''
    Run cmd as a subprocess and raise an exc-
    eption if there is any stderr.
    '''
    stdout, stderr = run_subprocess(cmd, stdin, work_dir, verbose)
    if stderr:
        raise SubprocessError(stderr)
    return stdout


class SubprocessError(RuntimeError):
    '''
    Raised when a subprocess fails, and contains stderr.
    '''
    pass


class JobQueue(object):
    '''
    An abstract interface for communicating with a job
    scheduling system such as Slurm or PBS Torque.
    '''
    @classmethod
    def get_submit_cmd(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_status_cmd(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def parse_submit_out(cls, stdout):
        raise NotImplementedError

    @classmethod
    def parse_status_out(cls, stdout):
        raise NotImplementedError

    @classmethod
    def submit_job_script(cls, job_file, *args, **kwargs):
        job_file = os.path.abspath(job_file)
        work_dir = os.path.dirname(job_file)
        submit_cmd = cls.get_submit_cmd(job_file, *args, **kwargs)
        submit_out = call_subprocess(submit_cmd, work_dir=work_dir)
        return cls.parse_submit_out(submit_out)

    @classmethod
    def submit_job_scripts(cls, job_files, *args, verbose=False, **kwargs):
        job_ids = []
        for job_file in tqdm.tqdm(job_files, file=sys.stdout):
            job_id = cls.submit_job_script(job_file, *args, **kwargs)
            job_ids.append(job_id)
        return job_ids

    @classmethod
    def get_job_status(cls, *args, **kwargs):

        cmd = cls.get_status_cmd(*args, **kwargs)

        class JobStatus(pd.DataFrame):

            @property
            def cmd(self):
                return cmd 

            def status(self, verbose=False):
                out = call_subprocess(self.cmd)
                new_stat = cls.parse_status_out(out, type(self))

                # merge new status with current status
                self['job_state'] = np.nan
                self['node_id'] = np.nan
                self['runtime'] = np.nan
                super().update(new_stat)
                df = self
                work_dir = df['work_dir'].astype(str)
                job_id = df['job_id'].astype(str)
                stdout_file = work_dir + '/' + job_id + '.stdout'
                stderr_file = work_dir + '/' + job_id + '.stderr'
                df['stdout'] = stdout_file.apply(
                    job_output.read_stdout_file, verbose=verbose
                )
                df['stderr'] = stderr_file.apply(
                    job_output.read_stderr_file, verbose=verbose
                )
                return self

        out = call_subprocess(cmd)
        return cls.parse_status_out(out, JobStatus)

    @classmethod
    def cancel_job(cls, *args, **kwargs):
        cmd = cls.get_cancel_cmd(*args, **kwargs)
        return call_subprocess(cmd)


def run_multiprocess(cmds, work_dirs=None, verbose=False, n_proc=1):
    '''
    Run cmds in parallel using multiprocessing.
    '''
    if work_dirs is None:
        work_dirs = [None] * len(cmds)

    def run_command(args):
        cmd, work_dir = args
        return run_subprocess(cmd, work_dir=work_dir, verbose=verbose)

    args = zip(cmds, work_dirs)
    if n_proc == 1:
        return (run_command(a) for a in args)
    else:
        pool = Pool(n_proc)
        return pool.imap(run_command, args)


class LocalQueue(object):

    @classmethod
    def submit_job_scripts(cls, job_files, verbose=False, n_proc=None):
        cmds = [f'bash {os.path.abspath(f)}' for f in job_files]
        work_dirs = [os.path.dirname(f) for f in job_files]
        return run_multiprocess(
            cmds, work_dirs, verbose=verbose, n_proc=n_proc
        ), work_dirs

    @classmethod
    def get_job_status(cls, results):
        results, work_dirs = results
        results = tqdm.tqdm(results, total=len(work_dirs), file=sys.stdout)
        status = pd.DataFrame(results, columns=['stdout', 'stderr'])
        status['work_dir'] = work_dirs
        return status


class SlurmQueue(JobQueue):

    @classmethod
    def get_submit_cmd(cls, job_files, *args, **kwargs):
        return 'sbatch' + as_cmd_args(job_files, *args, **kwargs)

    @classmethod 
    def get_status_cmd(cls, job_ids, *args, **kwargs):

        # slurm throws an error if you try to check the
        #   status of a single job that's not in the queue
        if len(job_ids) == 1:
            job_ids.append(1)

        return r'squeue --format="%j %i %P %T %R %M %Z"' + \
            as_cmd_args(*args, **kwargs, job=job_ids)

    @classmethod
    def get_cancel_cmd(cls, *args, **kwargs):
        return 'scancel' + as_cmd_args(*args, **kwargs)

    @classmethod
    def parse_submit_out(cls, stdout):
        return int(re.match(
            r'^Submitted batch job (\d+)( on cluster .+)?\n$',
            stdout
        ).group(1))

    @classmethod
    def parse_status_out(cls, stdout, job_stat):

        # parse the output table
        stdout = stdout[stdout.index('NAME'):]
        lines = stdout.split('\n')
        columns = lines[0].split(' ')
        col_data = {c: [] for c in columns}
        for line in filter(len, lines[1:]):
            fields = paren_split(line, sep=' ')
            for i, field in enumerate(fields):
                col_data[columns[i]].append(field)

        df = pd.DataFrame(col_data).rename(columns={
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

        df['job_id'] = df['job_id'].astype(int)
        df['array_idx'] = df['array_idx'] \
            .replace('', float('nan')).map(pd.to_numeric)

        # parse reason from node id
        #node_re = re.compile(r'^(.*)\((.+)\)?$')
        #matches = [node_re.match(x) for x in df['node_id']]
        #for i, m in enumerate(matches):
        #    if m is None:
        #        print(df.iloc[i])
        #df['node_id'] = [m.group(1) for m in matches]
        #df['reason'] = [m.group(2) for m in matches]
        return job_stat(df)


class TorqueQueue(JobQueue):

    @classmethod
    def get_submit_cmd(cls, job_file, array_idx=None):
        cmd = 'qsub ' + job_file
        if array_idx is not None:
            cmd += ' -t {}'.format(as_cmd_arg(array_idx))
        return cmd

    @classmethod
    def get_status_cmd(cls, job_names):
        return 'qstat'

    @classmethod
    def parse_submit_out(cls, stdout, job_stat):
        try:
            return int(re.match(
                r'^(\d+)\.n198\.dcb\.private\.net\n$',
                stdout
            ).group(1))
        except Exception as e:
            print(stdout)
            raise

    @classmethod
    def parse_status_out(cls, stdout):
        raise NotImplementedError('TODO')


def paren_split(string, sep):
    '''
    Split string by instances of sep character that are
    outside of balanced parentheses.
    '''
    fields = []
    last_sep = -1
    esc_level = 0
    for i, char in enumerate(string):
        if char in sep and esc_level == 0:
            fields.append(string[last_sep+1:i])
            last_sep = i
        elif char == '(':
            esc_level += 1
        elif char == ')':
            if esc_level > 0:
                esc_level -= 1
            else:
                raise ValueError('missing open parentheses')
    if esc_level == 0:
        fields.append(string[last_sep+1:])
    else:
        raise ValueError('missing close parentheses')
    return fields


def parse_qstat(buf, job_delim='\n\n', field_delim='\n    ', index_name=None):
    '''
    Parse the stdout of either qstat -f or pbsnodes and return it in a
    data frame indexed either by job ID or node ID, respectively.
    '''
    assert buf, 'nothing to parse'
    all_job_data = []
    for job_buf in filter(len, buf.split(job_delim)):
        job_data = dict()
        for field_buf in filter(len, job_buf.split(field_delim)):
            if not job_data:
                if index_name is None:
                    name, value = field_buf.split(': ', 1)
                    index_name = name
                else:
                    name, value = index_name, field_buf.split(': ', 1)[-1]
            else:
                name, value = field_buf.split(' = ', 1)
            job_data[name] = value.replace('\n\t', '')
        all_job_data.append(job_data)
    return pd.DataFrame(all_job_data).set_index(index_name)

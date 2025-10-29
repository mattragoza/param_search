import sys, os, shlex
from subprocess import Popen, PIPE

from . import utils


def as_command(cmd, *args, **kwargs):
    argv = [cmd]
    for v in args:
        argv.extend(_as_positional_arg(v))
    for k, v in kwargs.items():
        argv.extend(_as_optional_arg(k, v))
    return ' '.join(argv)


def _as_positional_arg(val):
    return [_as_arg_value(val)]


def _as_optional_arg(key, val):
    if val is True: # flag
        return [f'--{key}']
    return [f'--{key}={_as_arg_value(val)}']


def _as_arg_value(val):
    if utils.is_iterable(val, string_ok=False):
        s = ','.join([str(v) for v in val])
    else:
        s = str(val)
    return shlex.quote(s)


def _decode(s):
    return s.decode() if isinstance(s, bytes) else s


def run_subprocess(cmd, stdin=None, stdout=PIPE, stderr=PIPE, work_dir=None):
    '''
    Run cmd as a subprocess in work_dir with stdin.
    Return stdout, raise stderr as SubprocessError.
    '''
    utils.log(cmd)

    if sys.platform == 'win32':
        args = cmd
    else:
        args = shlex.split(cmd)

    proc = Popen(args, stdin=PIPE, stdout=stdout, stderr=stderr, cwd=work_dir)

    stdout, stderr = map(_decode, proc.communicate(stdin))
    utils.log(stdout)

    if stderr:
        raise SubprocessError(stderr)

    return stdout


def run_multiprocess(cmds, work_dirs=None, n_proc=1):
    '''
    Run cmds in parallel using multiprocessing.
    '''
    import multiprocessing as mp

    if work_dirs is None:
        work_dirs = [None] * len(cmds)

    def run_command(args):
        cmd, work_dir = args
        return run_subprocess(cmd, work_dir=work_dir)

    args = zip(cmds, work_dirs)
    if n_proc == 1:
        return (run_command(a) for a in args)
    else:
        pool = mp.Pool(n_proc)
        return pool.imap(run_command, args)


class SubprocessError(RuntimeError):
    '''
    Raised when a subprocess outputs to stderr.
    '''
    pass

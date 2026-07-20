import sys, os, shlex
from subprocess import Popen, PIPE

from . import utils


def as_command(cmd, *args, **kwargs):
    argv = [cmd]

    for arg in args:
        argv.append(as_arg_value(arg))

    for key, value in kwargs.items():
        argv.extend(as_optional_arg(key, value))

    return ' '.join(argv)


def as_optional_arg(key, value):
    option = f'--{key}'
    if value is False:
        return []
    elif value is True:
        return [option]
    return [f'{option}={as_arg_value(value)}']


def as_arg_value(val):
    if utils.is_iterable(val, string_ok=False):
        s = ','.join([str(v) for v in val])
    else:
        s = str(val)
    return shlex.quote(s)


def decode_bytes(s) -> str:
    return s.decode() if isinstance(s, bytes) else s


def run_subprocess(
    cmd,
    stdin=None,
    stdout=PIPE,
    stderr=PIPE,
    work_dir=None,
) -> str:
    '''
    Run cmd as a subprocess in work_dir with stdin.
    Return stdout, raise stderr as SubprocessError.
    '''
    proc = Popen(
        cmd if sys.platform == 'win32' else shlex.split(cmd),
        stdin=PIPE,
        stdout=stdout,
        stderr=stderr,
        cwd=work_dir
    )
    utils.log(cmd)

    stdout, stderr = map(decode_bytes, proc.communicate(stdin))
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

    assert len(work_dirs) == len(cmds)

    def run_command(args):
        cmd, work_dir = args
        return run_subprocess(cmd, work_dir=work_dir)

    args = zip(cmds, work_dirs)
    _map = mp.Pool(n_proc).imap if n_proc > 1 else map
    return _map(run_command, args)


class SubprocessError(RuntimeError):
    '''
    Raised when a subprocess outputs to stderr.
    '''
    pass


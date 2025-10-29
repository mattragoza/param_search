import sys, os, shlex


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
    val = _as_arg_value(val)
    if len(key) == 1:
        return [f'-{key}', val]
    else:
        return [f'--{key}={val}']


def _as_arg_value(val):
    if isinstance(val, str):
        return val
    elif hasattr(val, '__iter__'):
        return ','.join(map(str, val))
    return str(val)


def _decode(s):
    return s.decode() if isinstance(s, bytes) else s


def run_subprocess(cmd, stdin=None, work_dir=None):
    '''
    Run cmd as a subprocess in work_dir with stdin.
    Return stdout, raise stderr as SubprocessError.
    '''
    from subprocess import Popen, PIPE

    if sys.platform == 'win32':
        args = cmd
    else:
        args = shlex.split(cmd)

    proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=work_dir)

    stdout, stderr = map(_decode, proc.communicate(stdin))

    if stderr:
        raise SubprocessError(stderr)

    return stdout


class SubprocessError(RuntimeError):
    '''
    Raised when a subprocess outputs to stderr.
    '''
    pass


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


def paren_split(s, sep):
    '''
    Split string by instances of sep character that are
    outside of balanced parentheses.
    '''
    fields = []
    last_sep = -1
    esc_level = 0
    for i, char in enumerate(s):
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
        fields.append(s[last_sep+1:])
    else:
        raise ValueError('missing close parentheses')
    return fields


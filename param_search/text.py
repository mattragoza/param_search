import sys, os, re
from functools import cache


def read_tail(path, max_lines=20):
	import pandas as pd
	if not os.path.exists(path):
		return pd.NA
	lines = []
	for line in open_reversed(path, max_lines):
		lines.append(line)
	return '\n'.join(reversed(lines))


def parse_stdout(
    stdout, break_pat=None, ignore_pat=None, output_pat=r'^([a-z0-9]+).*bridges2'
):
    # compile parsing regexes
    output_re = as_compiled_regex(output_pat)
    if break_pat:
        break_re = as_compiled_regex(break_pat)
    if ignore_pat:
        ignore_re = as_compiled_regex(ignore_pat)

    # read and parse stdout lines
    n_parsed = 0
    for line in stdout:
        if break_pat and break_re.match(line):
            return '', n_parsed
        if ignore_pat and ignore_re.match(line):
            continue
        m = output_re.match(line)
        n_parsed += 1
        if m:
            return m.group(1), n_parsed
    return '', n_parsed


def parse_stderr(
    stderr,
    break_pat=r'^WARNING',
    ignore_pat=None,
    error_pat=r'^(.*(Error|Exception|error|fault|failed|Errno|Killed|No such file|command not found).*)$',
    **kwargs
):
    # compile parsing regexes
    error_re = as_compiled_regex(error_pat)
    if break_pat:
        break_re = as_compiled_regex(break_pat)
    if ignore_pat:
        ignore_re = as_compiled_regex(ignore_pat)

    # read and parse stderr lines
    n_parsed = 0
    for line in stderr:
        if break_pat and break_re.match(line):
            return '', n_parsed
        if ignore_pat and ignore_re.match(line):
            continue
        m = error_re.match(line)
        n_parsed += 1
        if m:
            return m.group(1), n_parsed
    return '', n_parsed


def open_reversed(
	path,
	max_lines=100,
	buffer_size=8192,
	encoding='utf-8',
	errors='replace'
):
    '''
    Generate file lines in reverse order up to max_lines.
    '''
    def _decode(b):
    	b = b[:-1] if b.endswith(b'\r') else b
    	return b.decode(encoding, errors=errors)

    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        curr_pos = f.tell()
        emitted = 0
        carry = b''

        while curr_pos > 0 and emitted < max_lines:
        	read_size = min(curr_pos, buffer_size)
        	curr_pos -= read_size
        	f.seek(curr_pos)
        	chunk = f.read(read_size)
        	if not chunk:
        		break

        	data = chunk + carry
        	parts = data.split(b'\n')
        	carry = parts[0] # first part may be partial line
        	for line in reversed(parts[1:]):
        		yield _decode(line)
        		emitted += 1
        		if emitted >= max_lines:
        			return

        if curr_pos == 0 and carry and emitted < max_lines:
        	yield _decode(carry)


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
            fields.append(s[last_sep+1:i])
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



def parse_stdout_file(stdout_file, verbose=False, reverse=False, **kwargs):
    # check that stdout file exists
    if not os.path.isfile(stdout_file):
        if verbose:
            print(stdout_file, file=sys.stderr)
        return np.nan
    if reverse:
        lines = open_reversed(stdout_file)
    else:
        lines = open(stdout_file)
    stdout, n_parsed = parse_stdout(lines, **kwargs)
    if verbose:
        print(f'{n_parsed} stdout lines parsed')
    return stdout


def parse_stderr_file(stderr_file, verbose=False, reverse=True, **kwargs):
    # check that stderr file exists
    if not os.path.isfile(stderr_file):
        if verbose:
            print(stderr_file, file=sys.stderr)
        return np.nan
    if reverse:
        lines = open_reversed(stderr_file)
    else:
        lines = open(stderr_file)
    stderr, n_parsed = parse_stderr(lines, **kwargs)
    if verbose:
        print(f'{n_parsed} stderr lines parsed')
    return stderr


import sys, os, re
from functools import cache
from . import utils


ERROR_TOKENS = [
    r'\bTraceback\b',
    r'\bException\b',
    r'\bError\b',
    r'\b(segmentation fault|core dumped)\b',
    r'\bKilled\b',
    r'\bErrno\s+\d+\b',
    r'\b(no such file|command not found)\b',
    r'\b(out of memory|outofmemory)\b'
]

def build_line_regex(tokens):
    alts = '|'.join(tokens)
    full = rf'^(?P<line>.*?(?:{alts}).*)$'
    return re.compile(full, re.IGNORECASE|re.MULTILINE)


ERROR_RE = build_line_regex(ERROR_TOKENS)


def parse_error_line(text: str):
    if utils.missing(text):
        return None
    m = ERROR_RE.search(text)
    if not m:
        return None
    line = m.group('line')
    return line


def read_tail(path, max_lines=20):
    import pandas as pd
    if utils.missing(path):
        return pd.NA
    if not os.path.exists(path):
        return pd.NA
    lines = []
    for line in open_reversed(path, max_lines):
        lines.append(line)
    return '\n'.join(reversed(lines))


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


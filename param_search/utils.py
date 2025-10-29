

_VERBOSE = True


def set_verbose(val):
    global _VERBOSE
    _VERBOSE = bool(val)


def log(msg):
    if _VERBOSE:
        print(msg, flush=True)


def read_file(p):
    with open(p, 'r') as f:
        return f.read()


def write_file(p, buf):
    with open(p, 'w') as f:
        f.write(buf)


def is_iterable(obj, string_ok=False):
    if isinstance(obj, str):
        return string_ok
    return hasattr(obj, '__iter__')


def as_list(obj):
    return list(obj) if is_iterable(obj) else [obj]


def namespace(dct, name):
    return {f'{name}.{k}': v for k, v in dct.items()}


def hash_func(s):
    import hashlib
    return hashlib.blake2b(s, digest_size=8).hexdigest()


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



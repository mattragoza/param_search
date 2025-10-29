

VERBOSE = True


def set_verbose(val):
    global VERBOSE
    VERBOSE = bool(val)


def log(msg):
    if VERBOSE:
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


def hash_params(params: dict) -> str:
    import json, hashlib
    to_hash = json.dumps(params, sort_keys=True).encode()
    return hashlib.blake2b(to_hash, digest_size=8).hexdigest()






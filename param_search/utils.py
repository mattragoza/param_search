from pathlib import Path
import re

LOG_RE = re.compile(r"(?P<job_id>\d+)(?:_(?P<array_idx>\d+))?\.(?P<ext>out|err)$")

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


def find_job_ids(
    log_dir: str,
    reverse=False,
    required_exts={'out', 'err'}
):
    from collections import defaultdict

    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []

    last_mtime = {}
    exts_found = defaultdict(set)

    for p in log_dir.iterdir():
        m = LOG_RE.match(p.name)
        if not m:
            continue
        if m.group('array_idx'):
            jid = f'{m.group("job_id")}_{m.group("array_idx")}'
        else:
            jid = m.group('job_id')

        last_mtime[jid] = max(last_mtime.get(jid, 0), p.stat().st_mtime)
        exts_found[jid].add(m.group('ext'))

    candidates = [j for j in last_mtime if required_exts.issubset(exts_found[j])]
    return sorted(candidates, key=lambda j: (last_mtime[j], j), reverse=reverse)

import sys, os, re
from pathlib import Path

LOG_RE = re.compile(r"(?P<job_id>\d+)(?:_(?P<array_idx>\d+))?\.(?P<ext>out|err)$")
VERBOSE = True


def set_verbose(val):
    global VERBOSE
    VERBOSE = bool(val)


def log(msg):
    if VERBOSE:
        print(msg, flush=True)


def warn(msg):
    print(msg, file=sys.stderr, flush=True)


def make_dirs(d: Path):
    if not d.is_dir():
        log(f'mkdir {d}')
        d.mkdir(parents=True)


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


def missing(val):
    import pandas as pd
    return pd.isna(val) or str(val).strip() == ''


def safe_load(path, *args, **kwargs):
    import pandas as pd
    if missing(path):
        return None, 'missing path'
    if not os.path.isfile(path):
        return None, 'file not found'
    if os.stat(path).st_size == 0:
        return None, 'file is empty'
    try:
        data = pd.read_csv(path, *args, **kwargs)
        return data, None
    except Exception as e:
        return None, 'read csv failed'


def atomic_write(path, df):
    import os, tempfile, shutil
    path = Path(path)
    make_dirs(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path) # atomic
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def get_base_dir(jobs):
    import pandas as pd
    if 'base_dir' in jobs.columns:
        vals = pd.unique(jobs['base_dir'].dropna().astype(str))
        if len(vals) == 1:
            return Path(vals[0]).resolve()
        raise RuntimeError('jobs do not have a unique base_dir')
    raise RuntimeError('jobs have no base_dir columns')


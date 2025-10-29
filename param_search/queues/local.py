import os, shlex
from . import base
from .. import shell


class LocalQueue(base.BaseQueue):

    def __init__(self):
        self._records = {} # job_id -> record dict
        self._counter = 0  # monotonic job_id

    def _next_id(self):
        self._counter += 1
        return f'{self._counter:d}'

    def submit(self, paths):
        job_ids = []
        for p in paths:
            abs_path = os.path.abspath(p)
            work_dir = os.path.dirname(abs_path)
            log_dir = os.path.join(work_dir, 'logs')

            stdout_path = os.path.join(log_dir, 'f{j}.out')
            stderr_path = os.path.join(log_dir, 'f{j}.err')
            
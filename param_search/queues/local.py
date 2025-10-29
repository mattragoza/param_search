import os

from . import base
from .. import shell


class LocalQueue(base.BaseQueue):

    def submit(self, job_files, verbose=False, n_proc=None):
        cmds = [f'bash {os.path.abspath(f)}' for f in job_files]
        work_dirs = [os.path.dirname(f) for f in job_files]
        return shell.run_multiprocess(
            cmds, work_dirs, verbose=verbose, n_proc=n_proc
        ), work_dirs

    def status(self, results):
        import pandas as pd
        results, work_dirs = results
        status = pd.DataFrame(results, columns=['stdout', 'stderr'])
        status['work_dir'] = work_dirs
        return status



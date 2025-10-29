import os

from .. import shell


class BaseQueue:
    '''
    Abstract interface for batch job backends.
    '''
    @staticmethod
    def _submit_cmd(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _status_cmd(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _cancel_cmd(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _parse_submit(stdout):
        raise NotImplementedError

    @staticmethod
    def _parse_status(stdout):
        raise NotImplementedError

    @staticmethod
    def _parse_cancel(stdout):
        raise NotImplementedError

    def submit(self, paths, *args, dry_run=False, **kwargs):
        job_ids = []
        for p in paths:
            abs_path = os.path.abspath(p)
            work_dir = os.path.dirname(abs_path)
            cmd = self._submit_cmd(abs_path, *args, **kwargs)
            if dry_run:
                print(cmd, flush=True)
                continue
            else:
                out = shell.run_subprocess(cmd, work_dir=work_dir)
            job_id = self._parse_submit(out)
            job_ids.append(job_id)
        return job_ids

    def status(self, job_ids, *args, **kwargs):
        cmd = self._status_cmd(job_ids, *args, **kwargs)
        out = shell.run_subprocess(cmd)
        return self._parse_status(out)

    def cancel(self, *args, **kwargs):
        cmd = self._cancel_cmd(*args, **kwargs)
        out = shell.run_subprocess(cmd)
        return self._parse_cancel(out)


from . import params, job_files, job_output, job_queues, results
from .job_files import setup_job_files as setup
from .job_output import get_job_metrics as metrics
from .params import ParamSpace
from .results import plot


def queue(use='slurm'):
    if use == 'local':
        return job_queues.LocalQueue
    elif use == 'slurm':
        return job_queues.SlurmQueue
    elif use == 'torque':
        return job_queues.TorqueQueue


def submit(
    template,
    name_format,
    param_space,
    work_dir='.',
    use='slurm',
    verbose=False,
    merge=True,
    job_ids=None,
    **kwargs,
):
    import pandas as pd

    # create list of job scripts to submit
    params = list(param_space)
    job_files = setup(template, name_format, params, work_dir)

    if job_ids is None: # submit jobs to queue
        job_ids = queue(use).submit_job_scripts(
            job_files, verbose=verbose, **kwargs
        )

    if verbose:
        print(job_ids)

    # return job status data frame
    status = queue(use).get_job_status(job_ids)

    if merge: # combine job params and job status
        params = pd.DataFrame(params)
        if use == 'local':
            return pd.concat([params, status], axis=1)
        else:
            return type(status)(params.merge(status, on='job_name'))
    else: # just return status
        return status


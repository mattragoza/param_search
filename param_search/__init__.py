from . import params, job_files, job_output, job_queues, results
from .job_files import setup_job_files as setup
from .job_output import get_job_metrics as metrics
from .params import ParamSpace
from .results import plot


def submit(
    template,
    name_format,
    param_space,
    work_dir='.',
    use='slurm',
    verbose=False,
    **kwargs,
):
    import pandas as pd

    # create list of job scripts to submit
    params = list(param_space)
    job_files = setup(template, name_format, params, work_dir)

    if use == 'local':
        queue = job_queues.LocalQueue
    elif use == 'slurm':
        queue = job_queues.SlurmQueue
    elif use == 'torque':
        queue = job_queues.TorqueQueue

    # submit jobs to queue
    job_ids = queue.submit_job_scripts(job_files, verbose=verbose, **kwargs)

    # return job status data frame
    status = queue.get_job_status(job_ids)

    params = pd.DataFrame(params)

    if use == 'local':
        return pd.concat([params, status], axis=1)
    else:
        return type(status)(params.merge(status, on='job_name'))

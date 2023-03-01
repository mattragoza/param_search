from . import params, job_files, job_output, job_queues, results

from .params import ParamSpace
from .job_files import setup_job_files as setup
from .job_output import read_job_metrics as metrics
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
    **kwargs,
):
    import pandas as pd

    # create list of job scripts to submit
    if isinstance(param_space, pd.DataFrame):
        params = param_space
        job_files = param_space.work_dir + '/job.sh'
    else:
        params = list(param_space)
        job_files = setup(template, name_format, params, work_dir)

    # submit jobs to queue
    job_ids = queue(use).submit_job_scripts(
        job_files, verbose=verbose, **kwargs
    )
    if verbose:
        print(job_ids)

    # get initial job status data frame
    status = queue(use).get_job_status(job_ids)

    if merge: # combine job params and job status
        params = pd.DataFrame(params)
        if use == 'local':
            return pd.concat([params, status], axis=1)
        else:
            return params.merge(status, on='job_name')

    else: # just return job status
        return status


def status(
    jobs,
    use='slurm',
    parse_stdout=False,
    parse_stderr=False,
    verbose=False
):
    from numpy import nan
    jobs = jobs.reset_index()
    
    # get new job status data frame from queue
    new_status = queue(use).get_job_status(jobs['job_id'])
    new_status = new_status.set_index('job_id')

    # merge new status with old status
    #   we set as nan the values in the old status
    #   that should be updated by the new status
    status = jobs.set_index('job_id')
    status['job_state'] = nan
    status['node_id'] = nan
    status['runtime'] = nan
    status.update(new_status)

    # read additional status from output files
    work_dir = status['work_dir'].astype(str)
    job_id = status.index.astype(int).astype(str)
    stdout_file = work_dir + '/' + job_id + '.stdout'
    stderr_file = work_dir + '/' + job_id + '.stderr'
    status['stdout'] = stdout_file.apply(
        job_output.read_stdout_file, parse=parse_stdout, verbose=verbose
    )
    status['stderr'] = stderr_file.apply(
        job_output.read_stderr_file, parse=parse_stderr, verbose=verbose
    )

    return status

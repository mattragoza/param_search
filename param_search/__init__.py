from . import params, job_files, job_queues, job_output, results
from .params import ParamSpace
from .job_files import setup_job_files as setup
from .job_queues import SlurmQueue, TorqueQueue
from .job_output import get_job_outputs
from .job_output import get_job_errors, parse_stderr
from .job_output import get_job_metrics
from .results import plot


# convenient aliases
submit = SlurmQueue.submit_job_scripts
status = SlurmQueue.get_job_status
cancel = SlurmQueue.cancel_job
output = get_job_outputs
errors = get_job_errors
metrics = get_job_metrics


def submit_local(cmd, name_format, param_space, work_dir=None):
    import sys, tqdm
    import pandas as pd

    # create list of commands to run
    cmds = []
    param_space = list(param_space)
    for job_params in param_space:
        job_name = name_format.format(hash=hash(job_params), **job_params)
        job_params['job_name'] = job_name
        cmds.append(cmd.format(**job_params))

    # run the commands in parallel and put results in data frame
    param_space = pd.DataFrame(param_space)
    res = job_queues.run_multiprocess(cmds, work_dir=work_dir)
    res = tqdm.tqdm(res, total=len(param_space), file=sys.stdout)
    results = pd.DataFrame(res, columns=['stdout', 'stderr'])
    return pd.concat([param_space, results], axis=1)

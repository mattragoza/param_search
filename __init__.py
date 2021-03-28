from . import params, job_files, job_queues, job_output, results
from .params import ParamSpace
from .job_files import setup_job_files as setup
from .job_queues import SlurmQueue, TorqueQueue
from .job_output import get_job_errors as errors
from .job_output import get_job_metrics as metrics
from .results import plot

# TODO allow switching to TorqueQueue
submit = SlurmQueue.submit_job_scripts
status = SlurmQueue.get_job_status

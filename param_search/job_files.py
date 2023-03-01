import sys, os, re, argparse
import itertools

from . import params
from .common import read_file, write_file


def setup_job_files(
    template,
    name_format,
    param_space,
    work_dir,
    dry_run=False,
    **kwargs
):
    '''
    Write a job file in a separate sub dir of expt_dir
    for every set of params in param_space, by formatting
    template_file with the params. Name the created dirs
    by formatting name_format with the params.
    '''
    job_base = 'job.sh'
    job_files = []
    for job_params in param_space:

        for k, v in kwargs.items():
            job_params[k] = v
        job_name = name_format.format(
            hash=hash(job_params), **job_params
        )
        job_params['job_name'] = job_name
        job_dir = os.path.join(work_dir, job_name)

        if not os.path.isdir(job_dir):
            os.makedirs(job_dir)

        job_file = os.path.join(job_dir, job_base)
        if not dry_run:
            write_job_file(job_file, template, job_params)
        job_files.append(job_file)

    return job_files


def write_job_file(job_file, template, job_params):
    '''
    Write a job file to job_file by filling in
    template with job_params.
    '''
    params_str = params.format_params(job_params, line_start='# ')
    write_file(job_file, template.format(job_params=params_str, **job_params))

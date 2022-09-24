import sys, os, re, shutil, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

from .params import read_params
from .job_queues import SlurmQueue


def open_reversed(fname, buf_size=8192):
    '''
    Iterate over lines of fname in reverse
    order, up to a max of tail lines. This
    enables more efficient output parsing.
    '''
    with open(fname) as f:
        segment = None
        offset = 0
        f.seek(0, os.SEEK_END)
        file_size = rem_size = f.tell()

        while rem_size > 0:
            offset = min(file_size, offset + buf_size)
            f.seek(file_size - offset)
            buffer = f.read(min(rem_size, buf_size))
            rem_size -= buf_size
            lines = buffer.split('\n')

            # the first line of the buffer is probably not a complete line
            # so save it and append it to the last line of the next buffer
            if segment is not None:
                # if the previous chunk starts at the beginning of a line
                # do not concat the segment to the last line of new chunk
                # instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment

            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]

        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


def read_file(file_, verbose=False):
    # check that file exists
    if not os.path.isfile(file_):
        if verbose:
            print(file_, file=sys.stderr)
        return np.nan
    with open(file_) as f:
        return f.read()


def read_stdout_file(stdout_file, parse=False, verbose=False):
    if parse:
        return parse_stdout_file(stdout_file, verbose=verbose)
    else:
        return read_file(stdout_file, verbose=verbose)


def read_stderr_file(stderr_file, parse=False, verbose=False):
    if parse:
        return parse_stderr_file(stderr_file, verbose=verbose)
    else:
        return read_file(stderr_file, verbose=verbose)


def parse_stdout_file(
    stdout_file,
    ignore_pat=None,
    output_pat=r'^(\[.*\].*)',
    verbose=False
):
    # check that stdout file exists
    if not os.path.isfile(stdout_file):
        if verbose:
            print(stdout_file, file=sys.stderr)
        return np.nan

    lines = open_reversed(stdout_file)
    return parse_stdout(lines, ignore_pat, output_pat)


def parse_stdout(
    stdout,
    ignore_pat=None,
    output_pat=r'^(\[.*\].*)',
):
    # convert to reversed lines
    if isinstance(stdout, str):
        stdout = reversed(stdout.split('\n'))

    # compile parsing regexes
    if ignore_pat:
        ignore_re = re.compile(ignore_pat)
    output_re = re.compile(output_pat)

    # read and parse lines in reverse order
    for line in stdout:
        if ignore_pat is None or not ignore_re.match(line):
            m = output_re.match(line)
            if m:
                return m.group(1)


def parse_stderr_file(
    stderr_file,
    break_pat=None,
    ignore_pat=None,
    error_pat=r'^(.*(Error|Exception|error|fault|failed|Errno|Killed).*)$',
    verbose=False,
):
    # check that stderr file exists
    if not os.path.isfile(stderr_file):
        if verbose:
            print(stderr_file, file=sys.stderr)
        return np.nan

    lines = open_reversed(stderr_file)
    return parse_stderr(lines, break_pat, ignore_pat, error_pat)


def parse_stderr(
    stderr,
    break_pat=None,
    ignore_pat=None,
    error_pat=r'^(.*(Error|Exception|error|fault|failed|Errno|Killed).*)$',
):
    # convert to reversed lines
    if isinstance(stderr, str):
        stderr = reversed(stderr.split('\n'))

    # compile parsing regexes
    if break_pat:
        break_re = re.compile(break_pat)
    if ignore_pat:
        ignore_re = re.compile(ignore_pat)
    error_re = re.compile(error_pat)

    for line in stderr:
        if break_pat is not None and break_re.match(line):
            break
        if ignore_pat is None or not ignore_re.match(line):
            m = error_re.match(line)
            if m:
                return m.group(1)


def as_compiled_re(obj):
    '''
    Compile obj as regex pattern if needed.
    '''
    return obj if hasattr(obj, 'match') else re.compile(obj)


def match_files_in_dir(dir, pat, verbose=False):
    '''
    Iterate through files in dir that match pat.
    '''
    if verbose:
        print(os.path.join(dir, pat))
    pat = as_compiled_re(pat)
    for file in os.listdir(dir):
        m = pat.match(file)
        if m is not None:
            yield m


def get_metrics_from_dir(
    work_dir, metric_pat, verbose=False, sep=' ', **kwargs
):
    '''
    Read the metrics files from a
    given working directory.

    Args:
        work_dir: Directory containing metrics files.
        metric_pat: Regex for detecting metrics file.
        **kwargs: Args passed to pandas.read_csv().
    Returns:
        pandas.DataFrame of metrics.
    '''
    metrics = []
    for m in match_files_in_dir(work_dir, metric_pat, verbose):
        metric_file = os.path.join(work_dir, m.group(0))
        metrics.append(pd.read_csv(metric_file, sep=sep, **kwargs))

    return pd.concat(metrics)


def read_job_metrics(
    jobs, metric_pat=r'(.+)\.(csv|(.*)metrics)', verbose=False, **kwargs
):
    '''
    Read metrics files for a set of jobs.

    Args:
        jobs: pandas.DataFrame of jobs.
        metric_pat: Regex for detecting metrics files.
        verbose: Print verbose output.
        **kwargs: Args passed to pandas.read_csv().
    Returns:
        pd.DataFrame from merging jobs with metrics.
    '''
    metrics = []
    for i, job in jobs.iterrows():
        try:
            job_metrics = get_metrics_from_dir(
                job.work_dir, metric_pat, verbose, **kwargs
            )
            job_metrics['job_name'] = job.job_name
            metrics.append(job_metrics)
        except ValueError as e:
            print(job.job_name, e, file=sys.stderr)

    return jobs.merge(pd.concat(metrics), on='job_name')


def print_array_indices(idx_set):
    s = get_array_indices_string(idx_set)
    print(s)


def get_array_indices_string(idx_set):
    s = ''
    last_idx = None
    skipping = False
    for idx in sorted(idx_set):
        if last_idx is None:
            s += str(idx)
        elif idx == last_idx + 1:
            skipping = True
        else: # gap
            if skipping:
                skipping = False
                s += '-' + str(last_idx)
            s += ',' + str(idx)
        last_idx = idx
    if skipping:
        s += '-' + str(last_idx)
    return s


def parse_array_indices_str(s):
    idx_pat = re.compile(r'^(\d+)(-(\d+))?$')
    indices = []
    for field in s.split(','):
        m = idx_pat.match(field)
        idx_start = int(m.group(1))
        if m.group(2):
            idx_end = int(m.group(3))
            indices.extend(range(idx_start, idx_end+1))
        else:
            indices.append(idx_start)
    return set(indices)


def find_job_ids(job_dir, stderr_pat=r'(\d+).stderr'):
    '''
    Find job ids that have been submitted by
    parsing stderr file names in job_dir.
    '''
    job_ids = []
    for f in os.listdir(job_dir):
        if os.path.isdir(os.path.join(job_dir, f)):
            m = re.match(r'^(\d+)', f)
        else:
            m = re.match(stderr_pat, f)
        if m:
            job_id = int(m.group(1))
            job_ids.append(job_id)

    return sorted(job_ids)


def find_job_id(job_dir, stderr_pat=r'(\d+).stderr'):
    '''
    Find the latest job id in job_dir.
    '''
    return max(find_job_ids(job_dir, stderr_pat))



def print_last_error(job_dir, stderr_pat):

    last_job_id = -1
    last_stderr_file = None
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = os.path.join(job_dir, m.group(0))
        job_id = int(m.group(1))
        if job_id > last_job_id:
            last_job_id = job_id
            last_stderr_file = stderr_file

    if last_stderr_file is None:
        print('no error file')
        return

    error = read_stderr_file(last_stderr_file)
    print(last_stderr_file + '\t' + str(error))


def find_submitted_array_indices(job_dir, stderr_pat):
    '''
    Find array indices and job ids that have been
    submitted by parsing stderr file names in job_dir.
    '''
    submitted = set()
    job_ids = []
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = m.group(0)
        job_id = int(m.group(1))
        array_idx = int(m.group(2))
        job_ids.append(job_id)
        submitted.add(array_idx)

    return submitted, job_ids


def copy_back_from_scr_dir(job_dir, scr_dir, copy_back_pat):
    '''
    Copy back output files from scr_dir to job_dir.
    '''
    copied = []
    for m in match_files_in_dir(scr_dir, copy_back_pat):
        copy_back_file = m.group(0)
        src_file = os.path.join(scr_dir, copy_back_file)
        dst_file = os.path.join(job_dir, copy_back_file)
        shutil.copyfile(src_file, dst_file)
        copied.append(dst_file)

    return copied


def find_completed_array_indices(job_dir, output_pat, read=False):
    '''
    Find array_indices that have completed by parsing output
    files in job_dir, also optionally read and return job dfs.
    '''
    job_dfs = []
    completed = set()
    for m in match_files_in_dir(job_dir, output_pat):
        array_idx = int(m.group(2))
        completed.add(array_idx)
        if read:
            output_file = os.path.join(job_dir, m.group(0))
            job_df = pd.read_csv(output_file, sep=' ')
            job_df['job_name'] = os.path.split(job_dir)[-1]
            job_df['array_idx'] = array_idx
            job_dfs.append(job_df)

    return completed, job_dfs


def print_errors_for_array_indices(job_dir, stderr_pat, indices):

    stderr_files = defaultdict(list)
    for m in match_files_in_dir(job_dir, stderr_pat):
        stderr_file = m.group(0)
        job_id = int(m.group(1))
        array_idx = int(m.group(2))
        if array_idx in indices:
            stderr_files[array_idx].append((job_id, stderr_file))

    for array_idx in sorted(indices):
        if not stderr_files[array_idx]:
            print('no error file for array_idx {}'.format(array_idx))
            continue
        job_id, stderr_file = sorted(stderr_files[array_idx])[-1]
        stderr_file = os.path.join(job_dir, stderr_file)
        error = read_stderr_file(stderr_file)
        print(stderr_file + '\t' + str(error))

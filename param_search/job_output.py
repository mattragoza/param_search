import sys, os, re, shutil, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

from .params import read_params
from .job_queues import SlurmQueue


def as_compiled_re(obj):
    '''
    Compile obj as regex pattern if needed.
    '''
    return obj if hasattr(obj, 'match') else re.compile(obj)


def match_files_in_dir(dir, pat):
    '''
    Iterate through files in dir that match pat.
    '''
    pat = as_compiled_re(pat)
    for file in os.listdir(dir):
        m = pat.match(file)
        if m is not None:
            yield m


def open_reversed(fname, buf_size=8192):
    '''
    Iterate over lines of fname in reverse
    order, up to a max of tail lines.
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


def read_stdout_file(
    stdout_file,
    ignore_pat=None,
    output_pat=r'^(.*)'
):
    #print('.', end='')
    if not os.path.isfile(stdout_file):
        return np.nan
    if ignore_pat:
        ignore_re = re.compile(ignore_pat)
    output_re = re.compile(output_pat)
    with open(stdout_file) as f:
        return f.read()
    for line in open_reversed(stdout_file):
        if not ignore_pat or not ignore_re.match(line):
            m = output_re.match(line)
            if m:
                return m.group(1)


def read_stderr_file(
    stderr_file,
    break_pat=None,
    ignore_pat=None,
    error_pat=r'^(.*(Error|Exception|error|fault|failed|Errno|Killed).*)$'
):
    #print('.', end='')
    if not os.path.isfile(stderr_file):
        return np.nan
    if break_pat:
        break_re = re.compile(break_pat)
    if ignore_pat:
        ignore_re = re.compile(ignore_pat)
    with open(stderr_file) as f:
        return f.read()
    error_re = re.compile(error_pat)
    for line in open_reversed(stderr_file):
        if break_pat and break_re.match(line):
            break
        if not ignore_pat or not ignore_re.match(line):
            m = error_re.match(line)
            if m:
                return m.group(1)


def parse_stderr(
    stderr,
    break_pat=None,
    ignore_pat=None,
    error_pat=r'^(.*(Error|Exception|error|fault|failed|Errno|Killed).*)$'
):
    if break_pat:
        break_re = re.compile(break_pat)
    if ignore_pat:
        ignore_re = re.compile(ignore_pat)
    error_re = re.compile(error_pat)
    for line in reversed(stderr.split('\n')):
        if break_pat and break_re.match(line):
            break
        if not ignore_pat or not ignore_re.match(line):
            m = error_re.match(line)
            if m:
                return m.group(1)


def get_job_errors(job_files, stderr_pat=r'(\d+).stderr'):
    '''
    Parse the latest errors for a set of job_files.
    '''
    errors = []
    for job_file in job_files:
        error = get_job_error(job_file, stderr_pat)
        errors.append(error)

    return errors


def get_job_output(job_file, stdout_pat):
    '''
    Parse the latest output for job_file.
    '''
    job_dir = os.path.dirname(job_file)
    stdout_files = []
    for m in match_files_in_dir(job_dir, stdout_pat):
        stdout_file = m.group(0)
        job_id = int(m.group(1))
        stdout_files.append((job_id, stdout_file))

    job_id, stdout_file = sorted(stdout_files)[-1]
    stdout_file = os.path.join(job_dir, stdout_file)
    output = read_stdout_file(stdout_file)
    return output


def get_job_outputs(job_files, stdout_pat=r'(\d+).stdout'):
    '''
    Parse the latest outputs for a set of job_files.
    '''
    outputs = []
    for job_file in job_files:
        output = get_job_output(job_file, stdout_pat)
        outputs.append(output)

    return outputs


def get_metrics_from_dir(work_dir, metric_pat):
    '''
    Read the metrics files from a
    given working directory.

    Args:
        work_dir: Directory containing metrics files.
        metric_pat: Regex for detecting metrics file.
    Returns:
        pandas.DataFrame of metrics.
    '''
    metrics = []
    for m in match_files_in_dir(work_dir, metric_pat):
        metric_file = os.path.join(work_dir, m.group(0))
        metrics.append(pd.read_csv(metric_file, sep=' '))

    return pd.concat(metrics)


def get_job_metrics(jobs, metric_pat=r'(.+)\.(.*)metrics', verbose=False):
    '''
    Read metrics files for a set of jobs.

    Args:
        jobs: pandas.DataFrame of jobs.
        metric_pat: Regex for detecting metrics files.
        verbose: Print verbose output.
    Returns:
        pd.DataFrame from merging jobs with metrics.
    '''
    metrics = []
    for i, job in jobs.iterrows():
        if verbose:
            print(job.job_name)
        try:
            job_metrics = get_metrics_from_dir(job.work_dir, metric_pat)
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


def find_job_ids(job_dir, stderr_pat):
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

    return job_ids


def read_job_output(job_dir, output_pat):
    '''
    Find job ids that have been submitted by
    parsing stderr file names in job_dir.
    '''
    job_dfs = []
    for m in match_files_in_dir(job_dir, output_pat):
        output_file = os.path.join(job_dir, m.group(0))
        print(output_file)
        job_df = pd.read_csv(output_file, sep=' ', error_bad_lines=False)
        job_df['job_name']  = os.path.split(job_dir)[-1]
        try:
            array_idx = int(m.group(2))
            job_df['array_idx'] = array_idx
        except:
            pass
        job_dfs.append(job_df)

    return job_dfs


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



def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('job_scripts', nargs='+')
    parser.add_argument('--job_type')
    parser.add_argument('--array_job', default=False, action='store_true')
    parser.add_argument('--submitted', default=None)
    parser.add_argument('--copy_back', '-c', default=False, action='store_true')
    parser.add_argument('--print_indices', '-i', default=False, action='store_true')
    parser.add_argument('--print_errors', '-e', default=False, action='store_true')
    parser.add_argument('--resub_errors', '-r', default=False, action='store_true')
    parser.add_argument('--output_file', '-o')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    all_job_dfs = []
    for job_script in args.job_scripts:

        assert os.path.isfile(job_script), 'file ' + job_script + ' does not exist'

        if args.job_type is None: # infer from file name
            if 'fit' in job_script:
                args.job_type = 'fit'
            elif 'train' in job_script:
                args.job_type = 'train'

        # determine relevant files based on job type
        if args.job_type == 'train':
            job_script_pat = re.compile(r'(.*)_train.sh')
            output_ext = 'training_output'
            copy_back_exts = [
                'model', 'solver','caffemodel', 'solverstate', 'training_output', 'png', 'pdf'
            ]
        elif args.job_type == 'fit':
            job_script_pat = re.compile(r'(.*)_fit.sh')
            output_ext = 'gen_metrics'
            copy_back_exts = [
                'types', 'model', 'caffemodel', 'dx', 'sdf', 'channels', 'latent', 'pymol', 'gen_metrics'
            ]

        # for array jobs, get output for any array indices present
        if args.array_job:
            stderr_pat = re.compile(r'slurm-(\d+)_(\d+)\.err$')
            output_pat = re.compile(r'(.*)_(\d+)\.' + output_ext + '$')
            copy_back_pat = re.compile(r'(.*)_(\d+)\.' + '({})$'.format('|'.join(copy_back_exts)))
        else:
            stderr_pat = re.compile(r'slurm-(\d+)\.err$')
            output_pat = re.compile(r'(.*)\.' + output_ext + '$')
            copy_back_pat = re.compile(r'(.*)\.' + '({})$'.format('|'.join(copy_back_exts)))

        print(job_script)
        job_dir = os.path.dirname(job_script)

        if args.array_job:

            if args.submitted is not None:
                submitted = parse_array_indices_str(args.submitted)
                job_ids = find_job_ids(job_dir, stderr_pat)
            else:
                submitted, job_ids = find_submitted_array_indices(job_dir, stderr_pat)
            n_submitted = len(submitted)

            if n_submitted == 0:
                print('none submitted')
                continue

            completed, job_dfs = find_completed_array_indices(job_dir, output_pat, read=args.output_file)
            n_completed = len(completed)

            if args.output_file:
                all_job_dfs.extend(job_dfs)

            incomplete = submitted - completed
            n_incomplete = len(incomplete)

            if args.print_indices:
                print('n_submitted = {} ({})'.format(n_submitted, get_array_indices_string(submitted)))
                print('n_completed = {} ({})'.format(n_completed, get_array_indices_string(completed)))
                print('n_incomplete = {} ({})'.format(n_incomplete, get_array_indices_string(incomplete)))
            else:
                print('n_submitted = {}'.format(n_submitted))
                print('n_completed = {}'.format(n_completed))
                print('n_incomplete = {}'.format(n_incomplete))

            if args.print_errors:
                print_errors_for_array_indices(job_dir, stderr_pat, indices=incomplete)

            if args.copy_back:

                last_job_id = sorted(job_ids)[-1]
                scr_dir = os.path.join(job_dir, str(last_job_id))

                copied = copy_back_from_scr_dir(job_dir, scr_dir, copy_back_pat)
                n_copied = len(copied)
                print('copied {} files from {}'.format(n_copied, last_job_id))

            if args.resub_errors: # resubmit incomplete jobs

                for m in match_files_in_dir(job_dir, job_script_pat):
                    job_script = os.path.join(job_dir, m.group(0))
                    SlurmQueue.submit_job(
                        job_script,
                        work_dir=job_dir,
                        array_idx=get_array_indices_string(incomplete)
                    )

        else:
            job_ids = find_job_ids(job_dir, stderr_pat)

            if args.output_file:
                job_dfs = read_job_output(job_dir, output_pat)
                all_job_dfs.extend(job_dfs)

            if args.print_errors:
                print_last_error(job_dir, stderr_pat)

            if args.copy_back:

                for last_job_id in sorted(job_ids):
                    scr_dir = os.path.join(job_dir, str(last_job_id))
                    copied = copy_back_from_scr_dir(job_dir, scr_dir, copy_back_pat)
                    n_copied = len(copied)
                    print('copied {} files from {}'.format(n_copied, last_job_id))

    if args.output_file:
        if all_job_dfs:
            job_df = pd.concat(all_job_dfs)
            pd.set_option('display.max_columns', 100)
            print(job_df.groupby('job_name').mean())
            job_df.to_csv(args.output_file, sep=' ')
            print('concatenated metrics to {}'.format(args.output_file))
        else:
            print('nothing to concatenate')


if __name__ == '__main__':
    main(sys.argv[1:])


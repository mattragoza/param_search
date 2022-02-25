## `param_search`: Hyperparameter search with cluster computing

This Python package makes it easy to launch lots of jobs with different parameters to be evaluated on a computer cluster. It also has features to monitor the status of running or completed jobs, aggregate their output, and create visualizations.

### Dependencies

- numpy
- scipy
- pandas
- parse
- tqdm

### Basic usage

Here is how you can launch an experiment on a Slurm cluster in about 10 lines of code:

```python
import param_search as ps

# define a basic job template and name format
template = '''\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -o %J.stdout
#SBATCH -e %J.stderr
pwd
product=`python3 -c "print({term1}*{term2})"`
quotient=`python3 -c "print({term1}/{term2})"`

echo "product quotient" > {job_name}.metrics
echo "$product $quotient" >> {job_name}.metrics
'''
name = 'job_{term1}_{term2}'

# define the ranges of parameters to evaluate
param_space = ps.ParamSpace(term1=range(4), term2=range(4))

# submit jobs for each parameter setting
jobs = ps.submit(template, name, param_space, use='slurm')

# check the stdout and stderr
print(jobs.iloc[0].stdout)
print(jobs.iloc[0].stderr)

# read in output metrics
metrics = ps.metrics(jobs)

# plot metrics against parameters
fig = ps.plot(metrics, x=['term1', 'term2'], y=['product', 'quotient'])

```

### Jobs templates and name formats

These are simple python formatting strings. When jobs are submitted, the templates and name formats are filled in with the parameter settings:

```python
job_hash = hash(job_params)
job_name = name.format(**job_params, hash=job_hash)
job_content = template.format(**job_params, job_name=job_name, hash=job_hash)
```

### Parameter spaces

A parameter space is a set of parameters and ranges of values they can take on. They are a sublcass of `collections.OrderedDict` where keys represent names of parameters and values are the ranges of the parameter values.

```python
param_space = ps.ParamSpace(
	a=range(10),
	b=1e-3,
	c='hello',
	d=[True, False],
)
```

Note that all values are promoted to non-string iterables on creation (i.e., they are put into a singleton list) to support iteration.

A `ParamSpace` can be iterated over to produce parameter assignments from the Cartesian product of the value ranges, or it can be randomly sampled, with or without replacement. The iterates are `Params` objects, which have the same keys as the `ParamSpace` but each value is a single element of the value range.

```python
for p in param_space:
	print(p)

assert len(param_space) == 20
assert len(param_space.sample(5, replace=False)) == 5
```

Parameter spaces can also be combined with algebraic operations of addition and multiplication, which allows certain subsets of parameters to be grouped together when iterating or sampling.

```python
param_space_a = ps.ParamSpace(type='a', param=[1,2,3])
param_space_b = ps.ParamSpace(type='b', param=[4,5,6])
param_space_c = ps.ParamSpace(other_param=1.5)

# addition iterates over the param spaces sequentially
#   which requires that the keys be the identical
param_space_ab = param_space_a + param_space_b

# scalar multiplication just repeats the parameter ranges
#   which can be useful for balanced sampling of two subspaces
param_space_ab = 10 * param_space_ab

# multiplying two spaces produces their Cartesian product
#   which requires that the keys be disjoint sets
param_space_abc = param_space_ab * param_space_c

assert len(param_space_abc) == 60
```

### Submitting jobs to a queue

The following queues are supported: `LocalQueue`, `SlurmQueue`, and `TorqueQueue`.

See [this Jupyter notebook](https://github.com/mattragoza/param_search/blob/master/example.ipynb) for a walkthrough.

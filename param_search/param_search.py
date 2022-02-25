import sys, os, re
import pandas as pd

from .job_files import setup_job_files
from .job_queues import SlurmQueue


class ParamSearch(object):
	'''
	A class that manages the submission and tracking
	of multiple parameter spaces.
	'''

	def __init__(self, expt_file):
		# we want to be able to recover the state of the search when the object is created, so we first check if an experiment file is present

		self.expt_file = expt_file
		if os.path.isfile(self.expt_file):
			self.jobs = pd.read_csv(self.expt_file, sep=' ')
		else:
			self.jobs = None

	def setup(self, space, name_format, template_file):
		job_files = setup_job_files

	def submit(self, space):
		# submit jobs for the given space
		new_jobs = pd.DataFrame(list(space))
		new_jobs['job_id'] = SlurmQueue.submit_job_scripts()


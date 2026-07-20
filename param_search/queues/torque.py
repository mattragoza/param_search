from typing import List
import pandas as pd
from . import base


class TorqueQueue(base.BaseQueue):
	
	@staticmethod
	def _submit_cmd(path, *args, **kwargs):
		pass

	@staticmethod
	def _status_cmd(job_ids, *args, **kwargs):
		pass

	@staticmethod
	def _history_cmd(job_ids, *args, **kwrags):
		pass

	@staticmethod
	def _cancel_cmd(*args, **kwargs):
		pass

	@staticmethod
	def _parse_submit(stdout: str) -> List[str]:
		pass

	@staticmethod
	def _parse_status(stdout: str) -> pd.DataFrame:
		pass

	@staticmethod
	def _parse_history(stdout: str) -> pd.DataFrame:
		pass


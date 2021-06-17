import sys, os, pytest, time
sys.path.insert(0, '.')
from param_search import job_queues as q


class TestCmdArgs(object):

    def test_arg_val(self):
        assert q.as_cmd_arg_value([]) == ''
        assert q.as_cmd_arg_value([1]) == '1'
        assert q.as_cmd_arg_value([1,2]) == '1,2'
        assert q.as_cmd_arg_value([1,2,3]) == '1,2,3'
        assert q.as_cmd_arg_value(None) == 'None'
        assert q.as_cmd_arg_value(0) == '0'
        assert q.as_cmd_arg_value(1) == '1'
        assert q.as_cmd_arg_value(1.0) == '1.0'
        assert q.as_cmd_arg_value('') == ''
        assert q.as_cmd_arg_value('asdf') == 'asdf'
        assert q.as_cmd_arg_value(['a','s','d','f']) == 'a,s,d,f'

    def test_pos_arg(self):
        assert q.as_positional_cmd_arg(0) == ' 0'
        assert q.as_positional_cmd_arg('asdf') == ' asdf'

    def test_opt_arg(self):
        assert q.as_optional_cmd_arg('a', 0) == ' -a 0'
        assert q.as_optional_cmd_arg('A', 0) == ' -A 0'
        assert q.as_optional_cmd_arg('asdf', 0) == ' --asdf=0'
        assert q.as_optional_cmd_arg('ASDF', 0) == ' --ASDF=0'

    def test_cmd_args(self):
        assert q.as_cmd_args() == ''
        assert q.as_cmd_args(1, 2, 3) == ' 1 2 3'
        assert q.as_cmd_args([1, 2, 3]) == ' 1,2,3'
        assert q.as_cmd_args(a=1, b=2, c=3) == ' -a 1 -b 2 -c 3'
        assert q.as_cmd_args(asdf=1, qwer=2) == ' --asdf=1 --qwer=2'
        assert q.as_cmd_args(asdf=[1, 2, 3]) == ' --asdf=1,2,3'
        assert q.as_cmd_args(asdf=[1, 2, 3], Q=4) == ' --asdf=1,2,3 -Q 4'

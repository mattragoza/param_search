import sys, os, pytest, time
from collections import OrderedDict
sys.path.insert(0, '.')
from param_search.params import AbstractParamSpace, ParamSpace

test_dims = [
    dict(),
    dict(a=[]),
    dict(a=1),
    dict(a=[1]),
    dict(a=[1,2]),
    dict(a=1, b=2),
    dict(a=[1,2], b=[3,4])
]

class TestParamSpace(object):

    @pytest.fixture(params=test_dims)
    def dims(self, request):
        return request.param

    def test_init(self, dims):
        params = ParamSpace(**dims)
        assert isinstance(params, AbstractParamSpace)
        assert isinstance(params, OrderedDict)
        assert params.Params.space is params
        assert set(params.keys()) == set(dims.keys())
        for k, v in dims.items():
            if isinstance(v, (list, tuple)):
                assert params[k] == v
            else:
                assert params[k] == [v]

    def test_iterate(self, dims):
        params = ParamSpace(**dims)
        i = 0
        for p in params:
            assert p.keys() == params.keys()
            for k, v in p.items():
                assert v in params[k]
            i += 1
        assert i == len(params)

    def test_sum(self, dims):
        space = ParamSpace(**dims)
        space + space

    def test_product(self, dims):
        space = ParamSpace(**dims)
        space + space

    def test_scalar(self, dims):
        space = ParamSpace(**dims)
        10 * space

    @pytest.fixture(params=range(6))
    def n_dims(self, request):
        return request.param

    @pytest.fixture(params=range(6))
    def n_vals(self, request):
        return request.param

    def test_benchmark_len(self, n_dims, n_vals):
        dims = {str(k): range(n_vals) for k in range(n_dims)}
        params = ParamSpace(**dims)
        t_start = time.time()
        assert len(params) == n_vals**n_dims
        t_delta = time.time() - t_start
        assert t_delta <= 1e-2, 'too slow ({:.4f}s)'.format(t_delta)

    def test_benchmark_iter(self, n_dims, n_vals):
        dims = {str(k): range(n_vals) for k in range(n_dims)}
        params = ParamSpace(**dims)
        t_start = time.time()
        assert len(list(params)) == n_vals**n_dims
        t_delta = time.time() - t_start
        assert t_delta <= 1e-2, 'too slow ({:.4f}s)'.format(t_delta)

    def test_benchmark_sample0(self, n_dims, n_vals):
        dims = {str(k): range(n_vals) for k in range(n_dims)}
        params = ParamSpace(**dims)
        n = min(1000, len(params))
        t_start = time.time()
        samples = params.sample(n, replace=True)
        t_delta = time.time() - t_start
        assert len(samples) == n
        assert t_delta <= 1e-2, 'too slow ({:.4f}s)'.format(t_delta)

    def test_benchmark_sample0(self, n_dims, n_vals):
        dims = {str(k): range(n_vals) for k in range(n_dims)}
        params = ParamSpace(**dims)
        n = min(1000, len(params))
        t_start = time.time()
        samples = params.sample(n, replace=False)
        t_delta = time.time() - t_start
        assert len(samples) == n
        assert len(set(samples)) == n
        assert t_delta <= 1e-2, 'too slow ({:.4f}s)'.format(t_delta)
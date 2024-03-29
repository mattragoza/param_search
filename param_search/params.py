import re, ast, random
import itertools
from collections import OrderedDict

from .common import read_file, write_file, as_non_string_iterable


class Params(OrderedDict):
    '''
    A Params object represents a point in a ParamSpace,
    i.e. a particular assignment of values to parameters.
    '''
    space = None

    @classmethod
    def from_file(cls, params_file, line_start=''):
        return cls(read_params(params_file, line_start).items())

    def to_tuple(self):
        return tuple(v for v in self.values())

    def __hash__(self):
        return hash(self.to_tuple())


class AbstractParamSpace(object):

    def __mul__(self, other):
        if isinstance(other, int):
            return ParamSpaceScalar(other, self)
        else:
            return ParamSpaceProduct(self, other)

    def __rmul__(self, other):
        if isinstance(other, int):
            return ParamSpaceScalar(other, self)
        else:
            return ParamSpaceProduct(other, self)

    def __add__(self, other):
        return ParamSpaceSum(self, other)

    def __radd__(self, other):
        return ParamSpaceSum(other, self)

    def __and__(self, other):
        return self * other

    def __rand__(self, other):
        return other * self

    def __or__(self, other):
        return self + other

    def __ror__(self, other):
        return other + self

    def __getitem__(self, idx):
        return list(self)[idx]

    def sample(self, k, replace=False):
        '''
        Return a random sample of k Params
        from the parameter space, with or
        without replacement.
        '''
        if replace: # don't care if we get repeats
            return [self.sample_one() for i in range(k)]

        elif k*100 < len(self):
            # just sample with replacement and
            # keep trying when we get a repeat
            samples = [None] * k
            sampled = set()
            for i in range(k):
                params = self.sample_one()
                while params in sampled:
                    params = self.sample_one()
                sampled.add(params)
                samples[i] = params
            return samples

        else: # let Guido deal with it
            return random.sample(list(self), k)


class ParamSpaceScalar(AbstractParamSpace):

    def __init__(self, const, space):
        self.const = const
        self.space = space

    def __iter__(self):
        for i in range(self.const):
            for params in self.space:
                yield params

    def __len__(self):
        return self.const * len(self.space)

    def keys(self):
        return list(self.space.keys())

    def sample_one(self):
        return self.space.sample_one()


class ParamSpaceProduct(AbstractParamSpace):

    def __init__(self, space1, space2):
        assert not any(k in space2.keys() for k in space1.keys()), \
            'cannot multiply spaces with common keys'
        self.space1 = space1
        self.space2 = space2

    def __iter__(self):
        for params1 in self.space1:
            for params2 in self.space2:
                params = params1.copy()
                params.update(params2)
                yield params

    def __len__(self):
        return len(self.space1) * len(self.space2)

    def keys(self):
        return list(self.space1.keys()) + list(self.space2.keys())

    def sample_one(self):
        params = self.space1.sample_one()
        params.update(self.space2.sample_one())
        return params


class ParamSpaceSum(AbstractParamSpace):

    def __init__(self, space1, space2):
        assert space1.keys() == space2.keys(), 'can only add spaces with same keys'
        self.space1 = space1
        self.space2 = space2

    def __iter__(self):
        for params in self.space1:
            yield params
        for params in self.space2:
            yield params

    def __len__(self):
        return len(self.space1) + len(self.space2)

    def keys(self):
        return self.space1.keys()

    def sample_one(self):
        n1 = len(self.space1)
        n2 = len(self.space2) 
        if random.random() > n1 / (n1 + n2):
            return self.space2.sample_one()
        else:
            return self.space1.sample_one()


class ParamSpace(AbstractParamSpace, OrderedDict):
    '''
    A ParamSpace defines ranges of possible values for a set
    of parameters. Iterating over the space produces Params
    from the Cartesian product of the parameter ranges.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Params = type('Params', (Params,), dict(space=self))

    @classmethod
    def from_file(cls, params_file):
        return cls(*read_params(params_file).items())

    def __setitem__(self, key, value):
        super().__setitem__(key, as_non_string_iterable(value))

    def __iter__(self):
        keys, values = self.keys(), self.values()
        for values in itertools.product(*values):
            yield self.Params(zip(keys, values))

    def __len__(self):
        # NOTE that 0**0 == 1 in python
        # in plain language, we still want
        # to submit a job, even if it has
        # no parameterized values
        n = 1
        for v in self.values():
            n *= len(v)
        return n

    def keys(self):
        return list(super().keys())

    def sample_one(self):
        return self.Params(
            (k, random.choice(v)) for k,v in self.items()
        )


def parse_params(buf, line_start='', converter=ast.literal_eval):
    '''
    Parse lines in buf as param = value pairs, filtering by an
    optional line_start pattern. After parsing, a converter
    function is applied to param values.
    '''
    params = OrderedDict()
    line_pat = r'^{}(\S+)\s*=\s*(.+)$'.format(line_start)
    for p, v in re.findall(line_pat, buf, re.MULTILINE):
        params[p] = converter(v)
    return params


def format_params(params, line_start='', converter=repr):
    '''
    Serialize params as param = value lines with an optional
    line_start string. Before formatting, a converter function
    is applies to param values.
    '''
    lines = []
    for p, v in params.items():
        lines.append('{}{} = {}\n'.format(line_start, p, converter(v)))
    return ''.join(lines)


def read_params(params_file, line_start='', converter=ast.literal_eval):
    '''
    Read lines from params_file as param = value pairs.
    '''
    buf = read_file(params_file)
    return parse_params(buf, line_start, converter)


def write_params(params_file, params):
    '''
    Write params to params_file as param = value lines.
    '''
    buf = format_params(params)
    write_file(params_file, buf)

import numpy as np
from collections.abc import Iterable
from collections import defaultdict

def points2length(p1, p2):
    return round(np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2), 4)

def str2bool(v, exception=None):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        if exception is None:
            raise ValueError('boolean value expected')
        else:
            raise exception

def concat(evaluations):
    """Receives an experiments' json and merges it's contents

    Params:
    -------
        * evaluations: list
        list of rollout evaluations

    Returns:
    --------
        * result: dict
        where `id` key certifies that experiments are the same
              `list` params are united
              `numeric` params are appended

    """
    result = {}
    result['id'] = []
    for qtb in evaluations:
        exid = qtb.pop('id')
        # can either be a rollout from the prev
        # exid or a new experiment
        if exid not in result['id']:
            result['id'].append(exid)

        for k, v in qtb.items():
            is_iterable = isinstance(v, list) or isinstance(v, dict)
            # check if integer fields match
            # such as cycle, save_step, etc
            if not is_iterable:
                if k in result:
                    if result[k] != v:
                        raise ValueError(
                            f'key:\t{k}\t{result[k]} and {v} should match'
                        )
                else:
                    result[k] = v
            else:
                if k not in result:
                    result[k] = defaultdict(list)
                result[k][exid].append(v)
    return result


def flatten(items, ignore_types=(str, bytes)):
    """

    Usage:
    -----
    > items = [1, 2, [3, 4, [5, 6], 7], 8]

    > # Produces 1 2 3 4 5 6 7 8
    > for x in flatten(items):
    >         print(x)

    Ref:
    ----

    David Beazley. `Python Cookbook.'
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x

def flatten2(items, ignore_types=(str, bytes)):
    return [item for item in flatten(items)]

# reduce a list of list by using the average
def reduce_mean(items):
    return [v for v in  map(np.mean, items)]

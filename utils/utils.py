from collections.abc import Iterable
from collections import defaultdict


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

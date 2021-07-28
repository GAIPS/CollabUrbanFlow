import numpy as np

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


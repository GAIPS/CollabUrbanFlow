'''WAVE: waiting and vehicles simply counts the number of cars in the incoming approaches.

    Converters objects transform traffic data into features.
    This state space generates signal plans that maximaze throughput.

    TODO:
        1. Remove reference to engine and associated implementation of __deepcopy__ method.

    References:
    ----------
    "Adaptive traffic signal control with actor-critic methods in a real-world traffic network with different traffic disruption events"
    Aslani, et al. 2017
'''
from copy import deepcopy

class WAVE(object):
    """Waiting and vehicle count state space"""
    def __init__(self, eng, phases_per_edges):
        self._eng = eng
        self._phases_per_edges = phases_per_edges

    def __call__(self, *args, **kwargs):
        # TODO: Handle tile coding.
        ret = []
        # key :=  edgeid_lanenum
        vehicles = self._eng.get_lane_vehicle_count()
        for edges in self._phases_per_edges.values():
            wv = sum(
                num_vehs for edge, num_vehs in vehicles.items() if is_edge_in_phase(edge, edges)
            )
            ret.append(int(wv))
        return ret

    def __deepcopy__(self, memo):
        cls = self.__class__
        ret = cls.__new__(cls)
        memo[id(self)] = ret
        # _eng is "unpickleble"
        for k, v in self.__dict__.items():
            if "_eng" not in k:
                setattr(ret, k, deepcopy(v, memo))
        return ret

def is_edge_in_phase(vehicles_edge, phase_edges):
    test_edge = vehicles_edge.split('_')[0]
    return test_edge in phase_edges

if __name__ == '__main__':
    # TODO: Implement a main that tells us how to use the object.
    pass

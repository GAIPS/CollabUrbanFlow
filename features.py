""" Maps an environment sensor data to RL features

    References:
    ----------
    [1] https://cityflow.readthedocs.io/en/latest/start.html
"""
from math import exp

def _delay(x): return exp(-5 * x)
def _volume(vehs, edges): return sum([vehs[edge] for edge in edges])

def compute_delay(phases, vehicles, velocities, speed_limits):
    """ Computes delay feature
        * A negative exponential of the deviation from a vehicles' speed 
        to the last.

    Parameters:
    -----------
    * phases: dict<str,<int, list<str>>
        Phases controlling the incoming approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    * vehicles: dict<str,list<str>>
        lane_id --> list of vehicle_ids: "get_lane_vehicles" [1]

    * velocities: dict<str,float>
        vehicle_id --> speed: "get_vehicle_speed" [1]

    * speed_limit: dict<str, float>
        lane_id --> speed_limit

    Returns:
    --------
    """

    features = {}
    for tl_id, phases  in phases.items():
        delays = []
            
        for phs, edges in phases.items():
            phase_delays = []
            for edge in edges:
                max_speed = speed_limits[tl_id][edge] 
                edge_velocities = [velocities[idv] for idv in vehicles[edge]]
                phase_delays += [_delay(vel / max_speed) for vel in edge_velocities]
            delays.append(round(float(sum(phase_delays)), 4))
        features[tl_id] = tuple(delays)
    return features

def compute_pressure(incoming_roadlinks, outgoing_roadlinks, vehicles):
    """ Computes pressure
        The pressure from the phase is the volume on the incoming approaches
        minus the volume on the outgoing approaches.

    Parameters:
    -----------
    * incoming_roadlinks: dict<str,<int, list<str>>
        Phases controlling the incoming approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    * outgoing_roadlinks: dict<str,<int, list<str>>
        Phases controlling the outgoing approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    * vehicles: dict<str,list<str>>
        lane_id --> list of vehicle_ids: "get_lane_vehicles" [1]

    Returns:
    --------
    * features: dict<str, tuple<int>> 
        Pressure on traffic light 
        intersection_code --> pressure (num_phases len)
    """
    features = {}
    assert incoming_roadlinks.keys() == outgoing_roadlinks.keys()
    gn0 = zip(incoming_roadlinks.keys(),
              incoming_roadlinks.values(),
              outgoing_roadlinks.values())

    vehicle_counts = {k: len(v) for k, v in vehicles.items()}
    def vn(x): return _volume(vehicle_counts, x)

    for ts_id, incoming, outgoing in gn0:
        assert incoming.keys() == outgoing.keys()
        gn1 = zip(incoming.values(), outgoing.values())
        features[ts_id] = tuple([
            vn(inco) - vn(outg) for inco, outg in gn1
        ])
    return features

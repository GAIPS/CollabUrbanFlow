""" Helpful cityflow's roadnet processing functions

    TODO: Change adjacency computation from phase inputs to lanes inputs
    References:
    ----------
    * https://cityflow.readthedocs.io/en/latest/roadnet.html
"""
import json
import numpy as np
from scipy.sparse import csr_matrix

from utils import points2length
from utils import flatten, flatten2

def get_phases(roadnet, phases_filter=[]):
    """ Forms a traffic light phase

    Parameters:
    -----------
    * roadnet: dict representing a roadnet in cityflow format.

    Returns:
    --------
    * incoming: dict<str,<int, list<str>>
        Phases controlling the incoming approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    * outgoing: dict<str,<int, list<str>>
        Phases controlling the outgoing approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    * speed_limit: dict<str, <dict<str, float>>
        Phases controlling the outgoing approaches.

    Usage:
    ------
    >>> import json
    >>> with open(roadnet_path, 'r') as f: roadnet = json.load(f)
    >>> incoming, outgoing, _ = get_phases(roadnet)
    """
    # Defines loop constants.
    incoming = {}
    outgoing = {}
    speed_limit = {}
    roads = roadnet['roads']
    intersections = [intr for intr in roadnet['intersections'] if not intr['virtual']]
    # Defines helper functions
    # fn: gets the name of roadlink y and intersection x
    # gn: gets the relative lanelink of roadlink y of intersection
    # sn: gets the max speed from road x, laneid y
    def fn(x, y): return x['roadLinks'][y]['startRoad']
    def gn(x, y): return x['roadLinks'][y]['laneLinks'][0]['startLaneIndex']
    def hn(x, y): return x['roadLinks'][y]['endRoad']
    def pn(x, y): return x['roadLinks'][y]['laneLinks'][0]['endLaneIndex']
    def sn(x, y):
        return [
            ln for rd in roads if rd['id'] == x for ln in rd['lanes']
        ][y]['maxSpeed']
    # filter phases
    def tn(x): return ((not any(phases_filter)) or (x in phases_filter))
    for intersection in intersections:
        lightphases = intersection['trafficLight']['lightphases']
        roadlinks_incoming = {}
        roadlinks_outgoing = {}
        max_speeds = {}
        phase_num = 0

        for linkids in lightphases:
            if len(linkids['availableRoadLinks']) > 0 and tn(phase_num):
                roadlinks = []
                roadlinks_reverse = []
                for linkid in linkids['availableRoadLinks']:
                    # startRoad should be the roadlinks_incoming links.
                    roadlink = fn(intersection, linkid)
                    laneid = gn(intersection, linkid)
                    roadlinks.append(f'{roadlink}_{laneid}')
                    max_speeds[roadlinks[-1]] = sn(roadlink, laneid)

                    # endRoad should be the roadlinks_outgoing links.
                    roadlink_reverse = hn(intersection, linkid)
                    laneid_reverse = pn(intersection, linkid)
                    roadlinks_reverse.append(f'{roadlink_reverse}_{laneid_reverse}')
                roadlinks_incoming[phase_num] = roadlinks
                roadlinks_outgoing[phase_num] = list(set(roadlinks_reverse))

            phase_num += int(len(linkids['availableRoadLinks']) > 0)
        incoming[intersection['id']] = roadlinks_incoming
        outgoing[intersection['id']] = roadlinks_outgoing

        speed_limit[intersection['id']] = max_speeds

    return incoming, outgoing, speed_limit


def get_lanes(roadnet):
    """Returns lanes incoming / lanes outgoing

    Parameters:
    -----------
    * roadnet: dict representing a roadnet in cityflow format.

    Returns:
    --------
    * lanes_incoming: dict<str, list<str>
        Phases controlling the incoming approaches.
        key: intersection_code <str> --> lane_ids <str>

    * lanes_outgoing: dict<str,list<str>>
        Phases controlling the outgoing approaches.
        key: intersection_code <str> --> lane_ids <str>

    Usage:
    ------
    >>> import json
    >>> with open(roadnet_path, 'r') as f: roadnet = json.load(f)
    >>> lanes_incoming, lanes_outgoing = get_lanes(roadnet)
    """
    # Defines loop constants.
    incoming = {}
    outgoing = {}
    roads = roadnet['roads']
    tl_ids = [intr['id'] for intr in roadnet['intersections'] if not intr['virtual']]
    # builds lane list from x road's information
    def bn(x): return [f"{x['id']}_{nl}" for nl in range(len(x['lanes']))]
    # iterates over roads that finish on intersection y.
    def fn(y): return [bn(road) for road in roads if road['endIntersection'] == y]
    # iterates over roads that start on intersection y.
    def sn(y): return [bn(road) for road in roads if road['startIntersection'] == y]
    incoming = {tl: sorted(flatten2(fn(tl))) for tl in tl_ids}
    outgoing = {tl: sorted(flatten2(sn(tl))) for tl in tl_ids}
    return incoming, outgoing


def get_neighbors(incoming, outgoing):
    """Reads incoming and outgoing

    Parameters:
    -----------
    * incoming: dict<str,<int, list<str>>
        Phases controlling the incoming approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    * outgoing: dict<str,<int, list<str>>
        Phases controlling the outgoing approaches.
        key: intersection_code <str> --> phase_id <int> --> lane_ids <str>

    Returns:
    --------
    * edge_list: list<tuple<int, int>>
        List of pairs of intersections where (i, j) is an edge iff:
                            i --> j 
        If an outgoing lane of i is an incoming lane of j, or
        i is a source of traffic flow for j, or
        i is upstream to j.

    * label_to_id: dict<str, int>
        A mapping of intersection codes to intersection ids.

    Usage:
    ------
    >>> edge_list, label_to_id = get_neighbors(incoming, outgoing)
    """
    assert incoming.keys() == outgoing.keys()
    # Get labels mapping
    id_to_label = dict(enumerate(sorted(incoming.keys())))
    all_edges = [(i, j) for i in id_to_label for j in id_to_label if i >= j]

    # Get incidence matrix
    edge_list = []
    prev_lu = ""
    for u, v in all_edges:
        lu, lv = id_to_label[u], id_to_label[v]
        if prev_lu != lu:
            uinc_set = set(incoming[lu])
            uout_set = set(outgoing[lu])

        vinc_set = set(incoming[lv])
        vout_set = set(outgoing[lv])

        # u = v
        if u == v: edge_list.append((u, v))
        # u --> v
        if len(uout_set & vinc_set) > 0: edge_list.append((u, v))
        # v --> u
        if len(vout_set & uinc_set) > 0: edge_list.append((v, u))
        prev_lu = lu
    
    return edge_list, id_to_label

# TODO: Change from phases to lanes.
def get_adjacency_from_roadnet(roadnet):
    *args, _ = get_lanes(roadnet)
    return get_adjacency_from_roadlinks(*args)

def get_adjacency_from_env(env):
    args = (env.lanes_incoming, env.lanes_outgoing)
    return get_adjacency_from_roadlinks(*args)

def get_adjacency_from_roadlinks(incoming, outgoing):
    edge_list, _ = get_neighbors(incoming, outgoing)
    data = np.ones(len(edge_list), dtype=int)
    # incidence: (i, j): i --> j
    incidence = csr_matrix((data, zip(*edge_list)), dtype=int).todense()
    # adjacency to be the `reverse` of `incidence`.
    # j --> i
    source = incidence.T
    return source

def get_capacity_from_roadnet(roadnet, flows=None):
    """Capacity from each tl_ids per phase"""

    if flows is None:
        vehlen, vehgap = 5, 2.5
    else:
        # Flows determine the min. length of the vehicles.
        # min. length --> generates the maximum capacity.
        vehlen = min([flow['vehicle']['length'] for flow in flows])
        vehgap = min([flow['vehicle']['minGap'] for flow in flows])

    capacities = {}
    tl_ids = []
    intersections = [intr for intr in roadnet['intersections'] if not intr['virtual']]
    roads = roadnet['roads']
    for intersection in intersections:
        lightphases = intersection['trafficLight']['lightphases']
        p = 0
        tl_id = intersection['id']
        phase_capacities = {}
        for linkids in lightphases:
            if len(linkids['availableRoadLinks']) > 0:
                linkids = linkids['availableRoadLinks']
                capacity = 0
                for linkid in linkids:
                    # startRoad should be the incoming links.
                    edgeid = intersection['roadLinks'][linkid]['startRoad']
                    capacity += sum([
                        points2length(*road['points'])
                        for road in roads if road['id'] == edgeid
                    ])

                phase_capacities[p] = int(capacity / (vehlen + vehgap))
                p += 1
        capacities[tl_id] = phase_capacities
    return capacities 

def get_capacity_from_roadnet2(roadnet, flows=None):
    """Capacity from each tl_ids per lane

    Usage:
    ------
    >>> import json
    >>> with open(roadnet_path, 'r') as f: roadnet = json.load(f)
    >>> incoming, outgoing = get_capacity_from_roadnet2(roadnet)
    >>> incoming
    >>> {'intersection_1_1': 
    >>>        {0: [('road_0_1_0_1', 10), ('road_0_1_0_0', 10), ('road_2_1_2_1', 10), ('road_2_1_2_0', 10)],
    >>>         1: [('road_1_0_1_0', 10), ('road_1_0_1_0', 10), ('road_1_2_3_0', 11), ('road_1_2_3_0', 11)]},
    >>>    'intersection_2_1': ...}
    """

    if flows is None:
        vehlen, vehgap = 5, 2.5
    else:
        # Flows determine the min. length of the vehicles.
        # min. length --> generates the maximum capacity.
        vehlen = min([flow['vehicle']['length'] for flow in flows])
        vehgap = min([flow['vehicle']['minGap'] for flow in flows])

    def cap(pts): return int(points2length(*pts) / (vehlen + vehgap))

    roads = roadnet['roads']
    roads = {road['id']: int(cap(road['points']))  for road in roads}
    incoming, outgoing, _ = get_phases(roadnet)
    for approach in (incoming, outgoing):
        for tl, phases in approach.items(): 
            for ph, lanes in phases.items(): 
                phases[ph] = [
                    (lane, roads[lane[:-2]]) for lane in lanes
                ]
    return incoming, outgoing

if __name__ == '__main__':

    networks = ('arterial', 'grid_6')
    for network in networks:
        with open(f'data/networks/{network}/roadnet.json', 'r') as f: roadnet = json.load(f)
        incoming, outgoing, _ = get_phases(roadnet)
        edge_list, id_to_label = get_neighbors(incoming, outgoing)
        print(f'vvvvvvvvvv   edgeList({network})    vvvvvvvvvv') 
        print(edge_list)

        print(f'vvvvvvvvvv   id_to_label({network})   vvvvvvvvvv') 
        print(id_to_label)

        get_capacity_from_roadnet2(roadnet)

        

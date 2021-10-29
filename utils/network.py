""" Helpful cityflow's roadnet processing functions

    References:
    ----------
    * https://cityflow.readthedocs.io/en/latest/roadnet.html
"""

from utils.utils import flatten

def get_phases(roadnet):
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
    incoming = {}
    outgoing = {}
    speed_limit = {}
    roads = roadnet['roads']
    intersections = [intr for intr in roadnet['intersections'] if not intr['virtual']]
    for intersection in intersections:
        lightphases = intersection['trafficLight']['lightphases']
        roadlinks_incoming = {}
        roadlinks_outgoing = {}
        edges_max_speeds = {}
        p = 0
        for linkids in lightphases:
            if any(linkids['availableRoadLinks']):
                linkids = linkids['availableRoadLinks']
                edges = []
                edges_inverse = []
                for linkid in linkids:
                    # startRoad should be the roadlinks_incoming links.
                    edgeid = intersection['roadLinks'][linkid]['startRoad']
                    lanes = [lane for road in roads if road['id'] == edgeid
                        for lane in road['lanes']]
                    num_lanes = len(lanes)

                    for lid in range(num_lanes):
                        _edgeid = f'{edgeid}_{lid}'
                        if _edgeid not in edges:
                            edges.append(_edgeid)
                            edges_max_speeds[_edgeid] = lanes[lid]['maxSpeed']


                    # endRoad should be the roadlinks_outgoing links.
                    roadlink = intersection['roadLinks'][linkid]
                    edgeid = roadlink['endRoad']
                    edges_inverse += [
                        f"{edgeid}_{rl['endLaneIndex']}"
                        for rl in roadlink['laneLinks']
                    ]
                roadlinks_incoming[p] = edges
                roadlinks_outgoing[p] = list(set(edges_inverse))
                p += 1
        incoming[intersection['id']] = roadlinks_incoming
        outgoing[intersection['id']] = roadlinks_outgoing
        speed_limit[intersection['id']] = edges_max_speeds
    return incoming, outgoing, speed_limit

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
            uinc_set = set(flatten(incoming[lu].values()))
            uout_set = set(flatten(outgoing[lu].values()))

        vinc_set = set(flatten(incoming[lv].values()))
        vout_set = set(flatten(outgoing[lv].values()))

        # u = v
        if u == v: edge_list.append((u, v))
        # u --> v
        if len(uout_set & vinc_set) > 0: edge_list.append((u, v))
        # v --> u
        if len(vout_set & uinc_set) > 0: edge_list.append((v, u))
        prev_lu = lu
    
    return edge_list, id_to_label


if __name__ == '__main__':

    import json
    networks = ('arterial', 'grid_6')
    for network in networks:
        with open(f'data/networks/{network}/roadnet.json', 'r') as f: roadnet = json.load(f)
        incoming, outgoing, _ = get_phases(roadnet)
        edge_list, id_to_label = get_neighbors(incoming, outgoing)
        print(f'vvvvvvvvvv   edgeList({network})    vvvvvvvvvv') 
        print(edge_list)

        print(f'vvvvvvvvvv   id_to_label({network})   vvvvvvvvvv') 
        print(id_to_label)

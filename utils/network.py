""" Helpful cityflow's roadnet processing functions

    References:
    ----------
    * https://cityflow.readthedocs.io/en/latest/roadnet.html
"""

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

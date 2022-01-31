''' Helps converts colight standard to Collaborative Urban Flow's standard. '''

import xml.etree.ElementTree as ET
import argparse
import json
from pathlib import Path
import os

import sys
# append the path of the
# parent directory
sys.path.append(Path.cwd().as_posix())
print(sys.path)
from utils import str2bool

def get_arguments():

    parser = argparse.ArgumentParser(
        description="""Helps converts colight standard to Collaborative Urban Flow"""
    )

    parser.add_argument('network_id', type=str, nargs='?',
                help='''Path to data/networks containing a sumo-like route file. ex: data/networks/grid_2_3''')
    return parser.parse_args()

def clearing(): return {'availableRoadLinks': [], 'time': 6}

def get_generic_element(path, target, file_type='net',
                        ignore=None, key=None, child_key=None):
    """ Parses the {network_id}.{file_type}.xml in search for target

    Usage:
    -----
    > # Returns a list of dicts representing the nodes
    > elements = get_generic_element('grid', 'junctions')
    """
    # Parse xml recover target elements
    file_path = path
    elements = []

    if os.path.isfile(file_path):
        root = ET.parse(file_path).getroot()
        for elem in root.findall(target):
            if ignore not in elem.attrib:
                if key in elem.attrib:
                    elements.append(elem.attrib[key])
                else:
                    elements.append(elem.attrib)

                if child_key is not None:
                    elements[-1][f'{child_key}s'] = \
                        [chlem.attrib for chlem in elem.findall(child_key)]

    return elements


def get_routes(network_id):
    """ Get routes as specified on Network
        routes must contain length and speed (max.)
        but those attributes belong to the lanes.

        Parameters:
        ----------
            * path: string
            string representing a path to sumo-like route file. ex: grid_2_3.rou.xml

        Returns:
        -------
            * routes: list of dictionaries
            as specified at flow.networks.py

        Specs:
        ------
        routes : dict
            A variable whose keys are the starting edge of a specific route, and
            whose values are the list of edges a vehicle is meant to traverse
            starting from that edge. These are only applied at the start of a
            simulation; vehicles are allowed to reroute within the environment
            immediately afterwards.

        References:
        ----------
        flow.network

        Update:
        ------
        2020-05-06: Before routes were equiprobable.
    """
    # Parse xml to recover all generated routes.
    routes = get_generic_element(path, 'vehicle/route',
                                 file_type='rou', key='edges')

    # Unique routes as array of arrays.
    routes = [rou.split(' ') for rou in set(routes)]


def calculate_intervals(routes, vehicles_per_hour)

    def softmax(x, temp=0.20):
        return np.exp(x/temp) / np.sum(np.exp(x/temp))

    # Weight routes.
    weighted_routes = OrderedDict()
    for start, paths in routes.items():

        weights = []

        for path in paths:

            # Criteria: Number of turns belonging to the path.
            counter_turns = 0
            for orig, dest in zip(path, path[1:]):
                if connections[(orig, dest)]['dir'] != 's':
                    counter_turns += 1

            # Path's weight.
            weight = 1 / (counter_turns + 1)
            weights.append(weight)

        t = -0.005 * (len(weights) - 10) + 0.2
        weights = list(softmax(np.array(weights), temp=t))

        weighted_routes[start] = [(p, w) for p, w in zip(paths, weights)]


    return weighted_routes

def filter_routes(routes, roadnet):
    # remove turn_right moviments
    intersections = {inter['id']: inter['roadLinks'] inter for inter in roadnet['intersections']} 
    for rou in routes:
        for edge, next_edge in zip(rou[:-1], rou[1:]):
            tl = next_edge[:-2].sub('road', 'intersection')
            road_links = intersections[tl]
            [rl for rl in road_links if rl['startRoad'] == edge and rl['endRoad'] == next_edge and rl['type'] == 'turn_left']


def main(path):
    # Load roadnet and routes
    roadnet_path = [p for p in path.rglob('roadnet.json')][0]
    with Path(roadnet_path).open('r') as f: roadnet = json.load(f)

    route_path = [p for p in path.rglob('*rou.xml')][0]
    routes = get_routes(route_pah)

    # remove lane-changing routes
    routes = filter_routes(routes, roadnet)

    # compute intervals based on a mean and
    # criteria.
    intervals = calculate_intervals(routes, 150)

    # make a flow dictionary
    flow = make_flow(routes, intervals)
    
    # save -- capture everything
    suffix = '_'.join(route_path.stem.split('_')[1:])
    filename = f'flow_{suffix}.json'
    flow_path = route_path.parent / filename
    with flow_path.open('w') as f: json.dump(d, f)

if __name__ == '__main__':
    opts = get_arguments()
    main(opts.path)



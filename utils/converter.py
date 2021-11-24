''' Helps converts colight standard to Collaborative Urban Flow's standard. '''

import argparse
import json
from pathlib import Path

def clearing(): return {'availableRoadLinks': [], 'time': 6}

def get_arguments():

    parser = argparse.ArgumentParser(
        description="""Helps converts colight standard to Collaborative Urban Flow"""
    )

    parser.add_argument('path', type=str, nargs='?',
                help='Path to the roadnet_X_Y.json file. ex: roadnet_3_3.json')

    return parser.parse_args()

def main(path):
    with Path(path).open('r') as f: roadnet = json.load(f)

    # Creates a clear timing between phases.
    for k, v in roadnet.items():
        if k == 'intersections':
            intersections = []
            for inter in v:
                if not inter['virtual']:
                    light_phases = inter['trafficLight']['lightphases']
                    yellows = len(light_phases) * [clearing()]
                    inter['trafficLight']['lightphases'] = [
                        elem for tup in zip(light_phases, yellows) for elem in tup
                    ]
    new_path = Path(path).parent / 'roadnet.json'

    with new_path.open('w') as f: json.dump(roadnet, f)

if __name__ == '__main__':
    opts = get_arguments()
    main(opts.path)



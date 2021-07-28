"""Value function approximators.

    TODO:
        * Remove loop on roadnet (also used on converters class).
    
    References:
    ---
    https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b
    https://cityflow.readthedocs.io/en/latest/roadnet.html
    https://cityflow.readthedocs.io/en/latest/flow.html
"""
import numpy as np
from utils import points2length

class TileCodingApproximator(object):
    """Thin wrapper around numpy tiling procedures"""
    def __init__(self, roadnet, flows, num_tilings=1, num_tiles=5,
                 min_green=5, yellow=5, max_green=90):
        """Initiatializes a TileCodingApproximator

            Reads roadnet and flows and computes the feature_bounds
            Target towards delay base

            Params:
            -------
            * roadnet: dict
                https://cityflow.readthedocs.io/en/latest/roadnet.html

            * flows: dict
                https://cityflow.readthedocs.io/en/latest/flow.html

            Returns:
            --------
            * tile_coding: object

        """
        # Signal plan constraints.
        self.yellow = yellow
        self.min_green = min_green
        self.max_green = max_green

        # Flows determine the min. length of the vehicles.
        # min. length --> generates the maximum capacity.
        vehlen = min([flow['vehicle']['length'] for flow in flows])
        vehgap = min([flow['vehicle']['minGap'] for flow in flows])
        print(vehlen, vehgap)
        
        # Roadnet determine the capacity.
        self.capacities = {}
        self.tl_ids = []
        self.tilings = {}

        intersections = [intr for intr in roadnet['intersections'] if not intr['virtual']]
        roads = roadnet['roads']
        for intersection in intersections:
            lightphases = intersection['trafficLight']['lightphases']
            p = 0
            tl_id = intersection['id']
            capacities = {}
            for linkids in lightphases:
                if any(linkids['availableRoadLinks']):
                    linkids = linkids['availableRoadLinks']
                    capacity = 0
                    for linkid in linkids:
                        # startRoad should be the incoming links.
                        edgeid = intersection['roadLinks'][linkid]['startRoad']
                        capacity += sum([
                            points2length(*road['points'])
                            for road in roads if road['id'] == edgeid
                        ])

                    capacities[p] = int(capacity / (vehlen + vehgap))
                    p += 1
            self.capacities[tl_id] = capacities
            self.tl_ids.append(tl_id) 
            feature_bounds = [int(cap / 2) for cap in capacities.values()] 
            self.tilings[tl_id] = self.create_tilings(num_tilings, num_tiles, feature_bounds)
            

    def create_tilings(self, num_tilings, num_tiles=5, feature_bounds=[16, 45]):
        """
        """
        if num_tilings > 1: raise ValueError('Must correct me!')
        if num_tilings == 1:
            offsets = [[0] * len(feature_bounds)]

        # TODO: verify num_phases * num_tilings
        bins =[[num_tiles for _ in range(len(feature_bounds))]]
        feature_ranges =[[0, fb] for fb in feature_bounds]
        return create_tilings(feature_ranges, num_tilings, bins, offsets)

    def approximate(self, observations):
        ret = {}
        for tl_id, obs in observations.items():
            state = [obs[0]]
            if (obs[1] <= self.min_green + self.yellow):
                state.append(0)
            elif (obs[1] <= int((self.min_green + self.max_green)) / 3):
                state.append(1)
            elif (obs[1] <= int((self.min_green + self.max_green)) / 2):
                state.append(2)
            else:
                state.append(3)
            tc = get_tile_coding(obs[2:], self.tilings[tl_id]). \
                 reshape(-1).astype(int).tolist()
            state += tc
            ret[tl_id] = tuple([int(sta) for sta in state])
        return ret

def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """
    return np.linspace(feat_range[0], feat_range[1], bins+1)[1:-1] + offset

def create_tilings(feature_ranges, number_tilings, bins, offsets):
        """
        feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
        number_tilings: number of tilings; example: 3 tilings
        bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
        offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
        """
        tilings = []
        # for each tiling
        for tile_i in range(number_tilings):
            tiling_bin = bins[tile_i]
            tiling_offset = offsets[tile_i]

            tiling = []
            # for each feature dimension
            for feat_i in range(len(feature_ranges)):
                feat_range = feature_ranges[feat_i]
                # tiling for 1 feature
                feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
                tiling.append(feat_tiling)
            tilings.append(tiling)
        return np.array(tilings)

def get_tile_coding(feature, tilings):
        """
        feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
        tilings: tilings with a few layers
        return: the encoding for the feature on each layer
        """
        num_dims = len(feature)
        feat_codings = []
        for tiling in tilings:
            feat_coding = []
            for i in range(num_dims):
                feat_i = feature[i]
                tiling_i = tiling[i]  # tiling on that dimension
                coding_i = np.digitize(feat_i, tiling_i)
                feat_coding.append(coding_i)
            feat_codings.append(feat_coding)
        return np.array(feat_codings)

if __name__ == '__main__':
    feat_range = [0, 1.0]
    bins = 10
    offset = 0.2
    tiling_spec = create_tiling(feat_range, bins, offset)
    # array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1])
    print(tiling_spec)

    feature_ranges = [[-1, 1], [2, 5]]  # 2 features
    number_tilings = 3
    bins = [[10, 10], [10, 10], [10, 10]]  # each tiling has a 10*10 grid
    offsets = [[0, 0], [0.2, 1], [0.4, 1.5]]

    tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)

    # # of tilings X features X bins
    # (3, 2, 9)
    print(tilings.shape)

    feature = [0.1, 2.5]

    coding = get_tile_coding(feature, tilings)
    print(coding)
    # array([[5, 1],
    #       [4, 0],
    #       [3, 0]])

    tilings1_5 = create_tilings([[0, 16]], 1, bins=[[5, 5]], offsets=[[0], [0]])
    print(tilings1_5)
    print(get_tile_coding([3], tilings1_5))
    print(get_tile_coding([4], tilings1_5))
    print(get_tile_coding([5], tilings1_5))
    print(get_tile_coding([10], tilings1_5))
    print(get_tile_coding([13], tilings1_5))
    print(get_tile_coding([16], tilings1_5))

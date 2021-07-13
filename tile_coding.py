""" Numpy tile_coding
    References:
    ---
    https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b
"""
import numpy as np
from ipdb import set_trace

class TileCodingMapper(object):
    """Thin wrapper around numpy tiling procedures"""

    def __init__(self, num_phases, num_tilings, num_tiles=5, feature_bounds=[16, 45]):
        """
            num_phases: int
            num_tilings: int
            feature_bounds: list<int>
            TODO: receive phase_capacitites and action to estimate feat_range.
        """
        if len(feature_bounds) != num_phases:
            raise ValueError('len(feature_bounds) == num_phases.')

        if num_tilings > 1: raise ValueError('Must correct me!')
        if num_tilings == 1:
            offsets = [[0] * len(feature_bounds)]

        # num_tilings is the number of output features
        # TODO: verify num_phases * num_tilings
        bins =[[num_tiles for _ in range(num_phases)]]
        feature_ranges =[[0, fb] for fb in feature_bounds]

        self.tilings = create_tilings(feature_ranges, num_tilings, bins, offsets)

    def map(self, state):
        return get_tile_coding(state, self.tilings)

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

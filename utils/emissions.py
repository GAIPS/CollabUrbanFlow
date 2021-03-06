import pandas as pd
import numpy as np

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

def get_vehicles(emissions_df):
    """Returns vehicle data

    On cityflow vehicles are not unique. The same index is
    used by different routes.
    Parameters:
    ----------
    * emissions_df: pandas DataFrame
        SEE get_emission

    Usage:
    -----
    ipdb> vehs_df = get_vehicles(emissions_df)
    ipdb> vehs_df.head()
               route finish  start  wait  total
    id
    flow_00.0  route309265401#0_0   11.3    1.0   0.0   10.3
    flow_00.1  route309265401#0_0   18.4    7.1   0.0   11.3
    flow_00.2  route309265401#0_2   24.0   13.3   0.0   10.7
    flow_00.3  route309265401#0_2   29.7   19.4   0.0   10.3
    flow_00.4  route309265401#0_2   36.1   25.6   0.0   10.5
    """
    emissions_df = emissions_df.reset_index()

    # Builds a dataframe with vehicle starts
    start_df = pd.pivot_table(
        emissions_df,
        index='id', values='time',
        aggfunc=min
    ). \
    rename(columns={'time': 'start'}, inplace=False)
    # reset_index('route'). \

    # Builds a dataframe with vehicle finish
    finish_df = pd.pivot_table(
        emissions_df,
        index='id', values='time',
        aggfunc=max
    ).\
    rename(columns={'time': 'finish'}, inplace=False)
    # reset_index('route'). \

    # Builds a dataframe with waiting times
    emissions_df['isStopped'] = (emissions_df['speed'] <= 0.1).astype(float)
    wait_df = pd.pivot_table(
        emissions_df,
        index='id', values='isStopped',
        aggfunc=sum
    ).\
    rename(columns={'isStopped': 'waiting'}, inplace=False)

    # Builds a dataframe with speeds
    speed_df = pd.pivot_table(
        emissions_df,
        index='id', values='speed',
        aggfunc=np.mean
    ).\
    rename(columns={'time': 'speed'}, inplace=False)
    # reset_index('route'). \

    # Builds a dataframe with the number of stops
    emissions_df['pos_rounded'] = emissions_df.pos.round(decimals=1)

    stops_df = pd.pivot_table(
        emissions_df,
        index=['id', 'lane', 'pos_rounded'], values='speed',
        aggfunc='count'
    ). \
    reset_index('pos_rounded'). \
    reset_index('lane'). \
    reset_index('id'). \
    rename(columns={'speed': 'stops'}, inplace=False)

    stops_df['stops'] = (stops_df['stops'] > 3).astype(float)
    stops_df = pd.pivot_table(
        stops_df.reset_index(),
        index='id', values='stops',
        aggfunc=sum
    )

    vehs_df = start_df.join(
        finish_df, on='id', how='inner',
    ). \
    sort_values('start', inplace=False). \
    join(wait_df, on='id', how='left')
    # Remove trips ending in the last timestep.
    # (because the vehicle may not have reached its destination)
    vehs_df = vehs_df[vehs_df['finish'] < vehs_df['finish'].max()]

    # Calculate travel time.
    vehs_df['total'] = vehs_df['finish'] - vehs_df['start']

    vehs_df = vehs_df.join(
        speed_df, on='id', how='inner',
    )

    vehs_df = vehs_df.join(
        stops_df, on='id', how='inner',
    )

    if 'length' in emissions_df:
        # Assumptions there are no loops
        # Gets unique timestamp for length
        dist_df = pd.pivot_table(
            emissions_df,
            index=('id', 'lane', 'length'),
            values='time',
            aggfunc=np.max
        )
        dist_df = pd.pivot_table(
            dist_df.reset_index(),
            index='id',
            values='length',
            aggfunc=np.sum
        ). \
        rename(columns={'length': 'dist'}, inplace=False)

        vehs_df = vehs_df.join(
            dist_df, on='id', how='inner',
        )
        vehs_df['velocity'] = vehs_df.apply(lambda x: x['dist'] / x['total'], axis=1)

    return vehs_df



def get_throughput(df_emission):

    # depending on the conversion options
    # and net configurations the field
    # might change labels.
    if 'edge_id' in df_emission.columns:
        col_edge = 'edge_id'
    else:
        col_edge = 'lane'

    # SUMO
    # in_junction = df_emission[col_edge].str.startswith(':')
    # CITYFLOW
    in_junction = df_emission[col_edge].str.contains('_TO_')
    df_junction = df_emission[in_junction].sort_values(by=['id', 'time'])

    df_junction = df_junction.drop_duplicates(subset='id', keep='first').reset_index()

    df_junction = df_junction[['time','id']]

    return df_junction

def get_intersections(df_emission):
    """Intersection data"""

    # depending on the conversion options
    # and net configurations the field
    # might change labels.
    if 'edge_id' in df_emission.columns:
        col_edge = 'edge_id'
    else:
        col_edge = 'lane'

    df_intersection = pd.pivot_table(
        df_emission.reset_index(),
        index=['route', col_edge],
        values=['id', 'time'],
        aggfunc=min
    ). \
    sort_values(['route', 'time'], inplace=False)

    return df_intersection




def resample(data, column, freq=6, to_records=True):
    """ Resample dataframe 

    Expect ticks to be 5 to 5 seconds. Aggregate 6
    10 to 10 seconds yielding minute data.

    Params:
    ------
    * data: dict
        info from training or testing aggregated 
        by 5 seconds (decision time).

    * column: str
        a metric from info dict 
        choice = ('actions', 'velocities', 'rewards', 'vehicles')

    * freq: int
        aggregation period (6*10 --> 60)

    * to_records: Boolean
        if records is true returns a list else
        a numpy array.

    Returns:
    -------
    * ret: list or numpy array
     resampled data

    """ 
    ret = []
    for episode in data[column]:
        df = pd.DataFrame.from_dict(episode)
        index = pd.DatetimeIndex(df.index)
        df.index = index

        if column in ('rewards',):
           df = df.resample(f'{freq}n').sum()
        elif column in ('actions', 'velocities', 'vehicles'):
           df = df.resample(f'{freq}n').mean()
        else:
           raise ValueError

        if to_records:
           if column in ('vehicles', 'velocities'):
                ret.append(df.to_dict(orient='list')[0])
           ret.append(df.to_dict(orient='records'))
        else:
           ret.append(np.sum(df.values, axis=1))

    return ret

def episodic_breakdown(data, timesteps, drop=100): 
    """Breaks data down into evenly spaced episodes"""
    ts = max(min(timesteps), drop)
    tf = max(timesteps)
    starts = [i for i, t in enumerate(timesteps) if t == ts]
    finishes = [i for i, t in enumerate(timesteps) if t == tf]
    separators = zip(starts, finishes) 
    for k, v in data.items():
        d = []
        for start, finish in separators:
            d.append(v[start:finish])
        data[k] = d

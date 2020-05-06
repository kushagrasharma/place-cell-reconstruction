import pandas as pd
import numpy as np

TIME_BUCKET_LENGTH = .1
LOCATION_BUCKET_LENGTH = .2 

def bucket_spikes(spike_train, bucket_size=TIME_BUCKET_LENGTH):
    spike_train.loc[len(spike_train)-1] = [float('nan'), pd.Timedelta(0, unit="sec")]
    spike_train = spike_train.sort_values('time').reset_index(drop=True)
    binned = spike_train.groupby(pd.Grouper(key='time', freq='{}S'.format(bucket_size), 
                                            base=spike_train['time'][0].seconds))
    binned = binned['spikes'].apply(lambda x: x.value_counts())
    binned = pd.DataFrame(binned).unstack(fill_value=0)
    binned.columns = binned.columns.droplevel()
    binned = binned.reindex(sorted(binned.columns), axis=1)
    binned.columns = list(range(len(binned.columns)))
    return binned

def x_round(x, round_num):
    if x < 0 or np.isnan(x):
        return x
    return round(x*round_num)/round_num

def bucket_location(locations, bucket_size=TIME_BUCKET_LENGTH, location_bucket_length=LOCATION_BUCKET_LENGTH):
    location_bucket_length = 1.0/location_bucket_length
    locations.loc[len(locations)-1] = ([float('nan')] * (len(locations.columns)-1))+ [pd.Timedelta(0, unit="sec")]
    locations = locations.sort_values('time').reset_index(drop=True)
    locations = locations.groupby(pd.Grouper(key='time', freq='{}S'.format(bucket_size)))
    locations = pd.DataFrame(locations.mean())
    locations = locations.fillna(method='ffill')
    bucketed_locs = locations.apply(lambda x: x.apply(lambda y: x_round(y, location_bucket_length)))
    # remove locations with no measurement, 
    bucketed_locs[0] = bucketed_locs[0].apply(lambda x: np.nan if x == -1 or x == 0 else x)
    locations[0] = locations[0].apply(lambda x: np.nan if x == -1 or x == 0 else x)
    # make min location 0
    bucketed_locs[0] -= bucketed_locs[0].min()
    locations[0] -= locations[0].min()
    return locations, bucketed_locs


import numpy as np
import pandas as pd
from time import time

## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
	
def mem_usage(df):
    return df.memory_usage().sum()/1024**2
	
def read_file(filename):
    t0 = time()
    df = pd.read_csv(filename)
    load_time = time()-t0
    print(f'{filename} read in {load_time:.1f} s')
    # reduce mem usage
    df = reduce_mem_usage(df)
#    print(df.shape)
#    print(df.head())
    return df

def read_train_data():
    df = read_file('train.csv')
    df = convert_timestamp(df)
    df = split_timestamp(df)
    df['log_meter_reading'] = np.log(1+df['meter_reading'])
    print(df.shape)
    print(df.head())
    return df
	
def read_test_data():
    df = read_file('test.csv')
    df = convert_timestamp(df)
    df = split_timestamp(df)
    print(df.shape)
    print(df.head())

    return df

def read_building_data():
    df = read_file('building_metadata.csv')
    df['log_square_feet']= np.log(df['square_feet'])
    print(df.shape)
    print(df.head())
    return df
	
def read_weather_train_data():
    df = read_file('weather_train.csv')
    df = convert_timestamp(df)
    return df
	
def read_full_weather_test_data():
    df = read_file('Weather_test_fulldata.csv')
    df = convert_timestamp(df)
    return df


def read_weather_test_data():
    df = read_file('weather_test.csv')
    df = convert_timestamp(df)
    return df
def split_timestamp(df):
    # dataframe must have a column called timestamp
    df["hour"] = df["timestamp"].dt.hour.astype('int8')
    df["day"] = df["timestamp"].dt.day.astype('int8')
    df["year"] = df["timestamp"].dt.year.astype('int32')
    df["weeknumber"] = df["timestamp"].dt.week.astype('int8')
    df["weekday"] = df["timestamp"].dt.weekday.astype('int8')
    df["month"] = df["timestamp"].dt.month.astype('int8')
    return df

def convert_timestamp(df):
	# converts from string to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
    return df


def null_data_stats(df):
    null_data = df.isnull().sum().sort_values(ascending=False).to_frame()
    null_data.columns = ['total']
    null_data['percentage'] = null_data['total']/len(df)*100
    return null_data
	


def nan_interpolate(inp, verbose=0):
    #interpolates linearly in correspondence of nans
    out = inp
    #find null points
    null_ix = inp.index[inp.isnull()]
    #extract non null series points
    non_null_series = inp[~inp.isnull()]
    # interpolate null points
    intres = np.interp(null_ix, non_null_series.index,non_null_series.values)
    # paste interpolated points into original series
    out.loc[null_ix] = intres
    
    if verbose>0:
        if len(null_ix) == 0:
            gc = []
        else:
            gc = continuity_group(null_ix)
        print('Series contains {:d} nans distributed in {:d} contiguous groups'.format(len(null_ix), len(gc)) )
    
    return out

def continuity_group(ix):
    #initialise first point
    out =  [[ix[0]] ]

    for ixx in ix[1:]:
        if (ixx - out[-1][-1]) ==1:
            out[-1].append(ixx)
        else:
            out.append([ixx])
    return out

def get_lines(filename, nlines=5):
    txt = []
    with open(filename,'r') as file:
        for k in range(nlines):
            txt.append(file.readline())
    return txt
from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

# control functions:
def parser(x):
  return datetime.strptime('190'+x, '%Y-%m')

# form a time series:
def timeseries_to_supervised(data, lag=1):
  df = DataFrame(data)
  columns = [df.shift(i) for i in range(1, lag+1)]
  columns.append(df)
  df = concat(columns, axis=1)
  df.fillna(0, inplace=True)
  return df

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# divide the data:
X = series.values
X = X.reshape(len(X), 1)
# scale it:
scaler = MinMaxScaler(feature_range=(-1,1))
scaler = scaler.fit(X)

scaled_X = scaler.transform(X)
scaled_series = Series(scaled_X[:, 0])
print(scaled_series.head())

inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
print(inverted_series.head())

'''
supervised = timeseries_to_supervised(X, 1)

#create a differenced series:
def difference(dataset, interval=1):
  diff = list()
  for i in range(interval, len(dataset)):
    value = dataset[i] - dataset[i-interval]
    diff.append(value)
  return Series(diff) 
  '''

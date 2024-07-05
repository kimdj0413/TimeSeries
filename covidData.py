import pandas as pd
all = pd.read_csv('./COVIDTimeSeries/covid_19_data.csv')
confirmed = pd.read_csv('./COVIDTimeSeries/time_series_covid19_confirmed_global.csv')
group = all.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum()
group = group.reset_index()
# print(group.head())

# print(confirmed[confirmed['Country/Region']=='Korea, South'])
korea = confirmed[confirmed['Country/Region']=='Korea, South'].iloc[:,4:].T # 행과 열을 바꿔주는 함수 .T
korea.index = pd.to_datetime(korea.index)
# print(korea)

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

rcParams['figure.figsize'] = 12, 8
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)

plt.plot(korea)
plt.show()

daily_case = korea.diff().fillna(korea.iloc[0]).astype('int') # diff = 전 행과의 차(일일 확진자 수)
# print(daily_case)
# plt.plot(daily_case)
# plt.show()

import numpy as np

def create_sequences(data, seq_length):
  xs = []
  ys = []
  for i in range(len(data)-seq_length):
    x = data.iloc[i:(i+seq_length)]
    y = data.iloc[i+seq_length]
    xs.append(x)
    ys.append(y)
  return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(daily_case, seq_length)
# print(X.shape, y.shape)
train_size = int(327*0.8)
# print(train_size)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+33], y[train_size:train_size+33]
X_test, y_test = X[train_size+33:], y[train_size+33:]

# print(X_train.shape, X_val.shape, X_test.shape)
# print(y_train.shape, y_val.shape, y_test.shape)

MIN = X_train.min()
MAX = X_train.max()
# print(MIN, MAX)

def MinMaxScale(array, min, max):
  return (array - min) / (max - min)

X_train = MinMaxScale(X_train, MIN, MAX)
y_train = MinMaxScale(y_train, MIN, MAX)
X_val = MinMaxScale(X_val, MIN, MAX)
y_val = MinMaxScale(y_val, MIN, MAX)
X_test = MinMaxScale(X_test, MIN, MAX)
y_test = MinMaxScale(y_test, MIN, MAX)

import torch

def make_Tensor(array):
  return torch.from_numpy(array).float()

X_train = make_Tensor(X_train)
y_train = make_Tensor(y_train)
X_val = make_Tensor(X_val)
y_val = make_Tensor(y_val)
X_test = make_Tensor(X_test)
y_test = make_Tensor(y_test)
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = fdr.DataReader('005930','2024-01-01','2024-01-16')

sns.set_style('whitegrid')
# df.plot()
# plt.show()
# print(df.head())
df.to_csv('005930_test_val.csv')
print(len(df))
df = pd.read_csv('005930_test_val.csv')
df = df[['Date','Close']]
df.to_csv('005930_test.csv')
print(df)
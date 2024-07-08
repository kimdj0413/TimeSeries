import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = fdr.DataReader('005930','2024-05-21','2024-06-04')
print(df)
sns.set_style('whitegrid')
# df.plot()
# plt.show()
# print(df.head())
df = df[['Close']]
print(df)
df.to_csv('005930_test.csv')
# print(len(df))
# df = pd.read_csv('005930_test_val.csv')
# df = df[['Date','Close']]
# df.to_csv('005930_test.csv')
# print(df)
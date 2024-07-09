import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import holidays

stockNum = '005930'
startDate = '2024-06-01'
days = 12
columns = ['Close']

kHolidays = holidays.KR(years=[2024])
startDate = pd.to_datetime(startDate)

def addBusinessDay(startDate, days, holidayDate):
  currentDate = startDate
  added = 0
  
  while added < days:
    if currentDate.weekday() < 5 and currentDate not in holidayDate:
      added += 1
    currentDate += pd.Timedelta(days=1)
  return currentDate

holidayDate = list(kHolidays.keys())
endDate = addBusinessDay(startDate, days, holidayDate)
endDate = endDate.strftime('%Y-%m-%d')
endDate = '2024-07-02'
df = fdr.DataReader(stockNum,startDate,endDate)
selected = df[columns]
selected.to_csv('005930_test.csv', index=True, encoding='utf-8')

# print(selected)
# sns.set_style('whitegrid')
# df.plot()
# plt.show()
# print(df.head())
# print(len(df))
# df = pd.read_csv('005930_test_val.csv')
# df = df[['Date','Close']]
# df.to_csv('005930_test.csv')
# print(df)
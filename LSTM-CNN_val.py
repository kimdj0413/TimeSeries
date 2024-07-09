import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import holidays
from FinanceData import addBusinessDay

predictDate = '2024-07-03'
predictDate = (pd.to_datetime(predictDate)).strftime('%Y-%m-%d')
cnt = 0
while(True):
  print(cnt)
  df = pd.read_csv('005930_test.csv', encoding='utf-8')
  print(df)
  kHolidays = holidays.KR(years=[2024])
  holidayDate = list(kHolidays.keys())
  def createSequence(data, seqLength):
    xList = []
    yList = []
    if len(data)-seqLength == 0:
      seqLength -= 1
    for i in range(len(data)-seqLength):
      x = data.iloc[i:(i+seqLength)]
      y = data.iloc[i+seqLength]
      xList.append(x)
      yList.append(y)
    print(xList)
    return np.array(xList), np.array(yList)
  def valScale(array, min, max):
    return (array - min) / (max - min)
  def makeTensor(array):
    return torch.from_numpy(array).float()
  class FinancePredict(nn.Module):
    def __init__(self, nFeature, nHidden, seqLen, nLayers):
      super(FinancePredict, self).__init__()
      self.nHidden = nHidden
      self.seqLen = seqLen
      self.nLayers = nLayers
      self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride = 1)
      self.lstm = nn.LSTM(
        input_size = nFeature,
        hidden_size = nHidden,
        num_layers = nLayers
      )
      self.linear = nn.Linear(in_features=nHidden, out_features=1)
      
    def resetHiddenState(self):
      self.hidden = (
        torch.zeros(self.nLayers, self.seqLen, self.nHidden),
        torch.zeros(self.nLayers, self.seqLen, self.nHidden)
      )
    def forward(self, sequences):
      lstmOut, self.hidden = self.lstm(
        sequences.view(len(sequences), self.seqLen, -1),
        self.hidden
      )
      lastTimeStep = lstmOut.view(self.seqLen, len(sequences), self.nHidden)[-1]
      yPred = self.linear(lastTimeStep)
      return yPred
  def trainModel(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100, verbose = 10, patience = 10):
    loss_fn = torch.nn.L1Loss() 
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    trainHist = []
    valHist = []
    for t in range(num_epochs):
        epoch_loss = 0
        for idx, seq in enumerate(train_data): 
            model.resetHiddenState()

            seq = torch.unsqueeze(seq, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        trainHist.append(epoch_loss / len(train_data))

        if val_data is not None:
            with torch.no_grad():
                val_loss = 0
                for val_idx, val_seq in enumerate(val_data):
                    model.resetHiddenState()
                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])

                    val_loss += val_step_loss
                  
            valHist.append(val_loss / len(val_data))

            if t % verbose == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

            if (t % patience == 0) & (t != 0):
                if valHist[t - patience] < valHist[t] :
                    print('\n Early Stopping')
                    break
        elif t % verbose == 0:
            print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')
      
    return model, trainHist, valHist
  model = torch.load('LstmCnnModel.pth')

  df.set_index('Date', inplace=True)
  headDf = df[cnt:11+cnt]
  
  X, y = createSequence(headDf, seqLength=10)
  minVal = X.min()
  maxVal = X.max()

  X = valScale(X, minVal, maxVal)
  y = valScale(y, minVal, maxVal)
  X = makeTensor(X)
  y = makeTensor(y)

  pred_dataset = X
  print(X.shape)
  with torch.no_grad():

    preds = []
    for _ in range(len(pred_dataset)):
      model.resetHiddenState()
      y_test_pred = model(torch.unsqueeze(pred_dataset[_], 0))
      pred = torch.flatten(y_test_pred).item()
      preds.append(pred)

  preds = round(float(np.array(preds)*maxVal))
  lastIndex = pd.to_datetime(headDf[-1:].index[0])
  endDate = lastIndex.strftime('%Y-%m-%d')
  if endDate > predictDate:
    break
  newData = {'Date': [endDate], 'Close': [preds]}
  newDf = pd.DataFrame(newData)

  with open('005930_predict.csv', 'a', newline='', encoding='utf-8') as f:
      newDf.to_csv(f, header=False, index=False)
  cnt += 1
"""
df1 = pd.read_csv('005930_predict.csv')
df2 = pd.read_csv('005930_test.csv')


plt.plot(df1.index, df1.iloc[:, 1], label='Predict')
plt.plot(df2.index, df2['Close'], label='Real')


plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Predict vs Real')
plt.legend()
plt.xticks(rotation=45)

plt.show()
"""
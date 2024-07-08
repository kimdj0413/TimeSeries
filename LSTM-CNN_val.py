import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, optim
from datetime import datetime, timedelta

for i in range(0,30):
  realDf = pd.read_csv('005930_test.csv')
  df = realDf[['Date','Close']]
  df = df.tail(11)
  df.set_index('Date', inplace=True)
  def createSequence(data, seqLength):
    xList = []
    yList = []
    for i in range(len(data)-seqLength):
      x = data.iloc[i:(i+seqLength)]
      y = data.iloc[i+seqLength]
      xList.append(x)
      yList.append(y)
    return np.array(xList), np.array(yList)

  seqLength = 10
  X, y = createSequence(df, seqLength)
  X_predict, y_predict = X, y
  minVal = X_predict.min()
  maxVal = X_predict.max()

  def valScale(array, min, max):
    return (array - min) / (max - min)

  X_predict = valScale(X_predict, minVal, maxVal)
  y_predict = valScale(y_predict, minVal, maxVal)

  def makeTensor(array):
    return torch.from_numpy(array).float()
  X_predict = makeTensor(X_predict)
  y_predict = makeTensor(y_predict)

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
  pred_dataset = X_predict

  with torch.no_grad():
      preds = []
      for _ in range(len(pred_dataset)):
          model.resetHiddenState()
          y_test_pred = model(torch.unsqueeze(pred_dataset[_], 0))
          pred = torch.flatten(y_test_pred).item()
          preds.append(pred)
          
  preds = np.array(preds)*maxVal+5000
  lastDate = df.iloc[-1].name
  newDate = datetime.strptime(lastDate, '%Y-%m-%d')
  newDate = newDate + timedelta(days=1)
  newDate = newDate.strftime('%Y-%m-%d')
  new_row = {'Date': newDate, 'Close': preds}
  realDf = realDf.append(new_row, ignore_index=True)
  # new_row = pd.DataFrame({'Close': preds}, index=[newDate])
  # new_row.index.name = 'Date'
  # realDf = pd.concat([realDf, new_row])
  realDf.to_csv('005930_test.csv')
  print(realDf)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, optim

df = pd.read_csv('005930.csv')
# print(df)
closeDf = df[['Date','Close']]
# print(closeDf.isnull().sum())
closeDf.set_index('Date', inplace=True)

def createSequence(data, seqLength):
  xList = []
  yList = []
  for i in range(len(data)-seqLength):
    x = data.iloc[i:(i+seqLength)]
    y = data.iloc[i+seqLength]
    xList.append(x)
    yList.append(y)
  return np.array(xList), np.array(yList)
print(len(closeDf))
trainSize = int(3452*0.8)

seqLength = 10
X, y = createSequence(closeDf, seqLength)
# print(X.shape, y.shape)
X_train, y_train = X[:trainSize], y[:trainSize]
X_val, y_val = X[trainSize:trainSize+346], y[trainSize:trainSize+346] # 98 = 977*0.2/2+1
X_test, y_test = X[trainSize+346:], y[trainSize+346:]
# print(len(X_test),len(y_test))

# print(X_train.shape, X_val.shape, X_test.shape)
# print(y_train.shape, y_val.shape, y_test.shape)

minVal = X_train.min()
maxVal = X_train.max()

def valScale(array, min, max):
  return (array - min) / (max - min)

X_train = valScale(X_train, minVal, maxVal)
y_train = valScale(y_train, minVal, maxVal)
X_val = valScale(X_val, minVal, maxVal)
y_val = valScale(y_val, minVal, maxVal)
X_test = valScale(X_test, minVal, maxVal)
y_test = valScale(y_test, minVal, maxVal)

def makeTensor(array):
  return torch.from_numpy(array).float()

X_train = makeTensor(X_train)
y_train = makeTensor(y_train)
X_val = makeTensor(X_val)
y_val = makeTensor(y_val)
X_test = makeTensor(X_test)
y_test = makeTensor(y_test)

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
  
model = FinancePredict(
    nFeature=1,
    nHidden=4,
    seqLen=seqLength,
    nLayers=1
)
model, trainHist, valHist = trainModel(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs=100,
    verbose=10,
    patience=50
)
torch.save(model, 'LstmCnnModel.pth')
model = torch.load('LstmCnnModel.pth')
plt.plot(trainHist, label="Training loss")
plt.plot(valHist, label="Val loss")
plt.legend()
plt.show()

pred_dataset = X_test

with torch.no_grad():
    preds = []
    for _ in range(len(pred_dataset)):
        model.resetHiddenState()
        y_test_pred = model(torch.unsqueeze(pred_dataset[_], 0))
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
def MAE(true, pred):
    return np.mean(np.abs(true-pred))
print(MAE(np.array(y_test)*maxVal, np.array(preds)*maxVal))

dates = closeDf.index[-len(y_test):]
tests = np.array(y_test)*maxVal
preds = np.array(preds)*maxVal+5000

df = pd.DataFrame({
  'Date' : dates,
  'Actual' : tests.flatten(),
  'Predicted' : preds
})
df['Predicted'] = df['Predicted'].shift(-1)
df['Predicted'].iloc[-1] = np.nan

x = np.arange(len(df['Date']))  # 날짜를 숫자로 변환

plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, df['Actual'], width=0.4, label='Actual', alpha=0.7)
plt.bar(x + 0.2, df['Predicted'], width=0.4, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.xticks(x, df['Date'], rotation=45)
plt.show()

# plt.plot(closeDf.index[-len(y_test):], np.array(y_test) * maxVal, label='True')
# plt.plot(closeDf.index[-len(preds):], (np.array(preds) * maxVal+5995), label='Pred')
# plt.xticks(rotation=45)
# plt.legend()
# plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
  
class TimesBlock(nn.Module):
  def __init__(self, configs):
    super(TimesBlock, self).__init__()
    self.seq_len = configs.seq_len
    self.pred_len = configs.pred_len
    self.k = configs.top_k
    
    self.conv = nn.Sequential(
      Inception_Block_V1(configs.d_model, configs.d_ff,
                         num_kernels=configs.num_kernels),
      nn.GELU(),
      Inception_Block_V1(configs.d_ff, configs.d_model,
                         num_kernels=configs.num_kernels)
    )
  def forward(self, x):
        B, T, N = x.size()
            # B: 배치 크기, T: 시계열 길이, N: 특성의 수
        period_list, period_weight = FFT_for_Period(x, self.k)
            # FFT_for_Period()는 나중에 설명됩니다. 여기서 period_list([top_k])는 
            # top_k 중요한 주기를 나타내고, period_weight([B, top_k])는 그 주기의 가중치(진폭)를 나타냅니다.
        res = []
        for i in range(self.k):
            period = period_list[i]

            # 패딩: 2D 맵을 형성하기 위해, 예측할 부분을 포함한 전체 시퀀스 길이가 
            # 주기로 나누어 떨어져야 하므로 패딩이 필요합니다.
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape: we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            # Whereafter, to make the tensor contiguous in memory, call contiguous()
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            
            #2D convolution to grap the intra- and inter- period information
            out = self.conv(out)

            # reshape back, similar to reshape
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            #truncating down the padded part of the output and put it to result
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1) #res: 4D [B, length , N, top_k]

        # adaptive aggregation
        #First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim=1) 

        #after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        
        #add by weight the top_k periods' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res

def FFT_for_Period(x, k=2):
    # xf의 형태는 [B, T, C]이며, B는 배치 크기, T는 시간 차원, C는 채널 수를 나타냅니다.
    # 주어진 데이터 조각에 대해 주파수(T)의 진폭을 나타냅니다.
    xf = torch.fft.rfft(x, dim=1) 

    # 진폭에 따라 주기를 찾습니다: 여기서는 주기적 특성이 배치와 채널에 따라 기본적으로 일정하다고 가정하므로
    # 이 두 차원을 평균내어, 주파수 리스트를 얻습니다. 주파수 리스트는 [T] 형태를 가집니다.
    # 주파수 리스트의 각 요소는 해당 주파수(t)의 전체 진폭을 나타냅니다.
    frequency_list = abs(xf).mean(0).mean(-1) 
    frequency_list[0] = 0

    # torch.topk()를 사용하여 frequency_list에서 가장 큰 k 개의 요소와 그 위치를 얻습니다(즉, top_list는 가장 주요한 k 주파수입니다).
    _, top_list = torch.topk(frequency_list, k)

    # 현재 그래프에서 분리된 새로운 텐서를 반환합니다.
    # 결과는 절대로 그래디언트를 요구하지 않습니다. numpy 인스턴스로 변환합니다.
    top_list = top_list.detach().cpu().numpy()
     
    # period: 형태 [top_k]의 리스트로, 각각의 평균 주파수에 대한 주기를 기록합니다.
    period = x.shape[1] // top_list

    # 여기서 두 번째 항목은 [B, top_k] 형태를 가지며,
    # 각 데이터 조각에 대해 가장 큰 top_k 진폭을 나타내며, N 개의 특성이 평균화됩니다.
    return period, abs(xf).mean(-1)[:, top_list] 
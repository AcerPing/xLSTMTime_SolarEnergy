
import torch.nn.functional as F
from torch import nn
import torch
import argparse
import numpy as np
from einops import rearrange

import argparse

from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig


mlstm_config = mLSTMBlockConfig() # mLSTMBlock：matrix 矩陣記憶，支援多特徵並行，適合高維輸入（如影片、語句）
slstm_config = sLSTMBlockConfig() # sLSTMBlock：scalar 標量記憶更新，適合細膩序列（如體重成長）

config = xLSTMBlockStackConfig(
        mlstm_block=mlstm_config,
        slstm_block=slstm_config, # Block組成：sLSTM + mLSTM 搭配，包含多個 mLSTMBlock/sLSTMBlock 的堆疊。
        num_blocks=3, # 3 個 block 疊加
        embedding_dim=128, # 是傳入 xLSTM block 的特徵維度。線性調整維度：target_points → 128 維；或 128 維 → target_points。
        # embedding_dim：經過線性層或處理後，要輸入給 xLSTMBlockStack 的內部維度。
        # 升維的目的，因為 模型的「表現空間」通常比輸入維度高。 embedding_dim：是模型內部抽象表徵的空間（更有表達力）。
        add_post_blocks_norm=True, # 加上 normalization
                                  # 在 xLSTMBlockStack 的每個 block 執行完後，自動加上一層 LayerNorm 或 BatchNorm。
                                  # 作用時間點：在 block stack 之後加上 norm，屬於 內部的 normalization。
        
        _block_map = 1, # 很可能代表使用「交替架構」：sLSTM、mLSTM、sLSTM（或類似順序） # ??

        #slstm_at="all", # 開啟後會錯 RuntimeError: Error building extension 'slstm_HS128BS8NH4NS4DBfDRbDWbDGbDSbDAfNG4SA1GRCV0GRC0d0FCV0FC0d0'
                        # 在 xLSTM block stack 的所有層中都使用 sLSTM block。
                        # 所有層都用 sLSTMBlock（不使用 mLSTMBlock）。
        context_length=1440,
    )
    

class moving_avg(nn.Module): # 1️⃣ 平滑移動平均 → 用於序列趨勢分解
    """
    Moving average block to highlight the trend of time series.
    # 移動平均，用來分解趨勢
    # 對時間序列進行移動平均操作(平滑化), 用來分離出時間序列中的長期趨勢(trend)。
    """
    def __init__(self, kernel_size, stride):
        """
        --  kernel_size: 控制平均的「時間範圍」有多大（即視窗大小）。
        -- stride: 移動的步幅（通常設為 1, 代表逐步平滑）。
        -- padding=0 的意思: 在池化運算時, 不自動對輸入的時間軸進行補零(padding)。
                            （沒有 padding, 那麼池化後的輸出長度會變短。）
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size # 預設 kernel_size = 25
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) # 這是 PyTorch 的一維平均池化，用來對時間序列做平滑。
                                                                                   # 對時間序列（或一維資料）滑動視窗取平均，達到平滑化效果。

    def forward(self, x):
        """
        # padding on the both ends of time series
        讓輸出長度與輸入一致（Same Padding）。因為池化操作會壓縮長度，這邊使用 重複最前面與最後面的時間點 來補值，讓資料不會因為池化而變短。
        假設： 輸入 x 形狀是 (B, T, F)，補值完後，形狀變為 (B, T+padding, F)。
        """
        # 1. 邊界補值 Padding（前後各補一段）
        # 程式作者手動做了padding，用「頭尾重複」補上缺的部份（更合理）。
        # 不依賴AvgPool1d(padding=...)的自動補零，而是自己補「邊界值」來保持時間長度不變（避免資訊流失）。
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) # 補上開頭的值
                                                                       # 1.) 取出「每一筆資料的第一個時間點」：x[:, 0:1, :] → 形狀變成 (B, 1, F)
                                                                       # 2.) 然後重複它 (kernel_size - 1) // 2 次
                                                                       # 3.) 最後的 front.shape = (B, 左側補長, F)
                                                                       # 用「最開頭那一個時間步」的資料 重複(kernel_size - 1) // 2 次來左側填充，沿用開頭值。
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1) # 補上結尾的值
                                                                     # 1.) 取出「每一筆資料的最後一個時間點」：x[:, -1:, :] → (B, 1, F)
                                                                     # 2.) 然後重複它 (kernel_size - 1) // 2 次
                                                                     # 3.) 得到 end.shape = (B, 右側補長, F)
                                                                     # 將序列的最後一個時間步（x[:, -1, :]）複製 (kernel_size - 1) // 2 次，補在後面，目的是對稱 padding，保持維度一致。
        x = torch.cat([front, x, end], dim=1) # 把「前補值 + 原始資料 + 後補值」沿著時間維度（dim=1）串接起來。
                                              # 最終的 x.shape = (B, T + padding_len * 2, F)
                                              # 確保後續的 AvgPool1d 不會讓輸出時間長度縮短。
        # 2. 套用 AvgPool1d 進行移動平均
        x = self.avg(x.permute(0, 2, 1)) # 套用 AvgPool1d 進行移動、提取平均值。
                                         # 因為 PyTorch 的 AvgPool1d 需要的是 (B, C, L) 形狀（即：batch × channel × time），所以需先 permute() 一下。
                                         # (B, F, T) → for AvgPool1d
        x = x.permute(0, 2, 1) # 回復成 (B, T, F)
        return x # 出來的數值為 "平滑趨勢 trend(t)"。
    

class series_decomp2(nn.Module): # 2️⃣ 將序列分解為趨勢與季節性項目（x = trend + seasonal）
    """
    Series decomposition block
    # 將輸入序列分成 seasonal（季節性）與 trend（趨勢）
    """
    def __init__(self, kernel_size):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x) # 趨勢 trend(t)，捕捉資料的長期平滑趨勢。
        res = x - moving_mean # seasonal = 原始 - 趨勢，反映快速波動，如週期性或突發變化。
        return res, moving_mean # 最終這兩部分都會進入 xLSTM 模型進行獨立處理，再合併預測。
                                # 目的：提高模型學習效率與解釋性。


class xlstm(torch.nn.Module):
    """
    # 整體預測模型的封裝（核心部分）
    # xLSTMBlockStack: 內部使用 sLSTM、mLSTM 的堆疊模型。
    """
    def __init__(self, configs, enc_in):
        super(xlstm, self).__init__()

        # 模型參數記錄
        self.configs = configs
        self.enc_in = enc_in # 輸入特徵數量，也就是原始資料的欄位數（features）

        self.batch_norm = nn.BatchNorm1d(self.enc_in) # 建立一層 一維 Batch Normalization 層，並且根據輸入特徵數設置。
                                                      # 對每個特徵維度做批次正規化，幫助模型訓練更快 且 避免梯度爆炸或消失。
        self.layer_norm = nn.LayerNorm(normalized_shape=self.configs.n2)

        # 〔分解模組〕
        # Decompsition Kernel Size （分解卷積核尺寸），將複雜的資料或結構分成較小的部分來處理。
        # 對輸入序列進行 分解（Decomposition），分離出趨勢與週期。
        kernel_size = 25 # 表示卷積核（filter）會一次滑動跨 25 個時間步，是個「長度為 25」的一維濾波器。
        self.decompsition = series_decomp2(kernel_size)
        # 對 trend/seasonal 做線性轉換
        self.Linear_Seasonal = nn.Linear(configs.context_points,configs.target_points) # 對 seasonal 進行線性轉換
        self.Linear_Trend = nn.Linear(configs.context_points,configs.target_points) # 對 trend 進行線性轉換
        self.Linear_Decoder = nn.Linear(configs.context_points,configs.target_points) # 未使用的變數
        
        # 初始預測為「所有輸入時間點的平均值」。每個輸出點都取輸入的平均值，用於趨勢與季節性成分的預測起點。
        # 設定神經網路，在一開始的時候，就像在算平均值，而不是亂猜一通。
        # 對每一個要預測的 target（例如 96 點），都取所有過去的輸入（336 點）進行等權平均，每個時間點的權重都是 1 / 336，也就是「一視同仁」。
        # 1.) 穩定起始預測結果。模型剛訓練開始還沒學會趨勢時，先用平均值當baseline，不會亂預測；且就算還沒訓練，模型初始的預測也會有「基本合理性」。
        # 2.) 提升收斂速度。讓模型初期就有「類似平滑線條」的預測基礎，有助於早期loss快速下降。
        self.Linear_Seasonal.weight = nn.Parameter((1/configs.context_points)*torch.ones([configs.target_points,configs.context_points])) # 將 Linear_Seasonal 的 weight 初始化為平均值權重
        self.Linear_Trend.weight = nn.Parameter((1/configs.context_points)*torch.ones([configs.target_points,configs.context_points])) # 將 Linear_Trend 的 weight 初始化為平均值權重
        # torch.ones([configs.target_points,configs.context_points]) => 建立一個填滿 1 的張量 (tensor)
        # nn.Parameter => 將一個張量標記為模型的可學習參數（會被 model.parameters() 自動加入到訓練中），在訓練時會被的optimizer自動更新。
        # 1/configs.context_points => 「平均權重」的概念，每個時間步都有相同影響力。
        # 廣播乘法 => 讓一個數字「自動擴展」成一個形狀跟你要運算的矩陣一樣的東西，然後進行逐元素運算（像是加法、乘法）。
                # => 結果是Tensor。
    
        # 線性轉換，特徵壓縮與輸出維度調整。
        self.mm= nn.Linear(self.configs.target_points, self.configs.n2) # 將模型的輸出時間點（target_points）經過線性轉換後，升維成 n2 維度。
                                                                        # 目的是讓輸出資料能符合 xLSTMBlockStack 所要求的輸入維度格式。
                                                                        # 預設 target_points = 1, n2 = 256
        
        self.mm2= nn.Linear(config.embedding_dim, configs.target_points) # 線性轉換層，把 輸入維度：embedding_dim（例如 256） 轉換為 target_points。
                                                                         # LSTM Block（xLSTMBlockStack）處理完後，輸出的是 embedding 表示，仍是 256 維的抽象空間。
                                                                         # 為了變回「實際的 target 預測數據」，需要從 embedding_dim 壓回 target_points。
                                                                         # 把 xLSTM block 的輸出從 抽象 embedding 維度（如 256） → 具體預測維度。
        
        # self.mm3= nn.Linear(configs.context_points,self.configs.n2) # * 未實際使用。

        self.head = nn.Linear(self.enc_in, 1)  # > feature-to-target regression head (Features → 1個目標)，讓最後輸出的 shape 變成 (batch_size, target_points, 1)。

        self.xlstm_stack = xLSTMBlockStack(config) # 4️⃣ 核心堆疊：xLSTMBlockStack
                                                   # 根據設定堆疊多層 sLSTMBlock 與 mLSTMBlock，進行時間與特徵關聯建模。

        
    def forward(self, x, verbose=False):
        if verbose: print('\n', '-----'*10, '\n') # --用於研究
        if verbose: print(f'輸入資料形狀: {x.shape} \n') # 輸入資料 x：形狀為 (batch_size, context_points, features) # --用於研究

        seasonal_init, trend_init = self.decompsition(x) # 分解為 seasonal 及 trend，形狀為 (batch_size, context_points, features)
        if verbose: print(f'[decompsition]分解長期趨勢(trend): {trend_init.shape} \n') # --用於研究
        if verbose: print(f'[decompsition]分解短期季節(seasonal): {seasonal_init.shape} \n') # --用於研究
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1) # 形狀為 (batch_size, features, context_points)
        seasonal_output = self.Linear_Seasonal(seasonal_init) # Linear 預測 seasonal ，形狀為 (batch_size, features, target_points)。 
        trend_output = self.Linear_Trend(trend_init) # Linear 預測 trend ，形狀為 (batch_size, features, target_points)。
        if verbose: print(f'〔線性轉換〕長期趨勢(trend): {trend_output.shape} \n') # --用於研究
        if verbose: print(f'〔線性轉換〕短期季節(seasonal)的: {seasonal_output.shape} \n') # --用於研究

        x = seasonal_output + trend_output # trend + seasonal，形狀為 (batch_size, features, target_points)。
        if verbose: print(f'〔合併〕長期趨勢(trend)與短期季節(seasonal): {x.shape} \n') # --用於研究


        x = self.mm(x) # 接入一層 mm() 線性層：壓縮或轉換維度 
                    # self.mm = nn.Linear(target_points, embedding_dim)
                    # 線性轉為 embedding 維度（target_points → embedding_dim）或（embedding_dim → target_points）
                    # linear layer（又叫 dense layer）全連接層，用來進行資料的維度轉換或特徵投影。
        if verbose: print(f'mm線性轉換: {x.shape} \n') # 形狀為 (batch_size, features, embedding_dim) # --用於研究

        
        # -- x = self.batch_norm(x) # 可視為前處理，對每個feature channel做標準化，有可能提升模型穩定性與收斂效果，但需依任務特性測試確認，補強模型訓練初期的穩定性。
        x = self.layer_norm(x)
    
        x = self.xlstm_stack(x) # 傳入 xLSTMBlockStack（核心 block），形狀為(batch_size, features, embedding_dim)
        if verbose: print(f'經過xLSTM特徵提取與時序建模: {x.shape} \n') # --用於研究
        
        # 還原回 target_points
        x = self.mm2(x) # 輸出再經過 mm2()，轉為 target_points 長度，形狀為(batch_size, features, target_points)
        if verbose: print(f'mm2線性轉換: {x.shape} \n') # --用於研究
        x = x.permute(0,2,1) # 形狀為(batch_size, target_points, features)
        if verbose: print(f"x shape before head: {x.shape} \n")  # Check shape before entering head

        x = self.head(x) # Linear: features → 1
        if verbose: print(f'最終輸出形狀: {x.shape} \n') # --用於研究
        if verbose: print('\n', '-----'*10, '\n') # --用於研究

        return x # 最後輸出 x：形狀為 (batch_size, target_points, fish_weight)

'''
features => 輸入資料的「特徵維度」。一筆序列中，每一個時間點（timestep）有幾個欄位或變數。
            例如：氣溫、濕度、水溫、光照量、風速（共 5 個 feature） → 所以 features = 5

--------------------------------------

⛳ 資料流程圖（維度）
Step 1: 輸入
  原始 x → (batch_size, context_points, features) = (batch_size, 1440, 6)

Step 2: Decomposition → Linear 預測 → 相加後變成:
  (batch_size, features, target_points) = (batch_size, 6, 1)

Step 3: mm 線性升維
  (batch_size, 6, 1) → mm → (batch_size, 6, 256)

Step 4: xLSTMBlockStack 處理
  (batch_size, 6, 256) → 經過 block 處理 → (batch_size, 6, 256)

Step 5: mm2 線性降維
  (batch_size, 6, 256) → mm2 → (batch_size, 6, 1)

Step 6: permute 回 output 格式
  最終輸出 shape = (batch_size, 1, 6)

Step 7: head Linear (features → 1) 只留下 fish_weight
  最終輸出 shape = (batch_size, 1, 1)

## mm / mm2 就像是時間維度與語意空間間的橋樑

--------------------------------------

Input: (temperature, turbidity, dissolved_oxg, pH, ammonia, nitrate)
         ↓
  LSTM (特徵抽取)
         ↓
 Linear Head (6 → 1)
         ↓
Output: (預測 fish_weight)

'''
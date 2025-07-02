import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

# 【1】 Dataset_ETT_hour
# 【2】 Dataset_ETT_minute
# 【3】 Dataset_Custom
# 【4】 Dataset_Pred
# 【5】 Dataset_Aquaponics
# 【6】 Dataset_IoT Monitoring Dataset of Water Quality and Tilapia 

# todo 【1】 Dataset_ETT_hour
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# todo 【2】 Dataset_ETT_minute ★
class Dataset_ETT_minute(Dataset):
    """"
    -- root_path: 資料檔所在資料夾路徑。
    -- split: 資料分割類別：'train'、'val'、'test'
    -- size: [seq_len, label_len, pred_len]
    -- features: 'S'表示單變數，'M'表示多變數。
    -- data_path: 檔案名稱。
    -- scale: 是否標準化資料。
    -- timeenc: 時間編碼方式, 0(簡單時間欄位), 1(sin/cos)
    -- freq: 時間頻率，如 t = 每分鐘。
    -- use_time_features: 是否加入時間欄位特徵。
    """
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4 # 4 天
            self.label_len = 24 * 4 # 1 天
            self.pred_len = 24 * 4 # 1 天
        else:
            self.seq_len = size[0] # 模型輸入長度（context），預設336。
            self.label_len = size[1] # 0
            self.pred_len = size[2] # 要預測的未來時間步，預設96。

        # init
        assert split in ['train', 'test', 'val'] # 檢查條件是否成立。如果 split 的值不是 'train'、'test' 或 'val'，就會丟出錯誤。
        type_map = {'train': 0, 'val': 1, 'test': 2} # 切資料區段：train/val/test
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 讀取 ETT 資料（含 'date' 與多個特徵）

        border1s = [0, # train 起點
                    12 * 30 * 24 * 4 - self.seq_len, # val 起點
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len # test 起點
                    ]
        border2s = [12 * 30 * 24 * 4,  # train 結束
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, # val 結束
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4 # test 結束
                    ]
        # 12 個月（假設每月 30 天），每天 24 小時，每小時 4 筆資料（因為資料是 15 分鐘一次） => 這是 一年（12 個月）× 每小時 4 筆 = 一整年的資料長度。
        # 前 12 個月 → 訓練集；接著 4 個月 → 驗證集；再來 4 個月 → 測試集。
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale: # 只以 train 資料進行標準化，確保測試集未洩漏。
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0: # 把 'date' 欄位轉成時間欄位（若 timeenc = 0）
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index): # 每次抓一段資料，每訓練一筆都會呼叫一次。 在 DataLoader 載入每一筆訓練資料時，實際會呼叫的函數。
        s_begin = index
        s_end = s_begin + self.seq_len # 0 + 336 = 336
        r_begin = s_end - self.label_len # 336 - 0 = 336
        r_end = r_begin + self.label_len + self.pred_len # 336 + 0 + 96 = 432

        seq_x = self.data_x[s_begin:s_end] # x：輸入序列 [index : index + 336]
        seq_y = self.data_y[r_begin:r_end] # y：預測目標 [336 : 432]
        seq_x_mark = self.data_stamp[s_begin:s_end] # 輸入時間特徵（如有）
        seq_y_mark = self.data_stamp[r_begin:r_end] # 預測時間特徵（如有）

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self): # 決定總共有幾筆樣本
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# todo 【3】 Dataset_Custom ★★★★★
class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2
                 ): # target='OT' 是 ETT-small 資料集（如 ETTh1.csv、ETTm1.csv）裡面的一個欄位名稱。
        """
        Dataset_Custom 是「針對訓練用」的 Dataset，讀入 CSV 資料，分成 train/val/test，標準化後，取出 (context, label, target) 三段資料，支援加時間特徵。
        -- root_path: 資料的根目錄。
        -- split: 是要用來 train / val / test 的哪一部分。
        -- size: 輸入序列長度 (seq_len)、標籤長度 (label_len)、預測長度 (pred_len)。
        -- features: 特徵模式（單變量 'S' 或多變量 'M' / 'MS'）。
        -- data_path: 資料檔名。
        -- scale: 是否要標準化。
        -- timeenc: 時間特徵編碼方式 (0: 手動拆解年月日, 1: 頻率編碼）。
        -- freq: 時間資料的頻率（如 'h'：小時級資料）。
        -- time_col_name: 時間欄位名稱（預設是 'date'）。
        -- use_time_features: 是否使用時間特徵。
        -- train_split、test_split: 設定訓練集、測試集比例（剩下是驗證集）。 Ex. 70%訓練模型、 20%驗證模型、剩下的 10% Validation。
        """
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0] # 模型輸入長度（context），預設336。
            self.label_len = size[1] # 0
            self.pred_len = size[2] # 要預測的未來時間步，預設96。
        # init
        assert split in ['train', 'test', 'val'] # 檢查條件是否成立。如果 split 的值不是 'train'、'test' 或 'val'，就會丟出錯誤。
        type_map = {'train': 0, 'val': 1, 'test': 2} # 切資料區段：train/val/test
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self): # 讀資料！
        self.scaler = StandardScaler() # 使用 StandardScaler() 建立標準化物件。
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 讀取 .csv 檔案。
        cols = list(df_raw.columns) #  df_raw.columns: [time_col_name, ...(other features), target feature]
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        # 依據比例劃分，train、val、test 用 train_split、test_split 依比例切開。
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 根據 features 參數：'S'：只選 target 欄位（單變量）、'M' 或 'MS'多欄位特徵。
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]] # 只有 target 特徵。

        # 標準化（只對訓練集資料 fit，後續所有資料 transform）。
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # 只取「訓練集」範圍資料
            self.scaler.fit(train_data.values) # 根據訓練集統計量（均值、標準差）做 fit
            data = self.scaler.transform(df_data.values) # 把整個資料（train/val/test）都用同樣的標準轉換
        else:
            data = df_data.values # 不標準化，直接使用原始資料

        # 處理時間欄位（self.data_stamp）
        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0: # 拆成 month/day/weekday/hour。
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1: # 使用內建的 time_features() 轉換。
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2] # 標準化後的特徵資料。
        self.data_y = data[border1:border2] # 標準化後的目標資料。
        self.data_stamp = data_stamp # 時間特徵資料。

    def __getitem__(self, index): # 取出一組訓練資料，每次回傳一個 sample（通常是訓練一個 batch 裡的一個）。
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # 輸入資料（input sequence）
        seq_y = self.data_y[r_begin:r_end] # 預測目標（target sequence）
        seq_x_mark = self.data_stamp[s_begin:s_end] # 對應的時間特徵（如果啟用 use_time_features）
        seq_y_mark = self.data_stamp[r_begin:r_end] # 對應的時間特徵（如果啟用 use_time_features）

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self): # 定義 Dataset 長度，確保切 patch 時，不會超出資料邊界。
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# todo 【4】 Dataset_Pred -> 「只預測未來」的推論，例如只有輸入，不給真實答案。
class Dataset_Pred(Dataset):
    def __init__(self, root_path, split='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# todo 【5】 Dataset_Aquaponics ★★★★★
class Dataset_Aquaponics (Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='MS', data_path='cleaned_IoTPond2.csv',
                 target='fish_weight', scale=True, timeenc=0, freq='T',
                 time_col_name='created_at', use_time_features=False, 
                 train_split=0.2*0.8, test_split=0.8
                 ):
        """
        Dataset_Custom 是「針對訓練用」的 Dataset，讀入 CSV 資料，分成 train/val/test，標準化後，取出 (context, label, target) 三段資料，支援加時間特徵。
        -- root_path: 資料的根目錄。
        -- split: 是要用來 train / val / test 的哪一部分。
        -- size: 輸入序列長度 (seq_len)、標籤長度 (label_len)、預測長度 (pred_len)。
        -- features: 特徵模式（單變量 'S' 或多變量 'M' / 'MS'）。
        -- data_path: 資料檔名。
        -- scale: 是否要標準化。
        -- timeenc: 時間特徵編碼方式 (0: 手動拆解年月日, 1: 頻率編碼）。
        -- freq: 時間資料的頻率（如 'h'：小時級資料）。
        -- time_col_name: 時間欄位名稱（預設是 'date'）。
        -- use_time_features: 是否使用時間特徵。
        -- train_split、test_split: 設定訓練集、測試集比例（剩下是驗證集）。 Ex. 70%訓練模型、 20%驗證模型、剩下的 10% Validation。
        """
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0] # 模型輸入長度（context），預設1440。
        self.label_len = size[1] # 0
        self.pred_len = size[2] # 要預測的未來時間步，預設1。

        # init
        assert split in ['train', 'test', 'val'] # 檢查條件是否成立。如果 split 的值不是 'train'、'test' 或 'val'，就會丟出錯誤。
                                                 #  split 是告訴 Dataset：這次你要給我哪一部分的資料！
        type_map = {'train': 0, 'val': 1, 'test': 2} # 切資料區段：train/val/test
        self.set_type = type_map[split] # 0

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self): # 讀資料！

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 讀取 .csv 檔案。
        cols = list(df_raw.columns) #  df_raw.columns: [time_col_name, ...(other features), target feature]
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        # 依據比例劃分，train、val、test 用 train_split、test_split 依比例切開。
        num_train = int(len(df_raw) * self.train_split) # train資料筆數
        num_test = int(len(df_raw) * self.test_split) # test資料筆數
        num_valid = len(df_raw) - num_train - num_test # valid資料筆數
        print(f"資料總數：{len(df_raw)}")
        print(f"訓練集：{num_train} 筆 ({num_train / len(df_raw):.2%})")
        print(f"驗證集：{num_valid} 筆 ({num_valid / len(df_raw):.2%})")
        print(f"測試集：{num_test} 筆 ({num_test / len(df_raw):.2%})")
        assert self.train_split + self.test_split <= 1.0, "train + test split 總和不能超過 1"

        # 定義每個 split 的「起點」索引
        border1s = [0, 
                    num_train - self.seq_len, 
                    len(df_raw) - num_test - self.seq_len] 
        # 定義每個 split 的「終點」索引
        border2s = [num_train, 
                    num_train + num_valid, 
                    len(df_raw)]
        # 根據 split('train'/'val'/'test')選出起點
        border1 = border1s[self.set_type]
        # 根據 split('train'/'val'/'test')選出終點
        border2 = border2s[self.set_type]

        # 根據 features 參數：'S'：只選 target 欄位（單變量）、'M' 或 'MS'多欄位特徵。
        # // if self.features == 'M':
        #     // cols_data = df_raw.columns[1:]
        #     // df_data = df_raw[cols_data]
        # // elif self.features == 'S':
        #     // df_data = df_raw[[self.target]] # 只有 target 特徵。
        assert self.features == 'MS', f"features 必須是 'MS'，但收到的是 {self.features}"
        # 取 feature columns 與 target column
        feature_cols = [col for col in df_raw.columns if col not in [self.time_col_name, self.target]] # 時間欄位不是feature，所以一併排除； target是要被預測的，也排除。
                                                                                                       # fish_weight 應為未知數，不能當作特徵。
        df_data_x = df_raw[feature_cols]
        df_data_y = df_raw[[self.target]] # target單獨取出，target = 'fish_weight'。

        # 標準化（只對訓練集資料 fit，後續所有資料 transform）。
        assert self.scale, f"scale 必須是 True，但收到的是 {self.scale}" # // if self.scale: # // else: data = df_data.values # 不標準化，直接使用原始資料
        # todo: 初始化 scaler 
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler() 
        # todo: 整份資料直接 fit_transform（train+val+test一起 fit，不區分） -> 讓 feature 和 target 都統一在完整資料範圍內做縮放（不是只依靠train區段）。
        data_x = self.feature_scaler.fit_transform(df_data_x.values)
        data_y = self.target_scaler.fit_transform(df_data_y.values)
        # todo: 根據 split 分成 train/val/test
        self.data_x = data_x[border1:border2]  # 標準化後的特徵資料。
        self.data_y = data_y[border1:border2]  # 標準化後的目標資料。        

        # 將scale後的數值輸出，進行核對。
        # pd.DataFrame(data_x, columns=feature_cols).to_csv("scaled_feature_data.csv", index=False)
        # pd.DataFrame(data_y, columns=[self.target]).to_csv("scaled_target_data.csv", index=False)

        # TODO: 處理時間欄位（self.data_stamp）
        # always convert created_at to datetime (基本動作)
        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        # 根據 use_time_features 決定要不要處理
        if self.use_time_features:
            if self.timeenc == 0: # 拆成 month/day/weekday/hour。
                df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
            elif self.timeenc == 1: # 使用內建的 time_features() 轉換。
                data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            # 不使用時間特徵，data_stamp設空
            data_stamp = None

        self.data_stamp = data_stamp # 時間特徵資料。

    def __getitem__(self, index): # 取出一組訓練資料，每次回傳一個 sample（通常是訓練一個 batch 裡的一個）。
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # 輸入資料（input sequence）
        seq_y = self.data_y[r_begin:r_end] # 預測目標（target sequence）


        if self.use_time_features: 
            seq_x_mark = self.data_stamp[s_begin:s_end] # 對應的時間特徵（如果啟用 use_time_features）
            seq_y_mark = self.data_stamp[r_begin:r_end] # 對應的時間特徵（如果啟用 use_time_features）
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark) # 額外回傳時間特徵（seq_x_mark、seq_y_mark）。
        else: return _torch(seq_x, seq_y)

    def __len__(self): # 定義 Dataset 長度，確保切 patch 時，不會超出資料邊界。
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform_y(self, data):
        """
        還原目標(Fish Weight)的標準化資料
        """
        return self.target_scaler.inverse_transform(data)


    def inverse_transform_x(self, data):
        """
        還原特徵（環境感測器資料，如水溫、pH、溶氧）的標準化資料
        """
        return self.feature_scaler.inverse_transform(data)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# 【6】 Dataset_IoT Monitoring Dataset of Water Quality and Tilapia
class Dataset_Tilapia (Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='MS', data_path='IoT Monitoring Dataset of Water Quality and Tilapia.csv',
                 target='Average Fish Weight (g)', scale=True, timeenc=0, freq='T',
                 time_col_name='Datetime', use_time_features=False, 
                 train_split=0.7, test_split=0.2,
                 ):
        """
        Dataset_Custom 是「針對訓練用」的 Dataset，讀入 CSV 資料，分成 train/val/test，標準化後，取出 (context, label, target) 三段資料，支援加時間特徵。
        -- root_path: 資料的根目錄。
        -- split: 是要用來 train / val / test 的哪一部分。
        -- size: 輸入序列長度 (seq_len)、標籤長度 (label_len)、預測長度 (pred_len)。
        -- features: 特徵模式（單變量 'S' 或多變量 'M' / 'MS'）。
        -- data_path: 資料檔名。
        -- scale: 是否要標準化。
        -- timeenc: 時間特徵編碼方式 (0: 手動拆解年月日, 1: 頻率編碼）。
        -- freq: 時間資料的頻率（如 'h'：小時級資料）。
        -- time_col_name: 時間欄位名稱（預設是 'date'）。
        -- use_time_features: 是否使用時間特徵。
        -- train_split、test_split: 設定訓練集、測試集比例（剩下是驗證集）。 Ex. 70%訓練模型、 20%驗證模型、剩下的 10% Validation。
        """
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0] # 模型輸入長度（context），預設1440。
        self.label_len = size[1] # 0
        self.pred_len = size[2] # 要預測的未來時間步，預設1。

        # init
        assert split in ['train', 'test', 'val'] # 檢查條件是否成立。如果 split 的值不是 'train'、'test' 或 'val'，就會丟出錯誤。
                                                 #  split 是告訴 Dataset：這次你要給我哪一部分的資料！
        type_map = {'train': 0, 'val': 1, 'test': 2} # 切資料區段：train/val/test
        self.set_type = type_map[split] # 0

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        print(self.train_split, self.test_split)
        self.__read_data__()

    def __read_data__(self): # 讀資料！

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 讀取 .csv 檔案。
        cols = list(df_raw.columns) #  df_raw.columns: [time_col_name, ...(other features), target feature]
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        # 依據比例劃分，train、val、test 用 train_split、test_split 依比例切開。
        num_train = int(len(df_raw) * self.train_split) # train資料筆數
        num_test = int(len(df_raw) * self.test_split) # test資料筆數
        num_valid = len(df_raw) - num_train - num_test # valid資料筆數
        # 定義每個 split 的「起點」索引
        border1s = [0, 
                    num_train - self.seq_len, 
                    len(df_raw) - num_test - self.seq_len] 
        # 定義每個 split 的「終點」索引
        border2s = [num_train, 
                    num_train + num_valid, 
                    len(df_raw)]
        # 根據 split('train'/'val'/'test')選出起點
        border1 = border1s[self.set_type]
        # 根據 split('train'/'val'/'test')選出終點
        border2 = border2s[self.set_type]

        assert self.features == 'MS', f"features 必須是 'MS'，但收到的是 {self.features}"
        # 取 feature columns 與 target column
        feature_cols = [col for col in df_raw.columns if col not in [self.time_col_name, self.target]] # 時間欄位不是feature，所以一併排除； target是要被預測的，也排除。
                                                                                                       # fish_weight 應為未知數，不能當作特徵。
        df_data_x = df_raw[feature_cols]
        df_data_y = df_raw[[self.target]] # target單獨取出，target = 'fish_weight'。

        # 標準化（只對訓練集資料 fit，後續所有資料 transform）。
        assert self.scale, f"scale 必須是 True，但收到的是 {self.scale}"
        # todo: 初始化 scaler 
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler() 
        # todo: 整份資料直接 fit_transform（train+val+test一起 fit，不區分） -> 讓 feature 和 target 都統一在完整資料範圍內做縮放（不是只依靠train區段）。
        data_x = self.feature_scaler.fit_transform(df_data_x.values)
        data_y = self.target_scaler.fit_transform(df_data_y.values)
        # todo: 根據 split 分成 train/val/test
        self.data_x = data_x[border1:border2]  # 標準化後的特徵資料。
        self.data_y = data_y[border1:border2]  # 標準化後的目標資料。        

        # 將scale後的數值輸出，進行核對。
        # pd.DataFrame(data_x, columns=feature_cols).to_csv("scaled_feature_data.csv", index=False)
        # pd.DataFrame(data_y, columns=[self.target]).to_csv("scaled_target_data.csv", index=False)

        # TODO: 處理時間欄位（self.data_stamp）
        # always convert created_at to datetime (基本動作)
        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        # 根據 use_time_features 決定要不要處理
        if self.use_time_features:
            if self.timeenc == 0: # 拆成 month/day/weekday/hour。
                df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
            elif self.timeenc == 1: # 使用內建的 time_features() 轉換。
                data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            # 不使用時間特徵，data_stamp設空
            data_stamp = None

        self.data_stamp = data_stamp # 時間特徵資料。

    def __getitem__(self, index): # 取出一組訓練資料，每次回傳一個 sample（通常是訓練一個 batch 裡的一個）。
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # 輸入資料（input sequence）
        seq_y = self.data_y[r_begin:r_end] # 預測目標（target sequence）


        if self.use_time_features: 
            seq_x_mark = self.data_stamp[s_begin:s_end] # 對應的時間特徵（如果啟用 use_time_features）
            seq_y_mark = self.data_stamp[r_begin:r_end] # 對應的時間特徵（如果啟用 use_time_features）
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark) # 額外回傳時間特徵（seq_x_mark、seq_y_mark）。
        else: return _torch(seq_x, seq_y)

    def __len__(self): # 定義 Dataset 長度，確保切 patch 時，不會超出資料邊界。
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform_y(self, data):
        """
        還原目標(Fish Weight)的標準化資料
        """
        return self.target_scaler.inverse_transform(data)

    
    def inverse_transform_x(self, data):
        """
        還原特徵(環境感測器資料,如水溫、pH、溶氧)的標準化資料
        """
        return self.feature_scaler.inverse_transform(data)
    

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


# todo 【7】 Dataset_SolarEnergy ★★★★★
class Dataset_SolarEnergy (Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='MS', data_path='Merged Plant1 Data(UnNormalized).csv',
                 target='DC_POWER', scale=True, timeenc=0, freq='T',
                 time_col_name='DATE_TIME', use_time_features=False, 
                 train_split=0.8, test_split=0.2
                 ):
        """
        Dataset_Custom 是「針對訓練用」的 Dataset, 讀入 CSV 資料，分成 train/val/test, 標準化後，取出 (context, label, target) 三段資料，支援加時間特徵。
        -- root_path: 資料的根目錄。
        -- split: 是要用來 train / val / test 的哪一部分。
        -- size: 輸入序列長度 (seq_len)、標籤長度 (label_len)、預測長度 (pred_len)。
        -- features: 特徵模式（單變量 'S' 或多變量 'M' / 'MS'）。
        -- data_path: 資料檔名。
        -- scale: 是否要標準化。
        -- timeenc: 時間特徵編碼方式 (0: 手動拆解年月日, 1: 頻率編碼）。
        -- freq: 時間資料的頻率（如 'h'：小時級資料）。
        -- time_col_name: 時間欄位名稱（預設是 'date'）。
        -- use_time_features: 是否使用時間特徵。
        -- train_split、test_split: 設定訓練集、測試集比例（剩下是驗證集）。 Ex. 70%訓練模型、 20%驗證模型、剩下的 10% Validation。
        """
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0] # 模型輸入長度（context），預設5。
        self.label_len = size[1] # 0
        self.pred_len = size[2] # 要預測的未來時間步，預設1。

        # init
        assert split in ['train', 'test', 'val'] # 檢查條件是否成立。如果 split 的值不是 'train'、'test' 或 'val'，就會丟出錯誤。
                                                 #  split 是告訴 Dataset：這次你要給我哪一部分的資料！
        type_map = {'train': 0, 'val': 1, 'test': 2} # 切資料區段：train/val/test
        self.set_type = type_map[split] # 0

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features

        # train test ratio
        self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self): # 讀資料！

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 讀取 .csv 檔案。
        cols = list(df_raw.columns) #  df_raw.columns: [time_col_name, ...(other features), target feature]
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        # 依據比例劃分，train、val、test 用 train_split、test_split 依比例切開。
        num_train = int(len(df_raw) * self.train_split) # train資料筆數
        num_test = int(len(df_raw) * self.test_split) # test資料筆數
        num_valid = len(df_raw) - num_train - num_test # valid資料筆數
        print(f"資料總數：{len(df_raw)}")
        print(f"訓練集：{num_train} 筆 ({num_train / len(df_raw):.2%})")
        print(f"驗證集：{num_valid} 筆 ({num_valid / len(df_raw):.2%})")
        print(f"測試集：{num_test} 筆 ({num_test / len(df_raw):.2%})")
        assert self.train_split + self.test_split <= 1.0, "train + test split 總和不能超過 1"

        # 定義每個 split 的「起點」索引
        border1s = [0, 
                    num_train - self.seq_len, 
                    len(df_raw) - num_test - self.seq_len] 
        # 定義每個 split 的「終點」索引
        border2s = [num_train, 
                    num_train + num_valid, 
                    len(df_raw)]
        # 根據 split('train'/'val'/'test')選出起點
        border1 = border1s[self.set_type]
        # 根據 split('train'/'val'/'test')選出終點
        border2 = border2s[self.set_type]

        # 根據 features 參數：'S'：只選 target 欄位（單變量）、'M' 或 'MS'多欄位特徵。
        # // if self.features == 'M':
        #     // cols_data = df_raw.columns[1:]
        #     // df_data = df_raw[cols_data]
        # // elif self.features == 'S':
        #     // df_data = df_raw[[self.target]] # 只有 target 特徵。
        assert self.features == 'MS', f"features 必須是 'MS'，但收到的是 {self.features}"
        # 取 feature columns 與 target column
        feature_cols = [col for col in df_raw.columns 
                        if col not in [self.time_col_name, self.target]] # 時間欄位不是feature，所以一併排除； target是要被預測的，也排除。
                                                                         # DC_POWER 應為未知數，不能當作特徵。

        # 分兩組： 先分 cyclic_cols & scaler_cols → scaler_cols 做 MinMaxScaler → 合併兩部分 → 再切 train/test
        # 1. cyclic_cols → 不進行 scaling，包含'TIME_SIN', 'TIME_COS'。
        # 2. scaler_cols → 特徵，需要進行 scaling。
        cyclic_cols = ['TIME_SIN', 'TIME_COS'] # cyclic_cols 不進入 MinMaxScaler，保留[-1, 1]
        scaler_cols = [col for col in feature_cols if col not in cyclic_cols] # 只有 scaler_cols 丟進 MinMaxScaler
        # ========== 分開處理 ==========
        df_cyclic = df_raw[cyclic_cols] # 先保留 cyclic_cols
        df_scaler = df_raw[scaler_cols] # scaler_cols → 做 MinMaxScaler
        df_data_y = df_raw[[self.target]] # target單獨取出，target = 'DC_POWER'。

        # 標準化（只對訓練集資料 fit，後續所有資料 transform）。
        assert self.scale, f"scale 必須是 True，但收到的是 {self.scale}" # // if self.scale: # // else: data = df_data.values # 不標準化，直接使用原始資料
        
        # todo: 初始化 scaler 
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler() 
        # ========== 做 scaling ==========
        # todo: 整份資料直接 fit_transform（train+val+test一起 fit，不區分） -> 讓 feature 和 target 都統一在完整資料範圍內做縮放（不是只依靠train區段）。
        data_scaler = self.feature_scaler.fit_transform(df_scaler.values) # 只對 scaler_cols 做 fit_transform
        data_y = self.target_scaler.fit_transform(df_data_y.values) # target 做 scaling
        # 把 cyclic_cols 和 scaler_cols 合併回去
        data_x = np.hstack([
            df_cyclic.values,   # 不經 scaler
            data_scaler         # 經過 scaler
        ]) # 最後再用 np.hstack 合併回來

        # todo: 根據 split 分成 train/val/test
        self.data_x = data_x[border1:border2]  # 標準化後的特徵資料。
        self.data_y = data_y[border1:border2]  # 標準化後的目標資料。        

        # 將scale後的數值輸出，進行核對。
        # feature_cols = cyclic_cols + scaler_cols
        # pd.DataFrame(data_x, columns=feature_cols).to_csv("scaled_feature_data.csv", index=False)
        # pd.DataFrame(data_y, columns=[self.target]).to_csv("scaled_target_data.csv", index=False)

        # TODO: 處理時間欄位（self.data_stamp）
        # always convert created_at to datetime (基本動作)
        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        # 根據 use_time_features 決定要不要處理
        if self.use_time_features:
            if self.timeenc == 0: # 拆成 month/day/weekday/hour。
                df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month)
                df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day)
                df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday())
                df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour)
                data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
            elif self.timeenc == 1: # 使用內建的 time_features() 轉換。
                data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            # 不使用時間特徵，data_stamp設空
            data_stamp = None

        self.data_stamp = data_stamp # 時間特徵資料。

    def __getitem__(self, index): # 取出一組訓練資料，每次回傳一個 sample（通常是訓練一個 batch 裡的一個）。
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end] # 輸入資料（input sequence）
        seq_y = self.data_y[r_begin:r_end] # 預測目標（target sequence）


        if self.use_time_features: 
            seq_x_mark = self.data_stamp[s_begin:s_end] # 對應的時間特徵（如果啟用 use_time_features）
            seq_y_mark = self.data_stamp[r_begin:r_end] # 對應的時間特徵（如果啟用 use_time_features）
            return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark) # 額外回傳時間特徵（seq_x_mark、seq_y_mark）。
        else: return _torch(seq_x, seq_y)

    def __len__(self): # 定義 Dataset 長度，確保切 patch 時，不會超出資料邊界。
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform_y(self, data):
        """
        還原目標(DC_POWER)的標準化資料
        """
        return self.target_scaler.inverse_transform(data)


    def inverse_transform_x(self, data):
        """
        還原特徵（環境感測器資料，如水溫、pH、溶氧）的標準化資料
        """
        # 先保留 cyclic_cols
        cyclic_cols = ['TIME_SIN', 'TIME_COS']
        data_cyclic = data[:, :len(cyclic_cols)]
        data_scaler = data[:, len(cyclic_cols):]
        data_scaler_inv = self.feature_scaler.inverse_transform(data_scaler) # 對 scaler_cols 還原
        return np.hstack([data_cyclic, data_scaler_inv])


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------


def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)
import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['SolarEnergy'] # 替換不同資料集。

# 1. ettm1 -> ETT 系列資料（電力需求、負載）
# 2. aquaponics -> 〔養殖〕魚菜共生數據集

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}" # 檢查設定的 params.dset 是否在允許的資料集列表 DSETS 中。
    if not hasattr(params,'use_time_features'): params.use_time_features = True

    if params.dset == 'ettm1': #  判斷目前指定的資料集是否為 'ettm1'
        root_path = 'datasets/ETT-small/' # 資料的資料夾路徑，表示原始的 ETTm1.csv 放在 datasets/ETT-small/ 裡。
        size = [params.context_points, 0, params.target_points] # size 定義輸入輸出長度
                                                                # context_points：輸入的歷史步數，例如過去 336 分鐘。
                                                                # 0：預留（目前沒使用，通常是預測前的空窗）預設不使用，即模型直接根據過去的資料預測未來資料。
                                                                # target_points：模型要預測未來幾點，例如未來 96 點。
                                                                #  xLSTM 這種結構可以直接輸入 context → 預測 target，因此中間 label 可省略。
        dls = DataLoaders( # 建立資料加載器
                datasetCls=Dataset_ETT_minute, # ETT 分鐘級資料專用
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features, # 'M' 表示多變量（multivariate）
                'scale': True, # 是否標準化
                'size': size, # 對應的輸入/輸出長度
                'use_time_features': params.use_time_features # 是否加上時間欄位（例如週期性特徵）
                },
                batch_size=params.batch_size, # 批次大小
                workers=params.num_workers, # 執行緒數
                ) # 給定 Dataset 所需的參數，包括資料檔名、標準化、是否加入時間特徵（如小時、週期）、資料切分長度（size），最後交由 DataLoaders() 包裝成 PyTorch 用的訓練與測試資料迭代器。

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

    elif 'aquaponics' in params.dset:
        root_path = 'datasets/aquaponics/'
        size = [params.context_points, 0, params.target_points] # 參考過去 1440 筆數據來預測下一筆資料。 # * context_points=1440, target_points=1

        # 根據 dset 名稱對應到正確的檔案
        if params.dset == 'aquaponics IoTPond2': data_file = 'cleaned_IoTPond2.csv'
        elif params.dset == 'aquaponics IoTPond3': data_file = 'cleaned_IoTPond3.csv'
        elif params.dset == 'aquaponics IoTPond4': data_file = 'cleaned_IoTPond4.csv'
        elif params.dset == 'aquaponics IoTPond1': data_file = 'cleaned_IoTPond1.csv'
        else: raise ValueError(f"❌ 未知的 aquaponics 資料集名稱: {params.dset}") # 值無效或不符合預期

        dls = DataLoaders(
                datasetCls=Dataset_Aquaponics, 
                dataset_kwargs={
                'root_path': root_path,
                'data_path': data_file,
                'features': params.features, # * MS
                'scale': True,
                'size': size, # * [1440, 0, 1]
                'use_time_features': params.use_time_features # * flase
                },
                batch_size=params.batch_size, # * 656
                workers=params.num_workers,
                )

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
 
    elif 'IoT Monitoring Dataset of Water Quality and Tilapia' in params.dset: #  判斷目前指定的資料集是否為 'ettm1'
        root_path = 'datasets/Water Quality & Tilapia/' # 資料的資料夾路徑，表示原始的 ETTm1.csv 放在 datasets/ETT-small/ 裡。
        size = [params.context_points, 0, params.target_points] # size 定義輸入輸出長度
                                                                # context_points：輸入的歷史步數，例如過去 336 分鐘。
                                                                # 0：預留（目前沒使用，通常是預測前的空窗）預設不使用，即模型直接根據過去的資料預測未來資料。
                                                                # target_points：模型要預測未來幾點，例如未來 96 點。
                                                                #  xLSTM 這種結構可以直接輸入 context → 預測 target，因此中間 label 可省略。
        # 根據 dset 名稱對應到正確的檔案
        if params.dset == 'IoT Monitoring Dataset of Water Quality and Tilapia_Segment01': data_file = 'IoT Monitoring Dataset of Water Quality and Tilapia_Segment01.csv'
        elif params.dset == 'IoT Monitoring Dataset of Water Quality and Tilapia_Segment02': data_file = 'IoT Monitoring Dataset of Water Quality and Tilapia_Segment02.csv'
        elif params.dset == 'IoT Monitoring Dataset of Water Quality and Tilapia_Segment03': data_file = 'IoT Monitoring Dataset of Water Quality and Tilapia_Segment03.csv'
        else: raise ValueError(f"❌ 未知的 aquaponics 資料集名稱: {params.dset}") # 值無效或不符合預期
        
        dls = DataLoaders( # 建立資料加載器
                datasetCls=Dataset_Tilapia, # ETT 分鐘級資料專用
                dataset_kwargs={
                'root_path': root_path,
                'data_path': data_file,
                'features': params.features, # 'M' 表示多變量（multivariate）
                'scale': True, # 是否標準化
                'size': size, # 對應的輸入/輸出長度
                'use_time_features': params.use_time_features # 是否加上時間欄位（例如週期性特徵） # * flase
                },
                batch_size=params.batch_size, # 批次大小
                workers=params.num_workers, # 執行緒數
                ) # 給定 Dataset 所需的參數，包括資料檔名、標準化、是否加入時間特徵（如小時、週期）、資料切分長度（size），最後交由 DataLoaders() 包裝成 PyTorch 用的訓練與測試資料迭代器。

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

    elif 'SolarEnergy' in params.dset:
        root_path = 'datasets/SolarEnergy/'
        size = [params.context_points, 0, params.target_points] # 參考過去5筆數據來預測下一筆資料。 # * context_points=5, target_points=1

        # 根據 dset 名稱對應到正確的檔案
        if params.dset == 'SolarEnergy Plant1': data_file = 'Merged Plant1 Data.csv'
        elif params.dset == 'SolarEnergy Plant2': data_file = 'Merged Plant2 Data.csv'
        else: raise ValueError(f"❌ 未知的 aquaponics 資料集名稱: {params.dset}") # 值無效或不符合預期

        dls = DataLoaders(
                datasetCls=Dataset_SolarEnergy, 
                dataset_kwargs={
                'root_path': root_path,
                'data_path': data_file,
                'features': params.features, # * MS
                'scale': True,
                'size': size, # * [5, 0, 1]
                'use_time_features': params.use_time_features # * flase
                },
                batch_size=params.batch_size, # * 128
                workers=params.num_workers,
                )

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points # dls.vars → 特徵數量（features）
                                                                                # dls.len → 輸入長度（context）
    dls.c = dls.train.dataset[0][1].shape[0] # 預測值的特徵數（通常與 vars 相同）
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'aquaponics' # params.dset
        context_points= 5
        target_points= 1
        batch_size= 128
        num_workers= 2
        features='MS'
    params = Params 
    dls = get_dls(params)

    print("\n===== DataLoader 特徵列表 =====")
    if hasattr(dls, 'vars'):  # 有些時候 vars 可能不存在，要防呆
        print(f"特徵欄位數量 (vars): {dls.vars}") # EX. 特徵欄位數量 (vars): 6
    else:
        print("找不到 dls.vars，請檢查 DataLoader 設定")
    
    # 測試看看一個 batch
    print("\n===== Batch 資料內容 =====")
    for i, batch in enumerate(dls.valid): # # 測試 valid 資料，用 enumerate 遍歷 valid dataloader。
       print(f"\n第 {i} 個 batch")
       print(f"  - batch 包含 {len(batch)} 個元素")
       print(f"  - seq_x 形狀: {batch[0].shape}")
       print(f"  - seq_y 形狀: {batch[1].shape}")
       
       if len(batch) == 4:
        print(f"  - seq_x_mark 形狀: {batch[2].shape}")
        print(f"  - seq_y_mark 形狀: {batch[3].shape}")
        # print(i, len(batch), batch[0].shape, batch[1].shape) # 印出 batch 編號、batch 內元素個數、seq_x 和 seq_y 的 shape
        # breakpoint() # breakpoint() => 用來進入除錯模式（debugging），讓你能在程式執行過程中停下來，查看變數內容、單步執行、觀察行為。


"""
① 判斷資料集類型，讀取資料集。
② 切分資料
③ 做前處理
④ 建立 DataLoader
"""
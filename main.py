import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass

from src.learner import Learner
from src.learner import transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.plots import save_arguments, save_lr_curve_from_csv, plot_feature_actual_vs_predicted, plot_error_histogram, plot_residuals, save_yy_plot, save_metrics
from datautils import get_dls # Get Data loaders：原始作者定義的載入資料模組。根據提供的參數，載入並處理對應的資料集，最後輸出 PyTorch 標準格式的 DataLoaders（訓練與測試用）。

from packaging import version

import torch
from torch import nn # import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

assert torch.__version__ >= '1.8.0', "DDP-based MoE requires Pytorch >= 1.8.0"

if version.parse(torch.__version__) >= version.parse("2.1.0"):
    import torch.utils.cpp_extension # Monkey Patch to fix include_paths(cuda=True) for torch >= 2.1.0
    # 取得原始 include_paths()，避免遞迴呼叫自己
    _original_include_paths = torch.utils.cpp_extension.include_paths # 備份原本的 include_paths() 函式
    def include_paths_patched(*args, **kwargs): # 攔截呼叫，把多餘的 cuda 參數去掉
        if 'cuda' in kwargs:
            del kwargs['cuda']  # 移除不支援的參數
        return _original_include_paths(*args, **kwargs)
    torch.utils.cpp_extension.include_paths = include_paths_patched # 進用新的版本取代原來的函式（Monkey Patch）
    # print("✅ Patched torch.utils.cpp_extension.include_paths successfully!")
from model import xlstm
from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig

from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor

import argparse

parser = argparse.ArgumentParser() # 解析命令列參數（Command-line arguments）、控制所有訓練與測試參數。

# TODO 【1】確實有用到的重要核心參數（✅代表有用到。）
# training
parser.add_argument('--is_train', type=int, default=0, help='training the model') # ✅ 控制是否訓練或測試。( 1: train, 0: test ) 
parser.add_argument('--context_points', type=int, default=5, help='sequence length') # ✅ 輸入序列長度
parser.add_argument('--target_points', type=int, default=1, help='forecast horizon') # ✅ 預測序列長度、預測步數。 # * 1
parser.add_argument('--batch_size', type=int, default=128, help='batch size') # ✅ DataLoader批次大小，在 get_dls() 會用到。  # -- 64, 128, 656
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # ✅ 學習率
                                                                            # -- 'FishAquaponics_IoTpond2': 1e-5；'FishAquaponics_IoTpond3': 1e-4；'FishAquaponics_IoTpond4': 1e-4

parser.add_argument('--dset', type=str, default='SolarEnergy Plant1', help='dataset name') # ✅ 資料集名稱（如 ettm1）                                                                                             
parser.add_argument('--model_name', type=str, default='xLSTMTime', help='model_name') # ✅ 模型命名，在 args.save_model_name 會用到。 

parser.add_argument('--model_id', type=int, default=1, help='id of the saved model') # ✅ 模型版本號（便於存檔），在 args.save_model_name 會用到。
# Optimization args
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs') # ✅ 訓練總迭代次數，在 learn.fit_one_cycle 會用到。
parser.add_argument('--n2', type=int, default=128, help='Second Embedded representation') # ✅ 要傳入 xLSTMBlockStack 的嵌入維度（可理解為 embedding_dim），用在 model.py。 

parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not') # ✅ 是否加入時間欄位特徵，用在 datautils.py。 # * 0, False
parser.add_argument('--features', type=str, default='MS', help='for multivariate model or univariate model') # ✅ 特徵類型（M: multivariate 多變量、 S: Single單變量），用在 datautils.py。
                                                                                                            # 單變量（S）=> 每筆資料只有一種特徵（只有一個欄位要預測）
                                                                                                            # 多變量（M）=> 每筆資料有多種特徵（同時觀察/預測多個欄位）
                                                                                                            # * NOTE MS -> 多變量預測單變量（multi→single）。
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader') # ✅ DataLoader 多執行緒設定，用在 datautils.py。
parser.add_argument('--out_dir', type=str, default='results', help='path for output directory') # ✅ 指定輸出目錄的路徑，預設值為 results。
parser.add_argument('--train_mode', type=str, default='pre-train', help="transfer-learning (default : pre-train)") # ✅ 設定模式（在訓練時辨別是否為transfer-learning）
parser.add_argument('--pretrain_path', type=str, default='pre_model_path(.pth)', help="使用遷移學習來訓練模型，從預訓練模型中提取權重並應用於新數據集。") # ✅ 設定模式（在訓練時辨別是否為transfer-learning）
parser.add_argument('--Freeze', action='store_true', help="Freeze transferred weights (default: unfreeze)") # ✅ 在遷移學習中凍結已轉移的權重。
parser.add_argument('--EarlyStoppingPatient', type=int, default='25', help="Early Stopping") # ✅ 設定Early Stopping Patinet

# TODO 【2】定義了但目前未被使用的參數（可能是保留、兼容或暫未實作）（❌代表未用到。）
# 模型初始化
# -- parser.add_argument('--n1', type=int, default=128, help='First Embedded representation')  #256 # 原意應為第一層 embedding，未使用。 ❌

# TODO 【3】取決於是否啟用某些功能的參數
parser.add_argument('--revin', type=int, default=0, help='reversible instance normalization') # 關閉 RevIN（可逆標準化）。 # cbs = [RevInCB(dls.vars)] if args.revin else []
                                                                                              # RevIN（Reversible Instance Normalization）可逆標準化技術 => 讓模型在統一的數值世界裡學習，學完再把預測翻譯回原本的語言。
                                                                                              # * 這是原始模型中，對一切特徵做「多變量輸入」做的Z-score標準化。
parser.add_argument('--scaler', type=str, default='minmax', help='scale the input data') # 特徵標準化方法。
# Patch 時間補丁設定（用在 PatchCB），用在部分 callback 或未啟用。
# patch補丁：把一整段長時間序列，切成一小段一小段的區塊（時間片段）來處理。
# -- parser.add_argument('--patch_len', type=int, default=12, help='patch length') # 每段看多長。 目前未啟用 PatchCB。❌
# -- parser.add_argument('--stride', type=int, default=12, help='stride between patch') # 每次滑動多少秒 目前未啟用 PatchCB。 ❌
# -- parser.add_argument('--pct_start', type=float, default=0.2, help='有多少比例的n_epochs用於「學習率從初始值提升到最高值」') # ✅ 訓練總迭代次數，在 learn.fit_one_cycle 會用到。

args = parser.parse_args()
print('args:', args, '\n')

# 設定儲存模型的名稱與路徑
args.save_model_name = f"{args.model_name}_cw{args.context_points}_tw{args.target_points}_epochs{args.n_epochs}_model{args.model_id}" # 模型名稱
args.save_path = os.path.join(args.out_dir, args.dset)  # 儲存模型位置路徑
if not os.path.exists(args.save_path): os.makedirs(args.save_path) # 建立資料夾

configs = args # 全域變數（global variable），只要在 get_model() 裡沒有重新定義名為 configs 的區域變數，Python 就會使用外層的 全域變數 configs。


def get_model(c_in, args):
    """
    產出模型結構
    -- c_in: number of input variables （輸入特徵數，也就是 features 數量。 enc_in = c_in = features）
    -- 若想導入其他模型（如 LSTM、Transformer），可以直接改寫這裡。
    """

    # * patch補丁，但實際上沒被使用！
    # 計算資料在經過 Patch 分段處理後，會被切成多少個時間片段（patches）。
    # 1.) 加入局部時間資訊（例如：用一小段資料判斷未來趨勢）。
    # 2.) 讓模型能觀察多個區間（patches）而非整體序列。
    # 3.) 模型更容易聚焦局部資訊（短期趨勢）。
    # 4.) 降低記憶體需求。
    # 5.) 可以重疊（用 stride 控制），保留更多上下文。
    # -- num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1 # 滑動視窗切patch，總共可以切幾段？
    # max(args.context_points, args.patch_len)：保證序列長度至少不小於 patch 長度（安全設計）。
    # max(...) - patch_len：可滑動的「剩餘距離」。
    # 除以stride：每次滑 stride 那麼遠，能滑幾次？
    # +1：加上第一次切（從 0 開始）
    # -- print('number of patches:', num_patch) # get number of patches  EX. 模型會把一筆長為 336 的序列切成 28 段，每段 12 個時間點。

    # todo: get model
    model = xlstm(configs, enc_in=c_in) # 把特徵數交給模型
                                        # xlstm() 是主模型結構，搭配 xLSTMBlockStack。
    return model


def combined_loss(input, target, alpha=0.5):
    """
    A combined loss function that computes a weighted sum of MSELoss and L1Loss.
    `alpha` is the weight for MSELoss and (1-alpha) is the weight for L1Loss.
    """
    mse_loss = torch.nn.MSELoss(reduction='mean')
    l1_loss = torch.nn.L1Loss(reduction='mean')
    return alpha * mse_loss(input, target) + (1 - alpha) * l1_loss(input, target)


def find_lr():
    """
    自動尋找一個適當的學習率（learning rate）
    1.) 模型會用 不同的學習率 進行一小段訓練（通常只跑一次 epoch 或更少）。
    2.) 這些學習率會以指數方式增加（如從 1e-7 → 1e-1）。
    3.) 對每個學習率，記錄loss的變化，最終會畫出一張 learning rate vs. loss 的圖
    """
    # get dataloader
    dls = get_dls(args) # 載入訓練資料。
    model = get_model(dls.vars, args) # 建立模型。此時模型裡面各層權重（weights/biases）都還是 隨機初始化（random init），尚未有任何預訓練知識。

    # 若為 transfer-learning，則載入預訓練權重並回傳模型
    if args.train_mode == 'transfer-learning': # TODO: 訓練遷移學習
        # 遷移預訓練權重，將一個訓練好的模型權重（.pth 檔）轉移到另一個模型。
        pretrain_path = args.pretrain_path # 預訓練模型檔案位置
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"❌ 找不到預訓練模型權重檔案: {pretrain_path}")
        print('transfer weights from pre-trained model (遷移學習：載入預訓練模型權重)')
        print(f'載入預訓練模型權重: {pretrain_path}')
        model = transfer_weights(pretrain_path, model, exclude_head=True, device='cuda')  # 若使用 GPU 則改為 'cuda'

    # get loss -> 做小型的「訓練」，計算 loss 曲線。
    # Ex. loss_func = combined_loss
    # -- loss_func = torch.nn.L1Loss(reduction='mean') # MAE（Mean Absolute Error）。
    loss_func = torch.nn.MSELoss(reduction='mean') # MSE（Mean Square Error）。
    print(f'find_lr.loss_func: {loss_func}')
    
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else [] # 使用 callback 記錄訓練過程 
                                                    # args.revin => 判斷是否啟用 RevIN 功能。
                                                    # RevInCB(dls.vars) => 建立一個 RevIN Callback 實例，傳入變數資訊。
                                                    # 在 輸入前 對資料做 normalization，在 模型輸出後 還原（denormalize）預測值。
    #cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)] # Patch-based 時間序列切片

    # define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)  # 使用 Learner() 包裝模型、資料集與 loss function。
    # fit the data to the model
    return learn.lr_finder() # 呼叫 .lr_finder() 來跑這個學習率掃描流程，不用手動設定學習率。


def train_func(lr=args.lr):
    """
    進行訓練、儲存模型權重
    """
    # get dataloader
    dls = get_dls(args) # 載入訓練模型的資料。
    #print('in out', dls.vars, dls.c, dls.len)

    # get model
    model = get_model(dls.vars, args)
    #model = get_model(dls.vars, args, model_type)

    # 若為 transfer-learning，則載入預訓練權重並回傳模型
    if args.train_mode == 'transfer-learning': # TODO: 訓練遷移學習
        # 遷移預訓練權重，將一個訓練好的模型權重（.pth 檔）轉移到另一個模型。
        pretrain_path = args.pretrain_path # 預訓練模型檔案位置
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"❌ 找不到預訓練模型權重檔案: {pretrain_path}")
        print('transfer weights from pre-trained model (遷移學習：載入預訓練模型權重)')
        print(f'載入預訓練模型權重: {pretrain_path}')
        model = transfer_weights(pretrain_path, model, exclude_head=True, device='cuda')  # 若使用 GPU 則改為 'cuda'

    # get loss -> 訓練過程中，模型要計算 loss 來更新權重（backpropagation），必須知道怎麼計算 loss！
    # Ex. loss_func = combined_loss 或 loss_func = HuberLoss(delta = 0.25)
    # -- loss_func = torch.nn.L1Loss(reduction='mean') # MAE（Mean Absolute Error）。
    loss_func = torch.nn.MSELoss(reduction='mean') # MSE（Mean Square Error）。
    print(f'train_func.loss_func: {loss_func}')

    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else [] # 使用 callback 記錄訓練過程 
                                                    # args.revin => 判斷是否啟用 RevIN 功能。
                                                    # RevInCB(dls.vars) => 建立一個 RevIN Callback 實例，傳入變數資訊。
                                                    # 在 輸入前 對資料做 normalization，在 模型輸出後 還原（denormalize）預測值。
    cbs += [
        #PatchCB(patch_len=args.patch_len, stride=args.stride), # Patch-based 時間序列切片
        SaveModelCB(monitor='valid_loss', min_delta=0.000001, fname=args.save_model_name, path=args.save_path), # 建立保存模型的callback，在訓練過程中自動儲存最佳模型權重檔（.pth）。
                                                                                           # 監控「驗證集損失」的表現（valid_loss）。如果新的驗證損失比先前更好，就保存模型。
                                                                                           # 設定 儲存的檔名 與 儲存的資料夾路徑。
                                                                                           # * 設 min_delta=0.001 或 0.002 會讓模型儲存更謹慎，僅在有意義的進步時更新最佳檔案。
        CSVLogger(save_dir=args.save_path, filename='epoch_log.csv'),  # 將訓練過程中每一個epoch的損失與評估指標儲存為 .csv 檔
        EarlyStoppingCB(monitor='valid_loss', min_delta=0.000001, patient=args.EarlyStoppingPatient) # 早停法，避免過擬合。
                                                                                # 由於有使用fit_one_cycle，因此 patient 要設大一點，例如 patient = 10~20。
    ]

    # define learner
    learn = Learner(dls, model, loss_func,
                    lr=lr,
                    cbs=cbs,
                    metrics=[mse, rmse, mae, r2_score, EVS_score]
                    )
    
    if args.train_mode == 'transfer-learning': # TODO: 訓練遷移學習
        if args.Freeze:
            print(f'在遷移學習中，是否凍結權重: {args.Freeze}，即凍結權重。')
            learn.freeze() # 先凍結 backbone，只訓練 head
        else:
            print(f'在遷移學習中，是否凍結權重: {args.Freeze}，即解凍權重。')
            learn.unfreeze() # 解凍模型的所有參數

    # fit the data to the model
    # learn.fine_tune(n_epochs=args.n_epochs, base_lr=lr, freeze_epochs=3, pct_start=0.2) # fine_tune
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2) # 使用 fit_one_cycle 進行訓練，先提高學習率再慢慢降低，先升高 → 達到高峰 → 再慢慢下降。


def test_func():
    """
    測試與視覺化（圖形化預測效果）
    """
    weight_path = args.save_path + '/' + args.save_model_name + '.pth' # 載入權重 .pth
    if not os.path.exists(weight_path): # 確認權重檔案是否存在
        raise FileNotFoundError(f"❌ 找不到模型權重檔案: {weight_path}！ \n")
    print(f"✅ 載入模型權重檔案: {weight_path}")

    # get dataloader
    dls = get_dls(args) # 載入測試集資料。
    model = get_model(dls.vars, args) # 第一階段：初始化模型架構、搭建出模型結構。
    # model = torch.load(weight_path) # 需要自己負責載入權重與設為推論模式，還要確保device匹配。

    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    # cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)] # Patch-based 時間序列切片

    learn = Learner(dls, model, cbs=cbs) # 第二階段：建立 Learner 實例。
                                         # 測試時，只要載入訓練好的權重，做 forward 預測即可，不需要做 loss.backward() 或梯度更新，所以可以不指定 loss function。
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse,  rmse, mae, r2_score, EVS_score])  # 第三階段：載入 .pth 權重
                                                                            # out: a list of [pred, targ, score_values]
                                                                            # preds: 模型預測出來的值。
                                                                            # targs = target（也就是 "ground truth"），測試資料中的「實際答案」。
                                                                            # scores: 評估結果，例如 MSE 和 MAE。
                                                                            
    return out, learn.model
    # dls.test.dataset # 〔備用〕如果需要還原實際值


if __name__ == '__main__':

    if args.is_train: # 執行訓練流程
        print(f'訓練模式: {args.train_mode}')
        suggested_lr = find_lr() # 自動尋找最適學習率
        args.suggested_lr = suggested_lr  # 將學習率加進參數紀錄
        print(f"建議學習率: {suggested_lr:.2e}")
        save_arguments(args.save_path, configs) # 儲存訓練參數。
        train_func(suggested_lr) # 執行訓練
        save_lr_curve_from_csv(os.path.join(args.save_path, 'epoch_log.csv'), args.save_path, f_name=f'{args.dset} Learning Curve') # 繪製 Learning Curve

    else:
        # testing mode 執行測試與可視化
        # 1.) 呼叫 test_func()，得到 out = [pred, targ, score_values] & learn_model。
        # 2.) 針對每個 feature_idx，使用 plot_feature_actual_vs_predicted() 畫出 pred vs targ 曲線。
        out, learn_model = test_func() # out: a list of [pred, targ, score_values] & learn_model

        metric_names = ['MSE', 'RMSE', 'MAE', 'R2 Score', 'Explained Variance Score']
        print('score:', out[2]) # MSE和MAE的評估結果。
        metrics_df = pd.DataFrame({'Metric': metric_names, 'Value': out[2]})
        print(metrics_df.to_string(index=False))
        save_metrics(actual=out[1], predicted=out[0], out_dir=args.save_path, model = learn_model)

        print('pred.shape:', out[0].shape) # 模型預測出來的值。
        print('targ.shape:', out[1].shape) # targ = target（也就是 "ground truth"），測試資料中的「實際答案」。

        # 〔備用〕如果需要還原 pred 和 targ（注意要用 target_scaler）。dataset從test_func()取得。
        # pred = dataset.target_scaler.inverse_transform(out[0].reshape(-1, 1)).reshape(out[0].shape) # 模型預測出來的值。
        # targ = dataset.target_scaler.inverse_transform(out[1].reshape(-1, 1)).reshape(out[1].shape) # 正確答案（真實的 fish_weight）

        # 預測只要針對 fish_weight 進行畫圖
        plot_feature_actual_vs_predicted(actual=out[1], predicted=out[0], out_dir=args.save_path)
        plot_error_histogram(actual=out[1], predicted=out[0], out_dir=args.save_path)
        plot_residuals(actual=out[1], predicted=out[0], out_dir=args.save_path)
        save_yy_plot(actual=out[1], predicted=out[0], out_dir=args.save_path)

    print('----------- Complete! -----------')

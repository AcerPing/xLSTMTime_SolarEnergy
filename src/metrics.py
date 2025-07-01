
import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics import r2_score as sk_r2_score, mean_absolute_percentage_error, explained_variance_score

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    #   看是 torch.Tensor 還是 np.ndarray 以及 看是不是在 GPU（cuda）或 CPU
    # print(f"y_true type: {type(y_true)}, y_true shape: {y_true.shape}, device: {getattr(y_true, 'device', 'N/A')}")
    # print(f"y_pred type: {type(y_pred)}, y_pred shape: {y_pred.shape}, device: {getattr(y_pred, 'device', 'N/A')}")
    #   確保轉到 CPU 並為 numpy array，並且展平成一維。
    if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy().reshape(-1)
    if hasattr(y_pred, 'cpu'): y_pred = y_pred.cpu().numpy().reshape(-1)
    score = sk_r2_score(y_true, y_pred) # float
    return torch.tensor(score) # 將 float 包成 torch.Tensor

def EVS_score(y_true, y_pred):
    """
    當 test 落在平台期時，R² 不可靠，即使模型預測非常準，R² 也可能低或為負。這不是模型爛，而是 R² 的設計不適用於這種情況。
    """
    if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy().reshape(-1)
    if hasattr(y_pred, 'cpu'): y_pred = y_pred.cpu().numpy().reshape(-1)
    score = explained_variance_score(y_true, y_pred) # float
    return torch.tensor(score) # 將 float 包成 torch.Tensor
    
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

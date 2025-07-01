
import torch
import torch.nn as nn
from .core import Callback
from src.models.layers.revin import RevIN

class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, 
                        affine:bool=False, denorm:bool=True):
        """        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param denorm: if True, the output will be de-normalized

        This callback only works with affine=False.
        if affine=True, the learnable affine_weights and affine_bias are not learnt
        """
        super().__init__()
        self.num_features = num_features # 這筆資料有幾個特徵
        self.eps = eps
        self.affine = affine
        self.denorm = denorm
        self.revin = RevIN(num_features, eps, affine)
    

    def before_forward(self): 
        self.revin_norm() # 模型尚未產生 self.pred，此時還看不到 pred.shape，也不能檢查 self.pred.shape[-1]
    
    def after_forward(self): 
        if self.denorm and self.pred.shape[-1] == self.num_features: # 只有在輸出特徵維度 == 輸入特徵數時才做 denorm，否則會錯
            # print(f"[RevInCB] pred shape before denorm: {self.pred.shape}")
            self.revin_denorm() 
        
    def revin_norm(self):
        """
        1.) 把輸入資料 self.xb 傳進 RevIN 模組裡面，做 normalization(標準化)
        2.) 得到的新資料 xb_revin (shape 不變，但每個特徵做了 Z-score 標準化），並且 直接改寫 self.learner.xb = xb_revin。
        =>  這代表：模型接下來看到的輸入就是標準化後的特徵。
        """
        xb_revin = self.revin(self.xb, 'norm') # xb_revin: [bs x seq_len x nvars]
        self.learner.xb = xb_revin

    def revin_denorm(self):
        """
        ! 對 self.pred 做 反標準化(denorm), 然後還原回原始特徵維度的shape, 也就是 nvars = 6
        """
        pred = self.revin(self.pred, 'denorm') # pred: [bs x target_window x nvars]
        self.learner.pred = pred
    


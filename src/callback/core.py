
__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB']

"""
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict
"""

from ..basics import *
import torch

DTYPE = torch.float32

class Callback(GetAttr): 
    _default='learner'


class SetupLearnerCB(Callback): 
    """
    模型與資料搬到 GPU 的自動工具
    1.) 在訓練開始前, 把模型搬到GPU。
    2.) 在每個 batch 開始前，把資料 batch 搬到 GPU (或指定的 device)。
    3.) 確保 learner 自己的 batch 屬性裝好 (xb, yb)，供後續使用。
    """
    def __init__(self):        
        self.device = default_device(use_cuda=True) # 檢查你電腦有沒有 CUDA（NVIDIA GPU）。

    # 每次 batch 開始時，不論是訓練、驗證、預測、測試，都先呼叫 _to_device()。
    # 這確保你的 batch 資料（通常是 tensor）會搬到 GPU 上執行，而不是卡在 CPU。
    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)
        #print(f"Batch content before unpacking: {batch}")  # Debug statement
        try:
            if self.n_inp > 1:
                xb = batch[0]
                yb = batch[1] if len(batch) > 1 else None
            else:
                xb, yb = batch, None # 如果只有單一 input，直接裝到 xb，yb 設為 None。
        except ValueError as e:
            #print(f"Error unpacking batch: {e}")
            raise e
        self.learner.batch = xb, yb # 最後存進 self.learner.batch，讓後面的運算知道要用哪個 input 與 target。


        
    def before_fit(self): 
        """
        Set model to cuda before training
        把模型 .to(self.device) → 丟到 GPU 或 CPU。
        同時把 device 記到 learner 裡，後續可以用。
        """
        self.learner.model.to(self.device)
        self.learner.device = self.device                        


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)

    def after_predict(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()

         

class GetTestCB(Callback):
    """
    test時收集結果的 callback
    """
    def __init__(self):
        super().__init__() # 繼承自 Callback 父類別，標準初始化，沒有特別設定。

    def before_test(self):
        self.preds, self.targets = [], [] # 建立空的 preds, targets
                                          # self.preds ➔ 用來收集每個 batch 的 預測值 pred；
                                          # self.targets ➔ 用來收集每個 batch 的 真實答案 target (yb)
    
    def after_batch_test(self):        
        # append the prediction after each forward batch
        # 每跑完一個 batch 測試後，把當前 batch 的預測結果 (self.pred) 與 真實答案 (self.yb) 分別 append 加進 list 裡！
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    def after_test(self):
        # 整個測試結束後， 把累積在 list 裡的 所有 batch 的 preds、targets，用 torch.concat 接成一個大 tensor！
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()
        self.targets = torch.concat(self.targets)#.detach().cpu().numpy()

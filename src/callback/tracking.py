__all__ = ['TrackTimerCB', 'TrackTrainingCB', 'PrintResultsCB', 'TerminateOnNaNCB',
            'TrackerCB', 'SaveModelCB', 'EarlyStoppingCB', 'CSVLogger']

from ..basics import *
from .core import Callback
import torch
import time
import numpy as np
from pathlib import Path
import csv # 將訓練過程中每一個 epoch 的損失與評估指標儲存為 .csv 檔


class TrackTimerCB(Callback):
    """
    用來記錄每個 epoch 花費的時間
    """
    def __init__(self):
        super().__init__()

    def before_fit(self):
        self.learner.epoch_time = None # 當訓練開始（fit 開始前），先把 epoch_time 清空。

    def before_epoch_train(self):         
        self.start_time = time.time() # 當一個 epoch 開始訓練時，記下開始時間。

    def after_epoch_train(self): 
        self.learner.epoch_time = self.format_time(time.time() - self.start_time) # 當 epoch 訓練結束時，算出當前時間減去開始時間，得到花費的秒數。

    def format_time(self, t): # 轉換成人類可讀的格式
        "Format `t` (in seconds) to (h):mm:ss"
        t = int(t)
        h, m, s = t // 3600, (t // 60) % 60, t % 60
        if h != 0:
            return f'{h}:{m:02d}:{s:02d}'
        else:
            return f'{m:02d}:{s:02d}'


class TrackTrainingCB(Callback):
    """
    用來「記錄訓練與驗證過程中各種數值」的 callback
    """

    def __init__(self, train_metrics=False, valid_metrics=True):
        super().__init__()        
        self.train_metrics, self.valid_metrics = train_metrics, valid_metrics # 要不要紀錄訓練集上的metrics 以及 要不要紀錄驗證集上的metrics

    def init_cb_(self):
        self.setup()    
        self.initialize_recorder()        
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False   

    def before_fit(self):        
        self.setup() # 決定要記哪些東西、判斷資料集是否有 valid 部分
        self.initialize_recorder() # 建立一個 recorder 字典（裡面放 epoch、loss、metrics 等欄位）
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = True if self.loss_func.reduction == 'mean' else False        
    
    def setup(self):
        """
        要不要記錄 validation loss、要不要計算 metrics
        """
        self.valid_loss = False
        if self.learner.dls: # 如果 learner.dls 存在（確定有 dataloader 資料）
            if not self.learner.dls.valid: self.valid_metrics = False # 如果沒有validation set，則強制關閉valid_metrics（因為沒東西驗證）。
            else: self.valid_loss = True # 如果有validation set，則打開self.valid_loss，代表要記錄validation loss。

        if self.metrics: # 如果 self.metrics 有提供（例如 [mse, mae]）
            if not isinstance(self.metrics, list): self.metrics = [self.metrics]   
            self.metric_names = [func.__name__ for func in self.metrics] # 提取這些函數的名稱，存在 self.metric_names（之後可以對應到記錄表欄位）。
        else: self.metrics, self.metric_names = [], []        
            
    def initialize_recorder(self):
        """
        建立一個字典，用來記錄每個 epoch 中重要指標的數值。
        """
        recorder = {'epoch': [],  'train_loss': []} 
        if self.valid_loss: recorder['valid_loss'] = []

        for name in self.metric_names: 
            if self.train_metrics: recorder['train_'+name] = [] # 紀錄訓練集對應指標。
            if self.valid_metrics: recorder['valid_'+name] = [] # 紀錄驗證集對應指標。
        self.recorder = recorder # 把這個 recorder 存入 callback 與 learner， 這樣後面其他 callback 也能用 learner.recorder 存取或更新它。
        self.learner.recorder = recorder            
        

    def initialize_batch_recorder(self, with_metrics):
        batch_recorder = {'n_samples': [], 'batch_losses': [], 'with_metrics': with_metrics}                                                         
        self.batch_recorder = batch_recorder

    def reset(self): 
        self.targs, self.preds = [],[]                
        self.n_samples = 0
        self.batch_loss = []


    def after_epoch(self):
        self.recorder['epoch'].append(self.epoch)
        self.learner.recorder = self.recorder              
        
    def before_epoch_train(self): 
        # define storage for batch training loss and metrics        
        self.initialize_batch_recorder(with_metrics=self.train_metrics)        
        self.reset()

    def before_epoch_valid(self):            
        # if valid data is available, define storage for batch training loss and metrics
        # if self.dls.valid:  self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.reset()


    def after_epoch_train(self): # 計算 epoch 統計值（整體平均 loss、metric 分數），並寫入 recorder。
        values = self.compute_scores()           
        # save training loss after one epoch                
        self.recorder['train_loss'].append( values['loss'] )
        # save metrics after one epoch         
        if self.train_metrics:
            for name, func in zip(self.metric_names, self.metrics): 
                self.recorder['train_'+name].append( values[name] ) 
            

    def after_epoch_valid(self): # 計算 epoch 統計值（整體平均 loss、metric 分數），並寫入 recorder。
        # if there is no valid data, don't store
        if not self.learner.dls.valid: return
        values = self.compute_scores()                
        # save training loss after one epoch
        self.recorder['valid_loss'].append( values['loss'] )
        # save metrics after one epoch         
        if self.valid_metrics:
            for name, func in zip(self.metric_names, self.metrics): 
                self.recorder['valid_'+name].append( values[name] ) 
            
    
    def after_batch_train(self): self.accumulate()  # save batch recorder                
    def after_batch_valid(self): self.accumulate()
        
    def accumulate(self ):
        xb, yb = self.batch
        bs = len(xb)                                
        self.batch_recorder['n_samples'].append(bs)
        # get batch loss 
        loss = self.loss.detach()*bs if self.mean_reduction_ else self.loss.detach()        
        self.batch_recorder['batch_losses'].append(loss)
        
        if yb is None: self.batch_recorder['with_metrics'] = False
        if len(self.metrics) == 0: self.batch_recorder['with_metrics'] = False
        # accumulate prediction and target          
        if self.batch_recorder['with_metrics']:
            self.preds.append(self.pred.detach().cpu())
            self.targs.append(yb.detach().cpu())
    

    def compute_scores(self):
        "calculate losses and metrics after each epoch"
        values = {}
        # calculate loss after each epoch        
        n = sum(self.batch_recorder['n_samples'])   # get total number of samples        
        values['loss'] = sum(self.batch_recorder['batch_losses']).item()/n  # averaging

        # calculate metrics if available after each epoch
        if len(self.preds) == 0: return values
        self.preds = torch.cat(self.preds)
        self.targs = torch.cat(self.targs)        
        for func in self.metrics:             
            # values[func.__name__] = func(self.targs, self.preds)
            values[func.__name__] = func(self.targs, self.preds)        
        return values
    

class TerminateOnNaNCB(Callback):
    " A callback to stop the training if loss is NaN"
    def after_batch_train(self):
        if torch.isinf(self.loss) or torch.isnan(self.loss): raise KeyboardInterrupt


class PrintResultsCB(Callback):
    """
    -- Learner.recorder 儲存每一個 epoch 訓練過程中的記錄,像是: train_loss, valid_loss, mse, mae 等。
    -- get_header()：把 recorder 的 key(指標名稱)抓出來，例如 ['train_loss', 'valid_loss', 'mse', 'mae'] + 'time'
    -- before_fit()：在訓練一開始會印出「表頭」。
    -- after_epoch():每次訓練完一個epoch,取出最新的數值,並使用 self.print_value.format(*epoch_logs) 印出結果。
    """
    def __init__(self):
        super().__init__()        

    def get_header(self, recorder):        
        "recorder is a dictionary"
        header = list(recorder.keys()) # 取出表頭  
        return header+['time'] # 加上一個 'time'

    def before_fit(self):
        if self.run_finder: return # don't print if lr_finder is called。若是 lr_finder() 階段，不顯示。
        if not hasattr(self.learner, 'recorder'): return      # don't print if there is no recorder。若沒有 recorder，不顯示。
        header = self.get_header(self.learner.recorder) # 會從 recorder 裡面抓出紀錄的 key。
        self.print_header = '{:>15s}'*len(header) # 靠右對齊、寬度為 15（不夠就補空格）、這欄是字串（string）。
        self.print_value = '{:>15d}' + '{:>15.6f}'*(len(header)-2) + '{:>15}' # 用來輸出每行「數值」的格式
        print(self.print_header.format(*header))        
    
    def after_epoch(self):      
        if self.run_finder: return # don't print if lr_finder is called。若是學習率尋找模式，不顯示結果。避免在 lr_finder()（尋找學習率階段）時，印出多餘的資訊。
        if not hasattr(self.learner, 'recorder'): return  # don't print if there is no recorder。初始化階段（模型未經訓練），若沒有 recorder，不顯示。
        epoch_logs = []        
        for key in self.learner.recorder: # 讀取 learner.recorder 裡紀錄的數值（如 loss、metrics）
            value=self.learner.recorder[key][-1] if self.learner.recorder[key] else None # 抓取每個紀錄項目中最後一筆（最新的）記錄
            epoch_logs += [value]
        if self.learner.epoch_time: epoch_logs.append(self.learner.epoch_time)
        # print('epoch_logs', epoch_logs)
        print(self.print_value.format(*epoch_logs)) # 第 N 個 epoch（從 0 開始）
                                                    # 訓練集上的 loss（training loss）
                                                    # 驗證集上的 loss（validation loss）
                                                    # 評估指標 1（可能是 MSE）
                                                    # 評估指標 2（可能是 MAE）
                                                    # 此 epoch 訓練花費時間（5 秒）        


class CSVLogger(Callback):
    """
    將每個 epoch 的 loss 與 metrics 儲存為 CSV 檔案
    """
    def __init__(self, save_dir='results', filename="epoch_log.csv"):
        super().__init__()
        self.save_path = Path(save_dir) / filename
        self.fields = None
        self.file = None
        self.writer = None
        self.header_written = False

    def before_fit(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.save_path, mode='w', newline='')
        self.writer = csv.writer(self.file)

    def after_epoch(self):
        if not hasattr(self.learner, 'recorder'): return
        row = []
        for key in self.learner.recorder:
            val = self.learner.recorder[key][-1] 
            if hasattr(val, "item"):
                val = val.item()  # EX. 將 tensor(0.35) 轉成 0.35
            row.append(val)

        if self.learner.epoch_time:
            row.append(self.learner.epoch_time)

        if not self.header_written:
            header = list(self.learner.recorder.keys()) + ['time']
            self.writer.writerow(header)
            self.header_written = True

        self.writer.writerow(row)

    def after_fit(self):
        if self.file:
            self.file.close()


class TrackerCB(Callback):
    """
    功能上是「監控某個指標」並偵測是否出現新的最佳值，但它不負責儲存模型，而是提供是否為最佳表現的判斷邏輯(self.new_best), 供其他callback使用, 例如SaveModelCB。
    """
    def __init__(self, monitor='train_loss', comp=None, min_delta=0.):
        super().__init__()
        if comp is None: comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater # 如果沒指定comp且監控的指標名稱中包含'loss'或'error'，就會使用comp = np.less，意思是：希望越小越好（e.g., loss, MAE, MSE）。
        if comp == np.less: min_delta *= -1 # 若是越小越好（np.less），那 min_delta 就要變成負數來做比較。
        self.monitor, self.comp, self.min_delta = monitor, comp, min_delta

    def before_fit(self): # 開始訓練
        if self.run_finder: return # 如果正在進行學習率尋找（lr_finder），就略過這個 callback，不儲存模型。
        if self.best is None: self.best = float('inf') if self.comp == np.less else -float('inf') # 設定目前最佳分數為 無限大（inf）（因為希望 loss 越小越好）。
        self.monitor_names = list(self.learner.recorder.keys()) # 取得可以監控的指標（來自 recorder，如 valid_loss, mse, mae）。
        assert self.monitor in self.monitor_names # 確保你設定的 monitor='valid_loss' 是 recorder 中存在的指標之一。

    def after_epoch(self): # 在每個 epoch 結束後，取出最新的指標值（如 valid_loss）。
        if self.run_finder: return # 如果是在學習率尋找流程中就略過
        val = self.learner.recorder[self.monitor][-1] # 從 learner.recorder 取出最新一筆 validation loss（或其他監控指標）。
        if self.comp(val - self.min_delta, self.best): self.best, self.new_best = val,True # 如果比過去好，就更新最佳值，並標記為這輪是最好的。
        else: self.new_best = False  # 否則標記為非最佳（不儲存模型）


class SaveModelCB(TrackerCB): # 繼承自 TrackerCB，核心是「監控指標 + 自動儲存」。
    """
    〔負責儲存模型〕每當出現新最佳指標，就自動儲存當下的模型參數（.pth 檔）。
    -- every_epoch: 如果不是 False, 代表每幾個 epoch 固定存檔一次，而不只存最佳模型。
    -- with_opt: 是否一起儲存 optimizer 狀態。
                如果 with_opt=False → 只儲存 模型參數(state_dict)；如果 with_opt=True → 會儲存 模型參數 + optimizer 狀態。
                如果只是訓練一輪，最後要的只是最佳模型來跑測試 → with_opt=False; 如果是分段式訓練（例如一次跑 50 個 epoch、存 checkpoint,下次再從 50 繼續跑 100) → with_opt=True。
    """
    def __init__(self, monitor='train_loss', comp=None, min_delta=0., 
                        every_epoch=False, fname='model', path=None, with_opt=False, save_process_id=0, global_rank=None):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)        
        self.every_epoch = every_epoch
        self.last_saved_path = None
        self.path, self.fname = path, fname
        self.with_opt = with_opt
        self.save_process_id = save_process_id

        if global_rank:
            self.global_rank = int(global_rank)
        else:
            if torch.cuda.is_available():
                self.global_rank = torch.cuda.current_device()
                if not torch.distributed.is_initialized():
                    self.save_process_id = self.global_rank
            else:
                self.global_rank = 0


    def _save(self, fname, path):
        if self.global_rank == self.save_process_id:
            print(f"Saving model to {fname} at path {path}")
            self.last_saved_path = self.learner.save(fname, path, with_opt=self.with_opt)

    def after_epoch(self):
        if self.every_epoch: # 如果設了 every_epoch，會根據 epoch 編號固定存檔。
            if ((self.epoch%self.every_epoch) == 0) or (self.epoch==self.n_epochs-1): 
                self._save(f'{self.fname}_{self.epoch}', self.path)                            
        else: # 如果沒設 every_epoch，則只要存最佳。
            super().after_epoch()
            if self.new_best: # 檢查這個 epoch 是否為最佳。
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save(f'{self.fname}', self.path) # 模型存檔（用 learner.save()）。

    def after_fit(self): # 在 after_fit()，如果 every_epoch 沒開啟，會在訓練結束後，把模型恢復為最佳紀錄。
        if self.run_finder: return
        if not self.every_epoch and self.global_rank == self.save_process_id:
            print(f"Loading best model from {self.last_saved_path}")
            self.learner.load(self.last_saved_path, with_opt=self.with_opt) # 載入最佳模型權重


class EarlyStoppingCB(TrackerCB):
    """
    當模型在驗證集（或指定監控指標）上長時間沒有進步時，提前結束訓練，避免浪費資源。
    """
    def __init__(self, monitor='train_loss', comp=None, min_delta=0, patient=5):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.patient = patient
    
    def before_fit(self):
        # set the impatient level （訓練開始前，重設 impatient_level（耐心計數器））。
        self.impatient_level = 0
        super().before_fit()
    
    def after_epoch(self):
        super().after_epoch() # # 先用 TrackerCB 判斷這一輪有沒有 new_best
        if self.new_best: # 有進步，耐心計數器歸零。
            self.impatient_level = 0   # reset the impatience
        else: #  # 沒進步，耐心計數器 +1
            self.impatient_level += 1
            if self.impatient_level > self.patient: # 超過 patient，直接 raise KeyboardInterrupt，中斷訓練。
                print(f'No improvement since epoch {self.epoch-self.impatient_level}: early stopping')
                raise KeyboardInterrupt


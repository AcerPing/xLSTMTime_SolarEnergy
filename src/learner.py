
from typing import List
import torch
from torch.optim import Adam, AdamW # -- SGD, RMSprop, Adadelta, Adagrad, RMSprop
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast

from .basics import *
from .callback.core import * 
from .callback.tracking import * 
from .callback.scheduler import *
from .callback.distributed import *
from .utils import *
from pathlib import Path
from tqdm import tqdm

import numpy as np

from sklearn.base import BaseEstimator
from unittest.mock import patch
import matplotlib.pyplot as plt


class Learner(GetAttr):
    """
    -- dls: 資料集、Dataloader
    -- model: 主模型(xLSTM)
    -- loss_func: 損失函數
    -- opt: 優化器(Adam)
    -- callbacks: 訓練流程控制（如 early stopping、save model)
    -- metrics: 評估指標(MSE、MAE)
    """

    def __init__(self, dls, model, 
                        loss_func=None, 
                        lr=1e-5, 
                        cbs=None, 
                        metrics=None, 
                        opt_func=AdamW, # -- Adam, AdamW 
                        weight_decay=1e-3, 
                        **kwargs): # 初始化資料集、模型、Loss、Optimizer、Callbacks
                
        self.model, self.dls, self.loss_func, self.lr = model, dls, loss_func, lr
        self.opt_func = opt_func # Adam
        self.weight_decay = weight_decay
        self.set_opt()
        self.metrics = metrics
        self.n_inp  = 2 # (x, y) 兩個元素，x 是 input（features 特徵）、y 指的是 ground truth（真實 y），
        # self.n_inp = self.dls.train.dataset.n_inp if self.dls else 0

        # Initialize callbacks                 
        if cbs and not isinstance(cbs, List): cbs = [cbs]    
        self.initialize_callbacks(cbs)
           
        # Indicator of running lr_finder
        self.run_finder = False

    def set_opt(self):
        if self.model:
            # 設定 optimizer 優化器。
            print(f"✅ Setting optimizer with lr={self.lr}, weight_decay={self.weight_decay}")
            self.opt = self.opt_func(self.model.parameters(), self.lr, weight_decay=self.weight_decay) # 用Adam做優化，用指定的學習率，並在每次梯度更新時自動對權重引入L2正則化。 # -- 1e-2
        else: self.opt = None


    def default_callback(self):
        "get a set of default callbacks"
        default_cbs = [ SetupLearnerCB(), # 設置 Learner，初始化與配置用
                        TrackTimerCB(), # 記錄訓練花費的時間
                        TrackTrainingCB(train_metrics=False, valid_metrics=True) #  在每個 epoch 結束後記錄驗證集的 metric。
                        ]                  
        return default_cbs
    

    def initialize_callbacks(self, cbs):
        """
        在 Learner 初始化時，把所有 callbacks 完整組裝好並連結到 Learner 身上。
        """
        default_cbs = self.default_callback() # 取得預設callbacks，拿到預設組（例如 SetupLearnerCB、TrackTimerCB、TrackTrainingCB）。
        self.cbs = update_callbacks(cbs, default_cbs) if cbs else default_cbs # 合併使用者傳入的cbs。如果使用者有傳入自訂cbs，就把它們和預設的default_cbs合併；如果沒傳，就只用預設的。
        self.cbs += [PrintResultsCB()] # add print CB 在終端機印出 epoch 訓練結果表格。
        for cb in self.cbs: cb.learner = self # 把 Learner 指派給每個 callback
        self('init_cb') # 〔觸發點〕呼叫所有 callback 的初始化


    def add_callback(self, cb):
        """
        加單一個 callback, 把單一個 callback (cb) 加進目前 Learner 的 self.cbs 列表裡，
        這樣 callback 就可以存取 Learner 裡的資料,例如模型、dataloader、參數等!
        """
        if not cb: return
        cb.learner = self # 把 Learner 指標傳進 callback (cb.learner = self)
        self.cbs = update_callback(cb, self.cbs)           

    def add_callbacks(self, cbs):
        """
        〔批次版〕可以一次加好幾個 callback,會自動把不是 list 的包成 list 處理。
        """
        if not isinstance(cbs, list):  cbs = [cbs]
        for cb in cbs: self.add_callback(cb)

    def remove_callback(self, cb): 
        cb.learn = None
        self.cbs, removed_cb = remove_callback(cb, self.cbs)
        return removed_cb
    
    def remove_callbacks(self, cb_list):
        for cb in cb_list: self.remove_callback(cb)


    def fit(self, n_epochs, lr=None, cbs=None, do_valid=True): 
        """ 
        fit the model 負責整體訓練過程
        """
        self.n_epochs = n_epochs
        if not self.dls.valid: do_valid = False
        if cbs: self.add_callbacks(cbs)
        if lr: self.opt = self.opt_func(self.model.parameters(), lr) 

        self('before_fit')
        try:
            for self.epoch in range(n_epochs):            
                self('before_epoch')                     
                self.one_epoch(train=True)            
                # if self.dls.valid:                    
                if do_valid: self.one_epoch(train=False)                
                self('after_epoch')        
        except KeyboardInterrupt: pass 
        self('after_fit')


    def fit_one_cycle(self, n_epochs, lr_max=None, pct_start=0.3): 
        """
        One Cycle Learning Rate(OneCycleLR): 學習率策略，「先提高學習率再慢慢降低」的學習率變化方式，幫助模型走出困境區、找到更好的參數組合。
        -- lr_max → 這輪訓練中最高學習率，如果沒指定就用預設 self.lr。
        -- pct_start 是指訓練週期中，有多少比例的 epoch 用於「學習率從初始值提升到最高值」。例如 pct_start=0.3 就是前 30% 時間往上爬，後 70% 慢慢降。
        """
        self.n_epochs = n_epochs        
        self.lr_max = lr_max if lr_max else self.lr
        cb = OneCycleLR(lr_max=self.lr_max, pct_start=pct_start)
        self.fit(self.n_epochs, cbs=cb) # 主訓練迴圈
         
         
    def one_epoch(self, train):                           
        self.epoch_train() if train else self.epoch_validate()        

    def epoch_train(self):
        self('before_epoch_train')
        self.model.train()                
        self.dl = self.dls.train
        self.all_batches('train')
        self('after_epoch_train')
    
    def epoch_validate(self, dl=None):
        self('before_epoch_valid')
        # model at evaluation mode  
        self.model.eval()                
        self.dl = dl if dl else self.dls.valid
        if self.dl:        
            with torch.no_grad(): self.all_batches('valid')
        self('after_epoch_valid')


    def all_batches(self, type_):
        # for self.num,self.batch in enumerate(progress_bar(dl, leave=False)):        
        for num, batch in enumerate(self.dl):            
            self.iter, self.batch = num, batch            
            if type_ == 'train': self.batch_train()
            elif type_ == 'valid': self.batch_validate()
            elif type_ == 'predict': self.batch_predict()             
            elif type_ == 'test': self.batch_test()

    def batch_train(self):
        self('before_batch_train')
        self._do_batch_train()
        self('after_batch_train')  

    def batch_validate(self):
        self('before_batch_valid')
        self._do_batch_validate()
        self('after_batch_valid')  
    
    def batch_predict(self):
        self('before_batch_predict')
        self._do_batch_predict()
        self('after_batch_predict') 

    def batch_test(self):
        self('before_batch_test')
        self._do_batch_test()
        self('after_batch_test') 
        
    def _do_batch_train(self):        
        # forward + get loss + backward + optimize          
        self.pred, self.loss = self.train_step(self.batch)                                      
        # zero the parameter gradients
        self.opt.zero_grad()                 
        # gradient
        self.loss.backward()
        # update weights
        self.opt.step() 

    def train_step(self, batch): 
        """
        每個 batch 的前向傳遞與 loss 計算
        """
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        # compute loss
        loss = self.loss_func(pred, self.yb)
        return pred, loss

    def model_forward(self):
        self('before_forward') # ! 觸發 Learner 所有 callback 的 before_forward() 方法
        self.pred = self.model(self.xb)
        self('after_forward') # ! 執行 callback 裡的 after_forward() 方法。
        return self.pred

    def _do_batch_validate(self):       
        # forward + calculate loss
        self.pred, self.loss = self.valid_step(self.batch)     

    def valid_step(self, batch): 
        """
        每個 batch 的前向傳遞與 loss 計算
        """
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        # compute loss
        loss = self.loss_func(pred, self.yb)
        return pred, loss                                     


    def _do_batch_predict(self):   
        self.pred = self.predict_step(self.batch)     
           
    def predict_step(self, batch):
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        return pred 
    
    def _do_batch_test(self):   
        self.pred, self.yb = self.test_step(self.batch)     
           
    def test_step(self, batch):
        # get the inputs
        self.xb, self.yb = batch
        # forward
        pred = self.model_forward()
        return pred, self.yb


    def _predict(self, dl=None):
        # self('before_validate')
        self('before_predict')
        if dl is None: return
        self.dl = dl
        self.n_inp = dl.dataset.n_inp                
        self.model.eval()        #  model at evaluation mode  
        with torch.no_grad(): self.all_batches('predict')        
        self('after_predict')


    def predict(self, test_data, weight_path=None, Dataset=None, Dataloader=None, batch_size=None): # 推論流程
        """_summary_
        Args:
            test_data can be a tensor, numpy array, dataset or dataloader
        Returns:
            _type_: _description_
        """                
        if weight_path is not None: self.load(weight_path)
        cb = GetPredictionsCB()
        self.add_callback(cb)                    
        test_dl = self._prepare_data(test_data, Dataset, Dataloader, batch_size)
        self._predict(test_dl)        
        self.preds = cb.preds
        return to_numpy(self.preds) 
   
    
    def test(self, dl, weight_path=None, scores=None): 
        """
        _summary_
        Args:
            test_data can be a tensor, numpy array, dataset or dataloader
        Returns:
            _type_: _description_
        
        測試流程，訓練完後，專門用來「載入權重、做推論、收集預測與真實值」的流程。
        讀進test資料 ➔ 預測 ➔ 收集預測與真實值 ➔（選擇性）計算分數 ➔ 回傳。
        """          
        # 載入 dataloader & 權重 -> 讀進資料、套上訓練好的模型。
        if dl is None: return # 是否載入資料集
        else: self.dl = dl
        if weight_path is not None: self.load(weight_path) # 如果提供了模型權重（.pth檔），就載入這個權重。
        
        # 預測所有 test 資料 -> 預測每個 batch，收集結果
        cb = GetTestCB() # 建立一個測試用的Callback，這個 Callback 會自動在測試時，把每個 batch 的預測(preds) 和 每個 batch 的正確答案(targets) 存起來！
        self.add_callback(cb)
        self('before_test') # 執行所有在 "before_test" 時應該觸發的 callbacks，可以想成是「測試開始前」的觸發點。 # * 這其實是在呼叫 Learner 類別的 __call__ 方法！ 觸發 GetTestCB.before_test()。
        self.model.eval() # 把模型切到 evaluation 模式（eval()）。
        with torch.no_grad(): # 用 torch.no_grad() 禁止梯度計算
            self.all_batches('test') # 處理完整個資料集（dataloader）所有 batch，並且對每個 batch 做forward 預測、收集結果的流程！
        self('after_test') # 測試結束後，執行所有 "after_test" callbacks。
        
        # 收集 pred & target -> 統一存起來，轉成 numpy
        self.preds, self.targets = to_numpy([cb.preds, cb.targets]) # 轉成 numpy 格式
        
        # calculate scores 計算分數
        if scores: # 如果有提供 scores（像是 MSE、MAE函數列表）
            s_names = [score_func.__name__ for score_func in scores] # 取得函數名稱（例如 'mse'）
            print(f'Metrics 函數名稱: {s_names}')
            s_vals = [score(cb.targets, cb.preds).to('cpu').numpy() for score in list(scores)] # 計算每個指標
            return self.preds, self.targets, s_vals # * 回傳：預測值、真實值、評分值
        else: return self.preds, self.targets


    def _prepare_data(self, test_data, Dataset=None, Dataloader=None, batch_size=None):
        if test_data is None: return test_data
        if Dataset and Dataloader:
            test_dset = Dataset(test_data)
            if not batch_size: batch_size=16
            test_dl = Dataloader(test_dset, batch_size)        
        else:            
            if self.dls: 
                # add test_data to the dataloader defined in the dls.train
                test_dl = self.dls.add_dl(test_data, batch_size=batch_size)  
            else: test_dl = test_data       # assume test_data is already a form of dataloader
        return test_dl
   
    
    def get_layer_output(self, inp, layers=None, unwrap=False):
        """
        Args:
            inp: can be numpy array, torch tensor or dataloader
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        if isinstance(inp, np.ndarray): inp = torch.Tensor(inp).to(device)
        if isinstance(inp, torch.Tensor): inp = inp.to(device)
        
        return get_layer_output(inp, model=self.model, layers=layers, unwrap=unwrap)
    

    def fine_tune(self, n_epochs, base_lr=None, freeze_epochs=1, pct_start=0.3):
        """
        fintune the pretrained model. First the entire model is freezed, only head is trained
        up to a freeze_epochs number. Then the model is unfreezed and the entire model is trained
        """
        assert (n_epochs>0)|(freeze_epochs>0), "Either n_epochs or freeze_epochs has to be > 0"
        if not base_lr: base_lr = self.lr
        # Finetune the head of freeze_epochs > 0:
        if freeze_epochs > 0:
            print('Finetune the head')
            self.freeze()
            self.fit_one_cycle(freeze_epochs, lr_max=base_lr, pct_start=pct_start)
        
        # Finetune the entire network if n_epochs > 0
        if n_epochs > 0:
            print('Finetune the entire network')        
            self.unfreeze()
            self.fit_one_cycle(n_epochs, lr_max=base_lr/2, pct_start=pct_start)
    

    def linear_probe(self, n_epochs, base_lr=None, pct_start=0.3):
        """
        linear probing the pretrained model. The model is freeze except the head during finetuning
        """
        assert (n_epochs>0), "n_epochs has to be > 0"
        if not base_lr: base_lr = self.lr
        print('Finetune the head')
        self.freeze()
        self.fit_one_cycle(n_epochs, lr_max=base_lr, pct_start=pct_start)
    

    def lr_finder(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp', show_plot=True, suggestion='valley'): 
        """
        find the learning rate
        自動尋找適合的學習率
        """
        n_epochs = num_iter//len(self.dls.train) + 1 # 計算要跑幾個 epoch 才能剛好跑完 num_iter 次迭代。
        # indicator of lr_finder method is applied
        self.run_finder = True # 讓系統知道現在是尋找學習率。
        # add LRFinderCB to callback list and will remove later
        cb = LRFinderCB(start_lr, end_lr, num_iter, step_mode, suggestion=suggestion) #　建立一個 LRFinderCB callback，負責在每個 batch 更新學習率、記錄 loss。
        # fit           
        self.fit(n_epochs=n_epochs, cbs=cb, do_valid=False) # 跑一個小的訓練階段，不跑驗證（do_valid=False），
        # should remove LRFinderCB callback after fitting                
        self.remove_callback(cb) # 跑完之後，把 LRFinderCB 移除，回到正常狀態。
        self.run_finder = False
        if show_plot: 
            cb.plot_lr_find() # 畫出「學習率 vs loss」的曲線圖。
            plt.show() #  x 軸 → 學習率（Learning Rate）； y 軸 → 對應的 Loss。
        if suggestion: return cb.suggested_lr # 回傳建議的學習率。
        

    def freeze(self):
        """ 
        freeze the model head
        require the model to have head attribute
        凍結模型的主體部分，只訓練 output head (輸出層)，是 Transfer Learning 或 Fine-tuning 常見的做法。
        只讓 model.head 可以被訓練，其餘主體參數全部凍結。
        """
        if hasattr(get_model(self.model), 'head'): # 檢查 self.model是否含有 .head 這個屬性。
            print('model head is available')
            for param in get_model(self.model).parameters(): param.requires_grad = False # 把整個模型所有參數的 requires_grad 設為 False，也就是不更新這些參數、凍結。
            for param in get_model(self.model).head.parameters(): param.requires_grad = True # 接著把 head 層的參數打開訓練，允許優化器更新這一部分的權重。
            print('model is frozen except the head')
            
            
    def unfreeze(self):
        for param in get_model(self.model).parameters(): param.requires_grad = True # 取得模型中所有可訓練的參數（如權重與偏差），並啟用該參數的梯度更新。


    def __call__(self, name): # 傳進來一個字串，例如 'before_test'
        for cb in self.cbs: # 遍歷所有 callback (self.cbs)
            attr = getattr(cb, name) # cb中是否有name這個屬性。
            if attr is not None: attr() # 如果某個 callback 有一個叫做 'before_test' 的方法，就去呼叫那個方法！
          

    def save(self, fname, path, **kwargs):
        """
        Save model and optimizer state (if `with_opt`) to `self.path/file`
        儲存模型參數
        """
        fname = join_path_file(fname, path, ext='.pth')        
        save_model(fname, self.model, getattr(self,'opt',None), **kwargs)
        return fname


    def load(self, fname, with_opt=False, device='cuda', strict=True, **kwargs):
        """
        load the model
        載入模型參數
        """
        if not torch.cuda.is_available():
            device = "cpu"
        load_model(fname, self.model, self.opt, with_opt, device=device, strict=strict)


    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        return params

    def _get_param_names(self):
        return (k for k in self.__dict__ if not k.endswith('_'))


    def set_params(self, **kwargs):
        params = {}
        for key, val in kwargs.items():
            params[key] = val
        BaseEstimator.set_params(self, **params)

    def to_distributed(self,
                       sync_bn=True,  # Whether to replace all batch norm with `nn.SyncBatchNorm`
                       **kwargs
                       ):
        local_rank = int(os.environ.get('LOCAL_RANK'))
        world_size = int(os.environ.get('WORLD_SIZE'))
        rank = int(os.environ.get('RANK'))
        print('Process {} (out of {})'.format(
            rank, torch.distributed.get_world_size()))

        self.add_callback(DistributedTrainer(local_rank=local_rank, world_size=world_size, sync_bn=sync_bn, **kwargs))

        return self


def save_model(path, model, opt, with_opt=True, pickle_protocol=2):
    "Save `model` to `file` along with `opt` (if available, and if `with_opt`)"
    if opt is None: with_opt=False
    state = get_model(model).state_dict()
    if with_opt: state = {'model': state, 'opt':opt.state_dict()}
    torch.save(state, path, pickle_protocol=pickle_protocol)


def load_model(path, model, opt=None, with_opt=False, device='cpu', strict=True):
    " load the saved model "
    state = torch.load(path, map_location=device)
    if not opt: with_opt=False
    model_state = state['model'] if with_opt else state
    get_model(model).load_state_dict(model_state, strict=strict)
    if with_opt: opt.load_state_dict(state['opt'])
    model = model.to(device)
      

def join_path_file(file, path, ext=''):
    "Return `path/file` if file is a string or a `Path`, file otherwise"
    if not isinstance(file, (str, Path)): return file
    if not isinstance(path, Path): path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path/f'{file}{ext}'


def get_model(model):
    "Return the model maybe wrapped inside `model`."    
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model


def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    """
    將預訓練模型的權重轉移到目前的模型。
    -- weights_path: 權重檔的路徑(.pth)。
    -- exclude_head: 若為 True, 則跳過名字中包含 'head' 的層（通常是輸出層）。
    """
    state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device) # 讀取儲存的 .pth 權重檔，並將權重放到指定的device。
    #print('new_state_dict',new_state_dict)
        
    # 將所有參數搬到 CPU，以便比較
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
    cpu_new_state_dict = {k: v.cpu() for k, v in new_state_dict.items()}
    # 比較每個參數是否完全相同
    all_equal = all(torch.equal(cpu_state_dict[k], cpu_new_state_dict[k]) for k in cpu_state_dict)
    print("✅ 完全相同！" if all_equal else "❌ 權重不一致！")

    print(f'Excluded head: {exclude_head}')
    if exclude_head:
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        print("Excluded head parameters:")
        for k in state_dict.keys():
            if k.startswith('head'):
                print(f" - {k}")
    
    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items(): # 逐層取出目前模型的每個參數名稱與內容。
        if exclude_head and 'head' in name: continue # 如果 exclude_head=True，就跳過包含 'head' 名稱的層，通常是 output head。
        if name in new_state_dict: # 如果這個層在預訓練模型裡也存在，且形狀一樣，就複製權重；否則記錄成 unmatched。
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape: param.copy_(input_param)
            else: unmatched_layers.append(name)
        else:
            unmatched_layers.append(name) # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0: raise Exception("No shared weight names were found between the models") # 若完全沒有層可對應，則拋出錯誤。
    else:
        if len(unmatched_layers) > 0: # 有未匹配層，EX. 預訓練模型的 .pth 權重中 沒有包含 layer_norm 層的參數。
            print(f'check unmatched_layers: {unmatched_layers}')
        else: # 完全匹配，表示此模型的架構與目前模型完全相符，且所有非 head 層的權重皆成功轉移。
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device) # 將模型放到指定裝置上並回傳。
    return model


def update_callback(cb, list_cbs): 
    """
    用來避免「callback重複」,確保同類型 callback 不重複！
    """
    for cb_ in list_cbs:
        if type(cb_) ==  type(cb): list_cbs.remove(cb_) # 如果原本 list_cbs 裡已經有相同類型的 callback，那就先移除舊的，再加上新的。
    list_cbs += [cb]
    return list_cbs

def update_callbacks(list_cbs, default_cbs):
    """
    把 使用者傳入的 callback 列表 (list_cbs) 逐一和 預設 callback 列表 (default_cbs) 做比對更新。
    換句話說，如果使用者傳入了一個 同類型的 callback, 就會用使用者的覆蓋掉預設的，否則就直接加入新的 callback。
    """
    for cb in list_cbs: default_cbs = update_callback(cb, default_cbs)
    return default_cbs

def remove_callback(cb, list_cbs):
    for cb_ in list_cbs:
        if type(cb_) ==  type(cb):             
            list_cbs.remove(cb_)
            break
    return list_cbs, cb_


def get_layer_output(inp, model, layers=None, unwrap=False):
    """
    layers is a list of module names
    """
    orig_model = model
    
    if unwrap: model = unwrap_model(model)
    if not layers: layers = list(dict(model.named_children()).keys())
    if not isinstance(layers, list): layers = [layers]

    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    # register forward hooks on the layers of choice    
    h_list = [getattr(model, layer).register_forward_hook(getActivation(layer)) for layer in layers]
    
    model.eval()
    out = orig_model(inp)    
    for h in h_list: h.remove()
    return activation

"""
建議學習順序
1. fit() → 看懂主訓練流程
2. train_step() / valid_step() → 如何執行一個 batch
3. lr_finder() → 自動搜尋學習率流程
4. test() / predict() → 測試與推論如何執行
5. callback → 如果要新增自訂流程（可選）

--------------------------------------

test()
    ↓
載入資料 & 權重
    ↓
建立 GetTestCB
    ↓
add_callback(GetTestCB)
    ↓
self('before_test')   # 叫大家做準備
    ↓
update_callback(GetTestCB, cbs_list)
    ↓
開始執行 all_batches('test')
    ↓
GetTestCB 在每個 batch 的 after_batch_test 觸發，收集 pred 和 target
    ↓
測試結束後 after_test，把 preds/targets 整理成大tensor
    ↓
Learner 回傳 preds, targets (out[0], out[1])


"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score, explained_variance_score


# plt.rcParams["font.size"] = 13 # matplotlib 字體設定
# 設定中文字體，例如 Noto Sans CJK 字體為默認字體
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
chinese_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = [chinese_font.get_name()] + plt.rcParams['font.family']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.sans-serif'] = plt.rcParams['font.sans-serif'] # 其他備選字體


def save_arguments(out_dir, args):
    """
    將 args (可以是 argparse.Namespace 或 dict) 儲存為 JSON 格式，存在指定的 out_dir 路徑下，檔名為 params.json。
    """
    os.makedirs(out_dir, exist_ok=True)  # 若資料夾不存在則建立
    path_arguments = os.path.join(out_dir, 'params.json')

    # 將 Namespace 轉為 dict（若已是 dict 就直接使用）
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args

    with open(path_arguments, mode="w") as f:
        json.dump(args_dict, f, indent=4)
    # print(f"✅ 已儲存參數至 {path_arguments}")


# 自訂RMSE函數 (內部函數)
def _rmse(mse_loss): # 因為Keras並未內建RMSE作為指標，需要自行定義一個自訂的RMSE指標函數。
    '''
    RMSE 是 mse 的平方根，更直觀地表示誤差，與實際數據單位一致。
    '''
    rmse_loss = np.sqrt(mse_loss)
    return rmse_loss


def save_metrics(actual: torch.Tensor, predicted: torch.Tensor, out_dir: str, model=None):
    """
    save mean squared error for tareget variable

    Args:
        actual: observed data for target variable # 實際值
        predicted: predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
        model : trained model (keras)
    """
    # 如果是 tensor，先轉成 numpy
    if isinstance(actual, torch.Tensor): actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor): predicted = predicted.cpu().numpy()
    # 展平成一維向量
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)

    # 使用 sklearn 進行驗算
    metrics_dict = {
        'MAE': mae(actual, predicted), # 計算平均絕對誤差
        'MSE': mse(actual, predicted), # 計算均方誤差
        'RMSE': _rmse(mse(actual, predicted)), # RMSE
        'R2_Score': r2_score(actual, predicted), # R-squared指標，反映模型解釋目標變數變異程度的能力。
        'Explained Variance Score': explained_variance_score(actual, predicted),
    }

    # Save as txt log
    with open(os.path.join(out_dir, 'log.txt'), 'w') as f:
        for k, v in metrics_dict.items():
            f.write(f'{k}: {v:.6f}\n')
        f.write('=' * 65 + '\n')

        if model: # 如果提供 PyTorch 模型，將模型架構印入 log
            f.write('\n===== PyTorch Model Summary =====\n')
            f.write(str(model) + '\n') # 將模型摘要資訊寫入文件。
    print(f"Metrics saved to: {os.path.join(out_dir, 'log.txt')}")


def save_lr_curve_from_csv(csv_path: str, out_dir: str, f_name: str = 'Learning_Curve'): 
    """
    學習率曲線程式
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(30, 10))
    plt.rcParams["font.size"] = 18 # 字體大小為 18。
    plt.plot(df['train_loss'], label='Train Loss', marker='o', markersize=5)
    plt.plot(df['valid_loss'], label='Validation Loss', marker='s', markersize=5)

    for i, value in enumerate(df['train_loss']):
        if i % 10 == 0:
            plt.annotate(f'{value:.4f}', xy=(i, value), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=12, color='blue', alpha=0.9)
    for i, value in enumerate(df['valid_loss']):
        if i % 10 == 0:
            plt.annotate(f'{value:.4f}', xy=(i, value), xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=12, color='orange', alpha=0.9)

    plt.title(f'[xLSTM] {f_name} (Model Loss)', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f'{f_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved to {save_path}")


def plot_feature_actual_vs_predicted(actual: torch.Tensor, predicted: torch.Tensor, out_dir: str):
    """
    Plot the actual vs predicted values for a specific feature for the first sequence.
    將每個特徵畫圖，繪製實際值與預測值圖表，用於觀察預測表現。

    Parameters: （紅色：預測、藍色：真實）
    - actual (np.array or torch.Tensor): Array of actual values. 真實目標資料
    - predicted (np.array or torch.Tensor): Array of predicted values. 預測出來的資料
    - feature_idx (int): Index of the feature to plot. 特徵索引
    """

    # 如果是 tensor，先轉成 numpy
    if isinstance(actual, torch.Tensor): actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor): predicted = predicted.cpu().numpy()
    
    # 【1】 reshape (batch_size, 1, 1) → (batch_size,)
    actual_flat = actual.reshape(-1)
    predicted_flat = predicted.reshape(-1)

    # 【2】 Select the first sequence for the given feature index （畫第 0 筆資料中，第 feature_idx 個特徵的所有時間點。）
    # 取出三維張量中特定序列與特徵的預測結果。假設資料維度為 actual.shape = (batch_size, target_points, num_features) 
    # 舉例：actual.shape = (64, 96, 7)，有 64 條不同序列、每條序列預測 96 個時間點、每個時間點包含 7 個特徵。
    # actual_feature = actual[0, :, feature_idx] # 選擇第 1 條序列（通常我們只拿來視覺化其中一筆），取出這筆序列的 所有時間步（96 個點），取出指定的某個特徵。
    # predicted_feature = predicted[0, :, feature_idx] # 畫第 0 筆資料中，第 feature_idx 個特徵的所有時間點。

    # 【3】 對「所有 batch 的同一個特徵」在每個時間點上進行平均，表示「平均趨勢線」，得到「這個特徵的整體預測趨勢 vs 真實趨勢」。
    #actual_feature = np.mean(actual[: , : ,feature_idx ], axis=0 )
    #predicted_feature = np.mean(predicted[: , : ,feature_idx ], axis=0)

    # 繪圖 Plot the first sequence
    plt.figure(figsize=(10, 6)) # figsize=(30, 10), plt.rcParams["font.size"] = 18 # 設置字體大小
    plt.plot(range(len(actual_flat)), actual_flat, label="Actual (Ground Truth)", color='blue', marker='o', markersize=2, linestyle='-') # plt.plot(actual_feature, label="Actual", color='blue')
    plt.plot(range(len(predicted_flat)), predicted_flat, label="Predicted", color='red', marker='x', markersize=2, linestyle='--') # plt.plot(predicted_feature, label="Predicted", color='red', linestyle='--')
    
    # 在每個點上顯示數據標籤 (實際數據)
    for i, value in enumerate(actual_flat):
        if i % 1000 == 0:  # 每隔 1000 個數據點顯示一次標籤
            plt.annotate(f'{value:.2f}', xy=(i, value), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', color='dodgerblue', fontsize=12, alpha=0.9)
    # 在每個點上顯示數據標籤 (預測數據)
    for i, value in enumerate(predicted_flat):
        if i % 1000 == 0:  # 每隔 1000 個數據點顯示一次標籤
            plt.annotate(f'{value:.2f}', xy=(i, value),  xytext=(0, -5), textcoords="offset points", ha='center', va='top', color='crimson', fontsize=12, alpha=0.9)
    
    plt.title(f"[xLSTM] Actual vs Predicted Fish Weight (All Test Data)")
    plt.xlabel("樣本序列")
    plt.xlim(0, len(actual_flat)) # 設置x軸範圍，從0到實際數據的長度。
    plt.ylabel("Fish Weight")
    plt.ylim(0, 1) # 設置y軸的顯示範圍為0到1。
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'actual_vs_predicted.png')
    plt.savefig(save_path, bbox_inches='tight') # 保存圖像
    plt.show()
    plt.close()
    print(f"Actual vs Predicted Plot saved to {save_path}")


def plot_error_histogram(actual, predicted, out_dir=None): # 誤差直方圖（Error Histogram）
    """
    - actual: 真實值。
    - predicted: 預測值。
    - out_dir: 若指定路徑，則儲存圖表；否則直接顯示。
    """
    # 確保是 numpy array
    if isinstance(actual, torch.Tensor): actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor): predicted = predicted.cpu().numpy()

    # 計算殘差
    residuals = actual.flatten() - predicted.flatten()

    plt.figure(figsize=(12, 8))
    plt.hist(residuals, bins='auto', color='deepskyblue', edgecolor='black', alpha=0.8, label='Residuals (actual - predicted)')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Ideal Line (Residual = 0)') # 畫理想線（Residual = 0）
    # 加上區域顏色
    # 填充Overestimation區域（Residual < 0） # min(residuals)~0
    plt.axvspan(-1, 0, facecolor='burlywood', alpha=0.2, label='OverEstimation Region (Residual < 0)') # 當 Residual < 0 時，表示實際值 < 預測值（高估）。
    # 填充Underestimation區域（Residual > 0） # 0~max(residuals)
    plt.axvspan(0, 1, facecolor='darkseagreen', alpha=0.2, label='UnderEstimation Region (Residual > 0)') # 當 Residual > 0 時，表示實際值 > 預測值（低估）。
    plt.xlabel("Residuals", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(f"[xLSTM] Error Histogram 誤差直方圖", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_dir:
        save_path = os.path.join(out_dir, f'Error Histogram.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"✅ 誤差直方圖已儲存至 {save_path}")


def plot_residuals(actual, predicted, out_dir=None): # 繪製殘差圖（Residual Plot）
    """
    - actual: 真實值。
    - predicted: 預測值。
    - out_dir: 若提供，將圖保存到此路徑。
    """
    # 確保是 numpy array
    if isinstance(actual, torch.Tensor): actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor): predicted = predicted.cpu().numpy()
    
    # Flatten 展平成一維
    actual = actual.flatten()
    predicted = predicted.flatten()
    residuals = actual - predicted # 計算殘差

    plt.figure(figsize=(12, 8))
    plt.scatter(predicted, residuals, color='blue',  alpha=0.6, s=10, label='Residuals (actual - predicted)') # 繪製殘差散點圖
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Ideal Line (Residual = 0)') # 畫水平0線
    # 標示模型低估與高估的區域
    plt.fill_between(x=np.linspace(0, 1, 500), y1=0, y2=1, color='lightgreen', alpha=0.2, label='Underestimation Region (Residual > 0)') # 淺綠色 (color='lightgreen'): 模型低估區域（殘差 > 0）。
    plt.fill_between(x=np.linspace(0, 1, 500), y1=-1, y2=0, color='lightsalmon', alpha=0.2, label='Overestimation Region (Residual < 0)') # 淺橙色 (color='lightsalmon'): 模型高估區域（殘差 < 0）。
    plt.title('[xLSTM] Residual Plot 殘差圖', fontsize=16)
    plt.xlabel('Predicted Values', fontsize=14)
    plt.xlim(0, 1) # 設定X軸，預測值範圍0到1。
    plt.ylabel('Residuals', fontsize=14)
    plt.ylim(-1, 1) # 設定Y軸，殘差範圍-1到1。
    plt.legend(loc='best', fontsize=12, frameon=True, edgecolor='black', fancybox=True)
    plt.tight_layout()
    if out_dir:
        save_path = os.path.join(out_dir, f'Residual Plot.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")
        plt.show()
        plt.close()
        print(f"Plot saved to {save_path}")


def save_yy_plot(actual: np.array, predicted: np.array, out_dir: str):
    """
    - actual: 真實值。
    - predicted: 預測值。
    - out_dir: 若提供，將圖保存到此路徑。
    """
        # 確保是 numpy array
    if isinstance(actual, torch.Tensor): actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor): predicted = predicted.cpu().numpy()
    
    # Flatten 展平成一維
    actual = actual.flatten()
    predicted = predicted.flatten()

    plt.figure(figsize=(10, 10))  # 設定圖表大小
    plt.rcParams["font.size"] = 18  # 設置字體大小

    plt.plot(actual, predicted, 'b.', label='Predicted vs Observed') # 繪製散點圖：預測值 vs 實際值
    diagonal = np.linspace(0, 1, 10000) # 繪製對角線（理想線）
    plt.plot(diagonal, diagonal, 'r--', label='Ideal Line') # 繪製一條紅色的對角線，表示理想狀況下的預測值與實際值相等。散點越接近這條線，表示預測越準確。

    # 標示高估與低估區域
    plt.fill_between(diagonal, diagonal, 1, color='peachpuff', alpha=0.2, label='Overestimation Region') # 淺橙色區域 (color='peachpuff '): 表示模型的高估區域（y_pred_test_time > y_test_time）。
    plt.fill_between(diagonal, 0, diagonal, color='palegreen', alpha=0.2, label='Underestimation Region') # 淺綠色區域 (color='peachpuff '): 表示模型的低估區域（y_pred_test_time < y_test_time）。
    # 設定範圍
    plt.xlim(0, 1) # 設定x軸的範圍為0到1。
    plt.ylim(0, 1) # 設定y軸的範圍為0到1。
    # 標題與標籤
    plt.title('觀察值與預測值的對角線分析圖', fontsize=16)
    plt.xlabel('Observed', fontsize=14) # 設置x軸標籤。
    plt.ylabel('Predicted', fontsize=14) # 設置y軸標籤。
    plt.legend(loc='best', fontsize=12) # 加入圖例
    # 儲存圖檔
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'yy_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Plot saved to {save_path}")
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

class BatteryPINNDataset(Dataset):
    def __init__(self, csv_path, use_cache=True):
        cache_path = os.path.splitext(csv_path)[0] + "_cache.pt"
        if use_cache and os.path.exists(cache_path):
            cached = torch.load(cache_path)
            if "y_min" in cached and "y_max" in cached:
                self.X = cached["X"]
                self.Y = cached["Y"]
                self.y_min = float(cached["y_min"])
                self.y_max = float(cached["y_max"])
                # Keep scaler attributes for compatibility, even when loaded from cache.
                self.scaler_X = None
                self.scaler_Y = None
                print(f"✓ Loaded cached dataset: {len(self.X)} samples")
                print(f"✓ X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
                print(f"✓ Y range: [{self.Y.min():.4f}, {self.Y.max():.4f}]")
                return
            print(f"⚠️ Cache metadata incomplete for {cache_path}, rebuilding cache...")

        # 1. 加载并插值
        df_raw = pd.read_csv(csv_path)
        df_temp = df_raw[['ExpTimeTemp', 'MidIntTemp', 'MidSurfTemp', 'NegSurfTemp', 'PosSurfTemp']].dropna()
        df_temp = df_temp.rename(columns={'ExpTimeTemp': 'ExpTime'})
        df_main = df_raw[['ExpTime', 'IntPre', 'CellVoltage']].copy()
        df_combined = pd.merge(df_main, df_temp, on='ExpTime', how='outer').sort_values('ExpTime').reset_index(drop=True)
        
        for col in ['MidIntTemp', 'MidSurfTemp', 'NegSurfTemp', 'PosSurfTemp']:
            df_combined[col] = df_combined[col].interpolate().bfill().ffill()

        # 2. 坐标映射 (Melt)
        df_melted = df_combined.melt(id_vars=['ExpTime', 'IntPre', 'CellVoltage'], 
                                     value_vars=['MidIntTemp', 'MidSurfTemp', 'NegSurfTemp', 'PosSurfTemp'],
                                     var_name='Loc', value_name='T')
        coord_map = {'MidIntTemp':(0.5,0.5,0.5), 'MidSurfTemp':(1,0.5,0.5), 'NegSurfTemp':(1,0.5,0), 'PosSurfTemp':(1,0.5,1)}
        coords = df_melted['Loc'].map(coord_map)
        df_melted[['x','y','z']] = pd.DataFrame(coords.tolist(), index=df_melted.index)
        
        # 3. 数据清理 - 移除 NaN 和异常值
        df_melted = df_melted.dropna()
        
        # 移除异常值 (使用 IQR 方法)
        for col in ['ExpTime', 'IntPre', 'CellVoltage', 'T']:
            Q1 = df_melted[col].quantile(0.25)
            Q3 = df_melted[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_melted = df_melted[(df_melted[col] >= lower_bound) & (df_melted[col] <= upper_bound)]
        
        print(f"✓ Dataset loaded: {len(df_melted)} samples")
        
        # 4. 归一化 (use separate scalers for X and Y)
        self.scaler_X = MinMaxScaler()
        self.scaler_Y = MinMaxScaler()
        # 输入 X: [t, x, y, z, I(补0), V, P]
        df_melted['I'] = 0.0 # 假设无电流测量
        cols_to_norm = ['ExpTime', 'x', 'y', 'z', 'I', 'CellVoltage', 'IntPre']
        
        X_raw = df_melted[cols_to_norm].values
        Y_raw = df_melted[['T']].values
        
        # 检查数据有效性
        if np.any(np.isnan(X_raw)) or np.any(np.isnan(Y_raw)):
            raise ValueError("❌ NaN found in data after preprocessing!")
        if np.any(np.isinf(X_raw)) or np.any(np.isinf(Y_raw)):
            raise ValueError("❌ Inf found in data after preprocessing!")
        
        # 归一化
        X_norm = self.scaler_X.fit_transform(X_raw)
        Y_norm = self.scaler_Y.fit_transform(Y_raw)

        self.y_min = float(self.scaler_Y.data_min_[0])
        self.y_max = float(self.scaler_Y.data_max_[0])
        
        # 转换为 tensor，同时限制数值范围
        self.X = torch.FloatTensor(np.clip(X_norm, -1e3, 1e3))
        self.Y = torch.FloatTensor(np.clip(Y_norm, -1e3, 1e3))

        if use_cache:
            torch.save({"X": self.X, "Y": self.Y, "y_min": self.y_min, "y_max": self.y_max}, cache_path)
            print(f"✓ Saved dataset cache: {cache_path}")
        
        print(f"✓ X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        print(f"✓ Y range: [{self.Y.min():.4f}, {self.Y.max():.4f}]")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloader(csv_path, batch_size=512, use_cache=True):
    dataset = BatteryPINNDataset(csv_path, use_cache=use_cache)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_dataset_from_csv_files(csv_files, use_cache=True, global_y_normalize=True):
    """Load one or multiple CSV files and return a single dataset object."""
    if not csv_files:
        raise ValueError("csv_files is empty")

    datasets = []
    for csv_path in csv_files:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print(f"\nLoading dataset file: {csv_path}")
        ds = BatteryPINNDataset(csv_path, use_cache=use_cache)
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    if global_y_normalize:
        global_y_min = min(ds.y_min for ds in datasets)
        global_y_max = max(ds.y_max for ds in datasets)
        span = max(1e-8, global_y_max - global_y_min)

        for ds in datasets:
            local_span = max(1e-8, ds.y_max - ds.y_min)
            y_physical = ds.Y * local_span + ds.y_min
            ds.Y = torch.clamp((y_physical - global_y_min) / span, 0.0, 1.0)
            ds.y_min = float(global_y_min)
            ds.y_max = float(global_y_max)

        print(
            f"✓ Applied global Y normalization across datasets: "
            f"[{global_y_min:.4f}, {global_y_max:.4f}]"
        )

    merged = ConcatDataset(datasets)
    print(f"\n✓ Combined dataset loaded: {len(merged)} samples from {len(datasets)} files")
    return merged
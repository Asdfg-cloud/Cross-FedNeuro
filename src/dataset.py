import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


class RestMetaDataset(Dataset):
    def __init__(self, site_id, root_dir, mode='private', transform=None, return_clinical=False):
        """
        Args:
            site_id (str): 站点名称
            root_dir (str): 数据根目录
            mode (str): 'private' (Label only) or 'public' (Clinical+Image)
        """
        self.mode = mode
        self.fc_path = os.path.join(root_dir, site_id, 'FC_matrices.npy')
        self.csv_path = os.path.join(root_dir, site_id, 'phenotypic.csv')
        self.return_clinical = return_clinical

        if os.path.exists(self.fc_path):
            self.data = np.load(self.fc_path).astype(np.float32)
            self.meta = pd.read_csv(self.csv_path)
            # 提取标签
            self.labels = self.meta['Label'].values.astype(np.longlong)
            # 提取临床特征并归一化 (Z-score)
            clinical_cols = ['Age', 'Sex', 'HAMD', 'HAMA']
            self.clinical = self.meta[clinical_cols].values.astype(np.float32)
            self.clinical = (self.clinical - self.clinical.mean(0)) / (self.clinical.std(0) + 1e-6)
        else:
            # Fallback for code testing without data
            print(f"Warning: Data for {site_id} not found. Using synthetic data.")
            self.data = np.random.randn(100, 90, 90).astype(np.float32)
            self.labels = np.random.randint(0, 2, 100)
            self.clinical = np.random.randn(100, 4).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).flatten()
        clinical = torch.from_numpy(self.clinical[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.mode == 'public':
            # public mode
            return img, clinical, label
        else:  # private mode
            if self.return_clinical:
                return img, clinical, label
            else:
                return img, label

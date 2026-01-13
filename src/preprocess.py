import os
import glob
import numpy as np
import scipy.io as sio
from tqdm import tqdm

def process_site_data(site_path, output_dir, roi_count=116):
    """
    读取单个站点下所有的 .mat 文件，计算 FC，并保存为 .npy
    (修复版：增加了对无效值 NaN/Inf 的处理)
    """
    site_name = os.path.basename(site_path)
    print(f"Processing Site: {site_name}...")

    mat_files = glob.glob(os.path.join(site_path, "*.mat"))
    if len(mat_files) == 0:
        print(f"  No .mat files found in {site_path}, skipping.")
        return

    site_fc_matrices = []
    valid_subjects = 0

    for mat_file in tqdm(mat_files, desc=f"  Loading {site_name}"):
        try:
            mat_data = sio.loadmat(mat_file)

            # 1. 寻找正确的时间序列变量
            time_series = None
            for key in mat_data:
                if key.startswith('__'): continue
                data = mat_data[key][:, :116]
                # 兼容 (ROI, T) 或 (T, ROI)
                if isinstance(data, np.ndarray) and (data.shape[0] == roi_count or data.shape[1] == roi_count):
                    time_series = data
                    break

            if time_series is None:
                # 尝试放松条件，查找是否包含 'ROISignals' 关键字
                for key in mat_data:
                    if 'ROISignals' in key:
                        time_series = mat_data[key]
                        break

            if time_series is None:
                print(f"  Warning: No valid ROI data found in {mat_file}")
                continue

            # 2. 维度转置 -> 确保形状为 (116, T)
            if time_series.shape[0] != roi_count:
                time_series = time_series.T

            # 3. 计算 Pearson Correlation (忽略除以0的警告)
            with np.errstate(invalid='ignore', divide='ignore'):
                fc_matrix = np.corrcoef(time_series)

            # --- 修复核心：处理 NaN ---
            # 如果某个ROI全是0，std为0，corrcoef会产生 NaN。将 NaN 替换为 0 (表示无相关)
            fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # 4. Fisher r-to-z 变换
            # 截断数值，防止 arctanh(1.0) 产生 inf
            # 1.0 可能会出现在自身相关或完全共线的情况
            fc_matrix = np.clip(fc_matrix, -0.99999, 0.99999)

            fc_matrix = np.arctanh(fc_matrix)

            # 将对角线置为0 (自身相关信息通常无用且值很大)
            np.fill_diagonal(fc_matrix, 0)

            site_fc_matrices.append(fc_matrix)
            valid_subjects += 1

        except Exception as e:
            print(f"  Error processing {mat_file}: {e}")

    # 5. 聚合保存
    if valid_subjects > 0:
        final_data = np.stack(site_fc_matrices, axis=0).astype(np.float32)
        save_path = os.path.join(output_dir, site_name)
        os.makedirs(save_path, exist_ok=True)

        # 再次检查最终数据是否包含 NaN，作为双重保险
        if np.isnan(final_data).any():
            print(f"  WARNING: Final data for {site_name} still contains NaNs! Replacing with 0.")
            final_data = np.nan_to_num(final_data, nan=0.0)

        np.save(os.path.join(save_path, "FC_matrices.npy"), final_data)
        print(f"  Saved {valid_subjects} subjects to {save_path}/FC_matrices.npy")
    else:
        print(f"  No valid subjects processed for {site_name}")

if __name__ == "__main__":
    # 配置
    RAW_DATA_ROOT = r"D:\Dataset\Rest-meta-MDD"  # 你之前的整理好的按站点分类的文件夹
    PROCESSED_DATA_ROOT = r"../data/Rest-meta-MDD"        # 新的工程目录数据文件夹
    ROI_COUNT = 116                                       # AAL116

    # 遍历所有站点文件夹 (S1, S2, ...)
    # 假设 RAW_DATA_ROOT 下面直接是 S1, S2...
    site_dirs = [d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))]

    for site in site_dirs:
        full_site_path = os.path.join(RAW_DATA_ROOT, site)
        process_site_data(full_site_path, PROCESSED_DATA_ROOT, ROI_COUNT)
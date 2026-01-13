import pandas as pd
import numpy as np
import os
import yaml
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# ==========================================
# 配置读取函数
# ==========================================
def load_config(config_path='config.yaml'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ==========================================
# 核心插补逻辑
# ==========================================
def train_imputation_models(df_all, public_site_id):
    """
    利用公共站点数据训练回归模型，用于MDD患者的插补。
    """
    print(f"Training imputation models using Anchor Site: {public_site_id}...")

    # 提取公共站点且数据完整的样本
    df_pub = df_all[df_all['Site'] == public_site_id].copy()
    df_clean = df_pub.dropna(subset=['HAMD', 'HAMA'])

    if len(df_clean) < 10:
        raise ValueError(f"Anchor site {public_site_id} has insufficient complete data for regression!")

    # 1. 训练 HAMD -> HAMA 的回归模型
    reg_hamd_to_hama = LinearRegression()
    reg_hamd_to_hama.fit(df_clean[['HAMD']].values, df_clean['HAMA'].values)
    r2_1 = reg_hamd_to_hama.score(df_clean[['HAMD']].values, df_clean['HAMA'].values)

    # 2. 训练 HAMA -> HAMD 的回归模型
    reg_hama_to_hamd = LinearRegression()
    reg_hama_to_hamd.fit(df_clean[['HAMA']].values, df_clean['HAMD'].values)
    r2_2 = reg_hama_to_hamd.score(df_clean[['HAMA']].values, df_clean['HAMD'].values)

    # 3. 计算 MDD 患者的均值 (作为Fallback)
    mdd_stats = df_clean[df_clean['Label'] == 1][['HAMD', 'HAMA']].mean()

    print(f"  > Regression Models Ready. R2(D->A): {r2_1:.2f}, R2(A->D): {r2_2:.2f}")

    return {
        'd2a': reg_hamd_to_hama,
        'a2d': reg_hama_to_hamd,
        'mdd_mean': mdd_stats
    }

def impute_row_logic(row, models):
    """
    单行数据的插补逻辑
    row: Pandas Series
    models: 训练好的模型字典
    """
    hamd = row['HAMD']
    hama = row['HAMA']
    label = row['Label'] # 0=HC, 1=MDD

    # -------------------------------------------------------
    # 策略 A: 健康对照组 (HC, Label=0) 的缺失处理
    # -------------------------------------------------------
    # 使用 "Clinical Prior-based Stochastic Injection"
    # 强行注入 U(0, 6) 的低分噪音，防止表征坍缩
    # -------------------------------------------------------
    if label == 0:
        # 如果 HAMD 缺失，随机生成 0-6 之间的数 (保留1位小数)
        r_hamd = hamd if pd.notna(hamd) else np.round(np.random.uniform(0, 6), 1)
        # 如果 HAMA 缺失，同上
        r_hama = hama if pd.notna(hama) else np.round(np.random.uniform(0, 6), 1)
        return r_hamd, r_hama

    # -------------------------------------------------------
    # 策略 B: 抑郁症患者 (MDD, Label=1) 的缺失处理
    # -------------------------------------------------------
    # 使用回归插补或均值插补
    # -------------------------------------------------------

    # 情况 1: 两个都缺 -> 使用 Anchor Site 的 MDD 均值
    if pd.isna(hamd) and pd.isna(hama):
        return models['mdd_mean']['HAMD'], models['mdd_mean']['HAMA']

    # 情况 2: 有 HAMD，缺 HAMA -> 使用回归预测
    elif pd.isna(hama):
        pred_hama = models['d2a'].predict([[hamd]])[0]
        return hamd, max(0, float(pred_hama)) # 确保不为负

    # 情况 3: 有 HAMA，缺 HAMD -> 使用回归预测
    elif pd.isna(hamd):
        pred_hamd = models['a2d'].predict([[hama]])[0]
        return max(0, float(pred_hamd)), hama

    # 情况 4: 都不缺 -> 保持原样
    return hamd, hama

# ==========================================
# 主处理流程
# ==========================================
def process_phenotypic_data():
    # 1. 加载配置
    try:
        cfg = load_config()
        raw_excel = cfg['data']['raw_excel_path']
        out_root = cfg['data']['processed_root']
        public_site = cfg['data']['public_site']
    except Exception as e:
        print(f"Configuration Error: {e}")
        return

    print(f"Loading Raw Excel: {raw_excel}")
    df_all = pd.read_excel(raw_excel, sheet_name="All")

    # 2. 统一列名 (请根据实际Excel表头调整 mapping)
    rename_map = {
        'Diagnosis': 'Label',
        'HAMD_17': 'HAMD',
        'HAMA_14': 'HAMA',
        'sub_id': 'ID',       # 或者是 ID
        'site': 'Site'        # 或者是 Site
    }
    # 仅重命名存在的列
    df_all = df_all.rename(columns={k:v for k,v in rename_map.items() if k in df_all.columns})

    # 数据清洗：将 Label 转为数字 (有些表可能是 'HC'/'MDD' 字符串)
    if df_all['Label'].dtype == object:
        df_all['Label'] = df_all['Label'].map({'HC': 0, 'MDD': 1, 0:0, 1:1})

    # 强制转换数值列
    for col in ['HAMD', 'HAMA', 'Age', 'Sex', 'Label']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    # 3. 训练插补模型
    impute_models = train_imputation_models(df_all, public_site)

    # 4. 遍历目标文件夹，生成对应的 CSV
    # 我们只为那些已经生成了 .npy 影像数据的站点生成 csv
    if not os.path.exists(out_root):
        print(f"Output root {out_root} does not exist. Please run .mat preprocessing first.")
        return

    site_dirs = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]

    print(f"Processing phenotypic data for {len(site_dirs)} sites...")

    for site in tqdm(site_dirs):
        site_dir = os.path.join(out_root, site)

        # 4.1 筛选该站点数据
        df_site = df_all[df_all['Site'] == site].copy()

        if len(df_site) == 0:
            print(f"  [Warning] Site {site} has no entries in Excel!")
            continue

        # 4.2 关键步骤：与 .npy 文件对齐
        # 假设 .npy 是按照 process_mat_files 生成的，顺序通常是按文件名(ID)排序
        # 我们需要按照同样的逻辑对 DataFrame 排序
        df_site = df_site.sort_values(by='ID')

        # (可选) 如果 process_mat.py 保存了 ids.txt，这里应该读取并 reindex
        # 假设: df_site['ID'] 与文件名一致 (去除.mat后缀)

        # 4.3 执行插补
        imputed_cols = df_site.apply(lambda row: impute_row_logic(row, impute_models), axis=1, result_type='expand')
        df_site[['HAMD', 'HAMA']] = imputed_cols

        # 4.4 填充 Age 和 Sex (简单中位数/众数填充)
        df_site['Age'] = df_site['Age'].fillna(df_site['Age'].median())
        df_site['Sex'] = df_site['Sex'].fillna(0) # 假设0为男性或众数

        # 4.5 保存
        # 只保留需要的列
        final_cols = ['ID', 'Label', 'Age', 'Sex', 'HAMD', 'HAMA']
        # 确保列存在
        save_cols = [c for c in final_cols if c in df_site.columns]

        save_path = os.path.join(site_dir, 'phenotypic.csv')
        df_site[save_cols].to_csv(save_path, index=False)

    print("Phenotypic data processing complete.")

if __name__ == "__main__":
    # 设置随机种子以保证插补结果可复现
    np.random.seed(42)
    process_phenotypic_data()
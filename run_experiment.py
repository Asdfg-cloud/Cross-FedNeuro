import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
import logging
import statistics
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src import (
    RestMetaDataset,
    ResNetFC,
    BrainTransformer,
    MDDClassifier,
    ClinicalNet,
    ClientTrainer,
    ServerTrainer,
    seed_everything,
    get_logger
)


# ================= 配置读取 =================
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ================= 站点扫描工具 =================
def get_client_site_ids(data_root, public_site_id):
    all_items = os.listdir(data_root)
    client_ids = [
        item for item in all_items
        if os.path.isdir(os.path.join(data_root, item))
           and item.startswith('S')
           and item != public_site_id
    ]
    # 按 S 后面的数字排序
    client_ids.sort(key=lambda x: int(x[1:]))
    return client_ids

# ================= 计算指标辅助函数 =================
def calculate_metrics(y_true, y_probs, threshold=0.5):
    """
    计算 AUC, ACC, SEN, SPE
    """
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.5  # 极端情况处理

    y_pred = (np.array(y_probs) > threshold).astype(int)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)

    # Sensitivity (Recall) = TP / (TP + FN)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return auc, acc, sen, spe

# ================= 主程序 =================
def main():
    cfg = load_config('config.yaml')
    seed_everything(cfg['experiment']['seed'])
    device = torch.device(cfg['experiment']['device'])
    logger = get_logger('Cross-FedNeuro', cfg['experiment']['save_dir'])
    logger.info(f"Config loaded. Starting 5-Fold Cross-Validation.")

    # 准备公共数据 (Server端数据是不变的)
    public_id = cfg['data']['public_site']
    public_dataset = RestMetaDataset(
        site_id=public_id,
        root_dir=cfg['data']['processed_root'],
        mode='public'
    )
    public_loader = DataLoader(public_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)

    # 获取所有私有站点 ID
    data_root = cfg['data']['processed_root']
    public_site = cfg['data']['public_site']
    all_site_ids = get_client_site_ids(data_root, public_site)

    # 用 n_clients 截断
    limit_n = cfg['federated'].get('n_clients', len(all_site_ids))
    site_ids = all_site_ids[:limit_n]
    logger.info(f"Sites Selected: {site_ids}")

    # --- 混合模态分配 ---
    n_total = len(site_ids)
    ratio = cfg['federated'].get('multimodal_ratio', 0.5)  # 默认 50% 多模态
    n_multi = int(n_total * ratio)
    multi_site_set = set(site_ids[:n_multi])
    logger.info(f"Multimodal Clients ({len(multi_site_set)}): {list(multi_site_set)}")

    # ================= 5折交叉验证循环 =================
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg['experiment']['seed'])

    # 存储每一折的最佳结果 (包含 AUC, ACC, SEN, SPE)
    folds_results = [] # List of dicts

    for fold_idx in range(n_folds):
        logger.info(f"\n{'=' * 20} Start Fold {fold_idx + 1} / {n_folds} {'=' * 20}")

        # --- 步骤 A: 初始化异构服务端 ---
        server = ServerTrainer(cfg, device)

        # --- 步骤 B: 初始化本折的客户端 ---
        clients = []
        for site_id in site_ids:
            full_ds = RestMetaDataset(
                site_id,
                cfg['data']['processed_root'],
                mode='private',
                return_clinical=(site_id in multi_site_set)
            )

            dummy_X = np.zeros(len(full_ds))
            site_labels = full_ds.labels
            train_idx, test_idx = list(skf.split(dummy_X, site_labels))[fold_idx]

            train_subset = Subset(full_ds, train_idx)
            test_subset = Subset(full_ds, test_idx)

            train_loader = DataLoader(train_subset, batch_size=cfg['data']['batch_size'], shuffle=True, drop_last=True)
            test_loader = DataLoader(test_subset, batch_size=cfg['data']['batch_size'], shuffle=False)

            # --- 异构模型分配 ---
            model_choice = random.choice(['resnet', 'transformer'])
            classifier_input_dim = 0
            backbone = None
            if model_choice == 'resnet':
                classifier_input_dim = cfg['model']['hidden_dim']
                backbone = ResNetFC(input_dim=cfg['data']['roi_count'] ** 2,
                                    hidden_dim=classifier_input_dim,
                                    feature_dim=cfg['model']['feature_dim'])
            elif model_choice == 'transformer':
                backbone = BrainTransformer(num_rois=cfg['data']['roi_count'],
                                            d_model=256,
                                            feature_dim=cfg['model']['feature_dim'])
                classifier_input_dim = 256

            model = MDDClassifier(backbone, hidden_dim=classifier_input_dim).to(device)

            cli_model = None
            is_multimodal = site_id in multi_site_set
            if is_multimodal:
                cli_model = ClinicalNet(input_dim=cfg['data']['clinical_dim'],
                                        feature_dim=cfg['model']['feature_dim']).to(device)

            trainer = ClientTrainer(site_id, model, train_loader, public_loader, cfg, device,
                                    is_multimodal=is_multimodal,
                                    clinical_model=cli_model)

            clients.append({
                'id': site_id,
                'trainer': trainer,
                'test_loader': test_loader
            })

        # --- 步骤 C: 联邦训练循环 ---
        best_fold_metric = 0.0 # 用于选模型的指标 (通常用 AUC)
        best_fold_stats = {}   # 记录最佳时刻的所有指标

        for round_idx in range(cfg['federated']['n_rounds']):
            # 1. 下发
            g_reps = server.get_global_reps()

            # 2. 采样
            sample_rate = cfg['federated'].get('client_sample_rate', 1.0)
            num_selected = max(1, int(len(clients) * sample_rate))
            selected_clients = random.sample(clients, num_selected)

            # 3. 训练 & 上传
            client_uploads = []
            for c in selected_clients:
                loss = c['trainer'].train_epoch(g_reps)
                pkg = c['trainer'].get_upload_package()
                client_uploads.append(pkg)

            # 4. 聚合
            server.aggregate(client_uploads)

            # 5. 评估 (Evaluation) - 计算综合指标
            # 策略：汇总所有客户端的预测结果，计算 micro-average 指标
            # 或者：计算每个客户端的指标然后平均 (macro-average)。这里采用汇总所有样本计算，更直观。

            total_labels = []
            total_probs = []

            for c in clients:
                trainer = c['trainer']
                loader = c['test_loader']
                trainer.model.eval()

                c_labels, c_probs = [], []
                with torch.no_grad():
                    for batch_data in loader:
                        if (len(batch_data) == 3):
                            imgs, _, labels = batch_data
                        else:
                            imgs, labels = batch_data

                        imgs = imgs.to(device)
                        outputs = trainer.model(imgs)

                        if outputs.shape[1] == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        else:
                            probs = torch.sigmoid(outputs).squeeze()

                        c_labels.extend(labels.cpu().numpy())
                        c_probs.extend(probs.cpu().numpy())

                total_labels.extend(c_labels)
                total_probs.extend(c_probs)

            # 计算本轮所有数据的综合指标
            r_auc, r_acc, r_sen, r_spe = calculate_metrics(total_labels, total_probs)

            # 记录最佳结果 (以 AUC 为准)
            if r_auc > best_fold_metric:
                best_fold_metric = r_auc
                best_fold_stats = {
                    'AUC': r_auc,
                    'ACC': r_acc,
                    'SEN': r_sen,
                    'SPE': r_spe
                }

            if (round_idx + 1) % 5 == 0:
                logger.info(f"  [Fold {fold_idx + 1}] Round {round_idx + 1} | AUC: {r_auc:.4f} | ACC: {r_acc:.4f}")

        # 本折结束
        logger.info(f"Fold {fold_idx + 1} Best Result: {best_fold_stats}")
        folds_results.append(best_fold_stats)

    # ================= 实验结束，输出统计 =================
    logger.info(f"\n{'=' * 20} 5-Fold Cross-Validation Final Report {'=' * 20}")

    # 提取各指标列表
    metrics_keys = ['AUC', 'ACC', 'SEN', 'SPE']
    final_stats = {}

    for key in metrics_keys:
        values = [res[key] for res in folds_results]
        if len(values) > 0:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            mean_val, std_val = 0.0, 0.0

        final_stats[key] = f"{mean_val:.4f} ± {std_val:.4f}"
        logger.info(f"{key}: {final_stats[key]}")

    logger.info("============================================================")

if __name__ == "__main__":
    main()
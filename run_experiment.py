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
import pandas as pd

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


# ================= 辅助函数：计算 Sen/Spe =================
def calculate_metrics(y_true, y_pred):
    """根据真实标签和预测标签计算 ACC, SEN, SPE"""
    # 处理二分类混淆矩阵
    # labels: [0, 1] 确保即使只有一类也能正确解包，但在 batch 很小且单一类别时需小心
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        sen = tp / (tp + fn + 1e-10) # 敏感度 / 召回率
        spe = tn / (tn + fp + 1e-10) # 特异度

        return acc, sen, spe
    except ValueError:
        # 极端情况处理
        return 0.0, 0.0, 0.0


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
    # 定义 K-Fold 切分器 (Shuffle=True 配合固定Seed，保证可复现)
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg['experiment']['seed'])
    folds_best_results = {'auc': [], 'acc': [], 'sen': [], 'spe': []}

    # 用于存储所有折的收敛曲线数据
    convergence_records = []

    # 存储每一折的最佳结果 (包含多个指标)
    folds_best_results = {
        'auc': [], 'acc': [], 'sen': [], 'spe': []
    }

    for fold_idx in range(n_folds):
        logger.info(f"\n{'=' * 20} Start Fold {fold_idx + 1} / {n_folds} {'=' * 20}")

        # --- 步骤 A: 初始化异构服务端 (必须重置) ---
        server = ServerTrainer(cfg, device)

        # --- 步骤 B: 初始化本折的客户端 (划分数据 & 重置模型) ---
        clients = []

        for site_id in site_ids:
            # 加载该站点的完整数据集
            full_ds = RestMetaDataset(
                site_id,
                cfg['data']['processed_root'],
                mode='private',
                return_clinical=(site_id in multi_site_set)  # 标记是否返回临床数据
            )

            # --- 关键：使用 StratifiedKFold 获取本折的索引 ---
            dummy_X = np.zeros(len(full_ds))
            site_labels = full_ds.labels

            train_idx, test_idx = list(skf.split(dummy_X, site_labels))[fold_idx]

            # 创建 Subset
            train_subset = Subset(full_ds, train_idx)
            test_subset = Subset(full_ds, test_idx)

            # 创建 DataLoader
            train_loader = DataLoader(train_subset, batch_size=cfg['data']['batch_size'], shuffle=True, drop_last=True)
            test_loader = DataLoader(test_subset, batch_size=cfg['data']['batch_size'], shuffle=False)

            # --- 异构模型分配 ---
            model_choice = random.choice(['resnet', 'transformer'])
            # model_choice = 'resnet'
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

            # 多模态客户端需要 ClinicalNet
            cli_model = None
            is_multimodal = site_id in multi_site_set
            if is_multimodal:
                cli_model = ClinicalNet(input_dim=cfg['data']['clinical_dim'],
                                        feature_dim=cfg['model']['feature_dim']).to(device)
            # 初始化 Trainer
            trainer = ClientTrainer(site_id, model, train_loader, public_loader, cfg, device,
                                    is_multimodal=is_multimodal,
                                    clinical_model=cli_model)

            clients.append({
                'id': site_id,
                'trainer': trainer,
                'test_loader': test_loader
            })

        # --- 步骤 C: 联邦训练循环 (N Rounds) ---
        best_fold_auc = 0.0
        # 同时记录产生最佳 AUC 时的其他指标
        best_fold_metrics = {'auc': 0.0, 'acc': 0.0, 'sen': 0.0, 'spe': 0.0}

        for round_idx in range(cfg['federated']['n_rounds']):
            # 1. 服务端分发全局特征
            g_reps = server.get_global_reps()

            # 2. 客户端采样
            sample_rate = cfg['federated'].get('client_sample_rate', 1.0)
            num_selected = max(1, int(len(clients) * sample_rate))
            selected_clients = random.sample(clients, num_selected)
            selected_ids = [c['id'] for c in selected_clients]
            logger.info(f"  Round {round_idx + 1} Selected: {selected_ids}")

            # 3. 客户端训练 & 上传
            client_uploads = []
            for c in selected_clients:
                loss = c['trainer'].train_epoch(g_reps)
                pkg = c['trainer'].get_upload_package()
                client_uploads.append(pkg)

            # 4. 服务端聚合
            server.aggregate(client_uploads)

            # 5. 评估 (Evaluation) - 计算多指标
            round_metrics = {'auc': [], 'acc': [], 'sen': [], 'spe': []}

            for c in clients:
                trainer = c['trainer']
                loader = c['test_loader']
                trainer.model.eval()

                all_labels, all_probs, all_preds = [], [], []
                with torch.no_grad():
                    for batch_data in loader:
                        if (len(batch_data) == 3):
                            imgs, _, labels = batch_data
                        else:
                            imgs, labels = batch_data

                        imgs = imgs.to(device)
                        outputs = trainer.model(imgs)

                        # 计算概率
                        if outputs.shape[1] == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        else:
                            probs = torch.sigmoid(outputs).squeeze()

                        # 计算预测类别 (Threshold = 0.5)
                        preds = (probs > 0.5).long()

                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())

                # --- 计算单站点指标 ---
                # AUC
                try:
                    auc = roc_auc_score(all_labels, all_probs)
                except ValueError:
                    auc = 0.5

                # ACC, SEN, SPE
                acc, sen, spe = calculate_metrics(all_labels, all_preds)

                round_metrics['auc'].append(auc)
                round_metrics['acc'].append(acc)
                round_metrics['sen'].append(sen)
                round_metrics['spe'].append(spe)

            # 计算本轮所有站点的平均指标
            avg_auc = np.mean(round_metrics['auc'])
            avg_acc = np.mean(round_metrics['acc'])
            avg_sen = np.mean(round_metrics['sen'])
            avg_spe = np.mean(round_metrics['spe'])

            # 收集收敛数据: 记录 (折数, 轮数, 平均AUC)
            convergence_records.append({
                'fold': fold_idx,
                'round': round_idx,
                'avg_auc': avg_auc
            })

            # 根据 AUC 择优
            if avg_auc > best_fold_auc:
                best_fold_auc = avg_auc
                best_fold_metrics = {
                    'auc': avg_auc, 'acc': avg_acc,
                    'sen': avg_sen, 'spe': avg_spe
                }

            if (round_idx + 1) % 5 == 0:
                logger.info(f"  [Fold {fold_idx + 1} Round {round_idx + 1}] "
                            f"AUC: {avg_auc:.4f} | ACC: {avg_acc:.4f} | "
                            f"SEN: {avg_sen:.4f} | SPE: {avg_spe:.4f}")

        # 本折结束，记录最佳结果
        logger.info(f"Fold {fold_idx + 1} Best Result -> "
                    f"AUC: {best_fold_metrics['auc']:.4f}, "
                    f"ACC: {best_fold_metrics['acc']:.4f}, "
                    f"SEN: {best_fold_metrics['sen']:.4f}, "
                    f"SPE: {best_fold_metrics['spe']:.4f}")

        for k in folds_best_results:
            folds_best_results[k].append(best_fold_metrics[k])

        # t-SNE 数据提取 (仅在最后一折进行)
        # 目的：获取测试集在特征空间的高维表示，用于画散点图
        if fold_idx == n_folds - 1:
            logger.info("Extracting features for t-SNE (Last Fold)...")
            tsne_feats, tsne_labels, tsne_sites = [], [], []

            for c in clients:
                c['trainer'].model.eval()
                with torch.no_grad():
                    for batch_data in c['test_loader']:
                        if len(batch_data) == 3: imgs, _, labels = batch_data
                        else: imgs, labels = batch_data

                        imgs = imgs.to(device)

                        # --- 关键：获取 Backbone 的输出特征，而不是分类头的 Logits ---
                        # 假设 MDDClassifier 结构为 self.backbone -> self.classifier
                        # 如果没有直接的方法，这里直接调用 backbone
                        features = c['trainer'].model.backbone(imgs)

                        tsne_feats.append(features.cpu().numpy())
                        tsne_labels.append(labels.cpu().numpy())
                        # 记录站点ID，以便观察是否存在站点效应
                        tsne_sites.extend([c['id']] * len(labels))

            # 保存 t-SNE 数据
            tsne_save_path = os.path.join(cfg['experiment']['save_dir'], 'tsne_data.npz')
            np.savez(tsne_save_path,
                     features=np.concatenate(tsne_feats),
                     labels=np.concatenate(tsne_labels),
                     sites=np.array(tsne_sites))
            logger.info(f"t-SNE data saved to {tsne_save_path}")

    # ================= 实验结束，输出统计 =================
    logger.info(f"\n{'=' * 20} 5-Fold Cross-Validation Final Results {'=' * 20}")

    # 保存收敛曲线数据
    df_convergence = pd.DataFrame(convergence_records)
    conv_save_path = os.path.join(cfg['experiment']['save_dir'], 'convergence_curve.csv')
    df_convergence.to_csv(conv_save_path, index=False)
    logger.info(f"Convergence data saved to {conv_save_path}")

    # 计算均值和标准差
    final_stats = {}
    for k, v in folds_best_results.items():
        if len(v) > 0:
            mean_val = statistics.mean(v)
            std_val = statistics.stdev(v) if len(v) > 1 else 0.0
            final_stats[k] = (mean_val, std_val)
        else:
            final_stats[k] = (0.0, 0.0)

    logger.info(f"AUC: {final_stats['auc'][0]:.4f} ± {final_stats['auc'][1]:.4f}")
    logger.info(f"ACC: {final_stats['acc'][0]:.4f} ± {final_stats['acc'][1]:.4f}")
    logger.info(f"SEN: {final_stats['sen'][0]:.4f} ± {final_stats['sen'][1]:.4f}")
    logger.info(f"SPE: {final_stats['spe'][0]:.4f} ± {final_stats['spe'][1]:.4f}")


if __name__ == "__main__":
    main()
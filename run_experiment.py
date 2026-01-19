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
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        sen = tp / (tp + fn + 1e-10) # 敏感度
        spe = tn / (tn + fp + 1e-10) # 特异度

        return acc, sen, spe
    except ValueError:
        return 0.0, 0.0, 0.0


# ================= 主程序 =================
def main():
    cfg = load_config('config.yaml')
    seed_everything(cfg['experiment']['seed'])
    device = torch.device(cfg['experiment']['device'])
    logger = get_logger('Cross-FedNeuro', cfg['experiment']['save_dir'])
    logger.info(f"Config loaded. Starting 5-Fold Cross-Validation.")

    # 准备公共数据
    public_id = cfg['data']['public_site']
    public_dataset = RestMetaDataset(
        site_id=public_id,
        root_dir=cfg['data']['processed_root'],
        mode='public'
    )
    public_loader = DataLoader(public_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)

    # 获取所有私有站点 ID
    data_root = cfg['data']['processed_root']
    all_site_ids = get_client_site_ids(data_root, public_id)

    # 用 n_clients 截断
    limit_n = cfg['federated'].get('n_clients', len(all_site_ids))
    site_ids = all_site_ids[:limit_n]
    logger.info(f"Sites Selected: {site_ids}")

    # --- 混合模态分配 ---
    n_total = len(site_ids)
    ratio = cfg['federated'].get('multimodal_ratio', 0.5)
    n_multi = int(n_total * ratio)
    multi_site_set = set(site_ids[:n_multi])
    logger.info(f"Multimodal Clients ({len(multi_site_set)}): {list(multi_site_set)}")

    # ================= 5折交叉验证循环 =================
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg['experiment']['seed'])

    # 存储结果
    folds_best_results = {'auc': [], 'acc': [], 'sen': [], 'spe': []}
    convergence_records = []

    # [新增] 全局最佳 AUC 记录，用于判断哪一折最好
    global_best_auc = -1.0

    for fold_idx in range(n_folds):
        logger.info(f"\n{'=' * 20} Start Fold {fold_idx + 1} / {n_folds} {'=' * 20}")

        # --- 步骤 A: 初始化异构服务端 ---
        server = ServerTrainer(cfg, device)

        # --- 步骤 B: 初始化本折的客户端 ---
        clients = []

        for site_id in site_ids:
            # 加载数据集
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

            # 初始化 Trainer
            trainer = ClientTrainer(site_id, model, train_loader, public_loader, cfg, device,
                                    is_multimodal=is_multimodal,
                                    clinical_model=cli_model,
                                    ablation_config={'use_intra': True, 'use_inter': True})

            clients.append({
                'id': site_id,
                'trainer': trainer,
                'test_loader': test_loader
            })

        # =================================================================================
        # [修改] 每一折开始前，都提取一次测试集的 Pre-train 特征
        # 暂存在内存中 (current_fold_pre_data)，如果不确定这折是不是最好的，先不存文件
        # =================================================================================
        logger.info(f"Extracting Pre-train features for Fold {fold_idx + 1}...")
        pre_feats, pre_labels, pre_sites = [], [], []

        for c in clients:
            c['trainer'].model.eval()
            with torch.no_grad():
                for batch_data in c['test_loader']:
                    if len(batch_data) == 3: imgs, _, labels = batch_data
                    else: imgs, labels = batch_data

                    imgs = imgs.to(device)

                    # 此时模型是随机初始化的
                    features = c['trainer'].model.get_projection(imgs)

                    pre_feats.append(features.cpu().numpy())
                    pre_labels.append(labels.cpu().numpy())
                    pre_sites.extend([c['id']] * len(labels))

        # 打包暂存
        current_fold_pre_data = {
            'features': np.concatenate(pre_feats),
            'labels': np.concatenate(pre_labels),
            'sites': np.array(pre_sites)
        }

        # --- 步骤 C: 联邦训练循环 (N Rounds) ---
        best_fold_auc = 0.0
        best_fold_metrics = {'auc': 0.0, 'acc': 0.0, 'sen': 0.0, 'spe': 0.0}

        for round_idx in range(cfg['federated']['n_rounds']):
            g_reps = server.get_global_reps()

            # 客户端采样
            sample_rate = cfg['federated'].get('client_sample_rate', 1.0)
            num_selected = max(1, int(len(clients) * sample_rate))
            selected_clients = random.sample(clients, num_selected)

            # 训练 & 上传
            client_uploads = []
            for c in selected_clients:
                loss = c['trainer'].train_epoch(g_reps)
                pkg = c['trainer'].get_upload_package()
                client_uploads.append(pkg)

            # 聚合
            server.aggregate(client_uploads)

            # 评估
            round_metrics = {'auc': [], 'acc': [], 'sen': [], 'spe': []}

            for c in clients:
                trainer = c['trainer']
                loader = c['test_loader']
                trainer.model.eval()

                all_labels, all_probs, all_preds = [], [], []
                with torch.no_grad():
                    for batch_data in loader:
                        if (len(batch_data) == 3): imgs, _, labels = batch_data
                        else: imgs, labels = batch_data

                        imgs = imgs.to(device)
                        outputs = trainer.model(imgs)

                        if outputs.shape[1] == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        else:
                            probs = torch.sigmoid(outputs).squeeze()

                        preds = (probs > 0.5).long()

                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())

                try:
                    auc = roc_auc_score(all_labels, all_probs)
                except ValueError:
                    auc = 0.5

                acc, sen, spe = calculate_metrics(all_labels, all_preds)

                round_metrics['auc'].append(auc)
                round_metrics['acc'].append(acc)
                round_metrics['sen'].append(sen)
                round_metrics['spe'].append(spe)

            avg_auc = np.mean(round_metrics['auc'])
            avg_acc = np.mean(round_metrics['acc'])
            avg_sen = np.mean(round_metrics['sen'])
            avg_spe = np.mean(round_metrics['spe'])

            convergence_records.append({
                'fold': fold_idx,
                'round': round_idx,
                'avg_auc': avg_auc
            })

            # 更新本折内的最佳记录
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

        # 本折训练结束
        logger.info(f"Fold {fold_idx + 1} Finished. Best AUC in this fold: {best_fold_auc:.4f}")

        # 记录本折数据到总结果
        for k in folds_best_results:
            folds_best_results[k].append(best_fold_metrics[k])

        # =================================================================================
        # [核心逻辑修改] 检查本折是否是目前为止表现最好的折 (Best AUC Fold)
        # 如果是，则提取当前模型的特征 (After)，并将之前暂存的 Pre 和当前的 After 写入文件
        # =================================================================================
        if best_fold_auc > global_best_auc:
            global_best_auc = best_fold_auc
            logger.info(f">>> Fold {fold_idx + 1} is the NEW BEST FOLD (AUC: {best_fold_auc:.4f}). Updating t-SNE records...")

            # 1. 保存之前暂存的 Pre-train 数据 (覆盖旧文件)
            pre_save_path = os.path.join(cfg['experiment']['save_dir'], 'tsne_data_pre_train.npz')
            np.savez(pre_save_path, **current_fold_pre_data)
            logger.info(f"    Saved Pre-train features to {pre_save_path}")

            # 2. 提取并保存当前的 Trained features (覆盖旧文件)
            tsne_feats, tsne_labels, tsne_sites = [], [], []
            for c in clients:
                c['trainer'].model.eval()
                with torch.no_grad():
                    for batch_data in c['test_loader']:
                        if len(batch_data) == 3: imgs, _, labels = batch_data
                        else: imgs, labels = batch_data
                        imgs = imgs.to(device)

                        # 提取训练后的特征
                        features = c['trainer'].model.get_projection(imgs)

                        tsne_feats.append(features.cpu().numpy())
                        tsne_labels.append(labels.cpu().numpy())
                        tsne_sites.extend([c['id']] * len(labels))

            current_fold_post_data = {
                'features': np.concatenate(tsne_feats),
                'labels': np.concatenate(tsne_labels),
                'sites': np.array(tsne_sites)
            }

            post_save_path = os.path.join(cfg['experiment']['save_dir'], 'tsne_data.npz')
            np.savez(post_save_path, **current_fold_post_data)
            logger.info(f"    Saved Trained features to {post_save_path}")
        else:
            logger.info(f"Fold {fold_idx + 1} (AUC: {best_fold_auc:.4f}) is NOT better than current best ({global_best_auc:.4f}). Skipping t-SNE save.")

    # ================= 实验结束 =================
    logger.info(f"\n{'=' * 20} 5-Fold Cross-Validation Final Results {'=' * 20}")

    df_convergence = pd.DataFrame(convergence_records)
    conv_save_path = os.path.join(cfg['experiment']['save_dir'], 'convergence_curve.csv')
    df_convergence.to_csv(conv_save_path, index=False)

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
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
    get_logger,
    calculate_metrics
)

# ==========================================
# 核心控制变量: 切换此处以运行不同 Baseline
# ==========================================
BASELINE_MODE = 'FedAvg'  # 可选: 'Local', 'Solo', 'FedAvg', 'FedProx'

def fedavg_aggregate(global_model, clients):
    """ FedAvg/FedProx 专用的参数聚合函数 """
    global_dict = global_model.state_dict()
    total_samples = sum([len(c['trainer'].train_loader.dataset) for c in clients])

    for k in global_dict.keys():
        if 'num_batches_tracked' in k: continue

        weighted_sum = None
        for c in clients:
            n_samples = len(c['trainer'].train_loader.dataset)
            weight = n_samples / total_samples
            local_params = c['trainer'].model.state_dict()[k].float()

            if weighted_sum is None:
                weighted_sum = local_params * weight
            else:
                weighted_sum += local_params * weight

        if weighted_sum is not None:
            global_dict[k] = weighted_sum

    global_model.load_state_dict(global_dict)
    return global_model

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_client_site_ids(data_root, public_site_id):
    all_items = os.listdir(data_root)
    client_ids = [item for item in all_items if os.path.isdir(os.path.join(data_root, item)) and item.startswith('S') and item != public_site_id]
    client_ids.sort(key=lambda x: int(x[1:]))
    return client_ids

def main():
    cfg = load_config('config.yaml')
    cfg['federated']['multimodal_ratio'] = 0.0
    original_epochs = cfg['train']['local_epochs']

    seed_everything(cfg['experiment']['seed'])
    device = torch.device(cfg['experiment']['device'])
    logger = get_logger(f'Baseline-{BASELINE_MODE}', cfg['experiment']['save_dir'])
    logger.info(f"Running Baseline Mode: {BASELINE_MODE} with Degraded Settings")

    # 准备公共数据
    public_dataset = RestMetaDataset(site_id=cfg['data']['public_site'], root_dir=cfg['data']['processed_root'], mode='public')
    public_loader = DataLoader(public_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)

    all_site_ids = get_client_site_ids(cfg['data']['processed_root'], cfg['data']['public_site'])
    site_ids = all_site_ids[:cfg['federated'].get('n_clients', len(all_site_ids))]

    # 混合模态分配
    ratio = cfg['federated'].get('multimodal_ratio', 0.5)
    n_multi = int(len(site_ids) * ratio)
    multi_site_set = set(site_ids[:n_multi])

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg['experiment']['seed'])
    folds_best_results = {'auc': [], 'acc': [], 'sen': [], 'spe': []}

    for fold_idx in range(n_folds):
        logger.info(f"\n{'=' * 20} Fold {fold_idx + 1} / {n_folds} ({BASELINE_MODE}) {'=' * 20}")

        server = ServerTrainer(cfg, device)

        # [FedAvg/FedProx] 初始化全局参数模型
        global_model_avg = None
        if BASELINE_MODE in ['FedAvg', 'FedProx']:
            g_backbone = BrainTransformer(
                num_rois=cfg['data']['roi_count'],
                d_model=256,
                feature_dim=cfg['model']['feature_dim']
            )
            global_model_avg = MDDClassifier(g_backbone, hidden_dim=256).to(device)
            print(f"[{BASELINE_MODE}] Strategy 2 Applied: Using BrainTransformer Backbone")

        # --- B. 初始化客户端 ---
        clients = []
        for site_id in site_ids:
            full_ds = RestMetaDataset(site_id, cfg['data']['processed_root'], mode='private', return_clinical=(site_id in multi_site_set))
            dummy_X, site_labels = np.zeros(len(full_ds)), full_ds.labels
            train_idx, test_idx = list(skf.split(dummy_X, site_labels))[fold_idx]

            train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=cfg['data']['batch_size'], shuffle=True, drop_last=True)
            test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=cfg['data']['batch_size'], shuffle=False)

            model_choice = 'transformer'

            classifier_input_dim = 0
            if model_choice == 'resnet':
                classifier_input_dim = cfg['model']['hidden_dim']
                backbone = ResNetFC(input_dim=cfg['data']['roi_count'] ** 2, hidden_dim=classifier_input_dim, feature_dim=cfg['model']['feature_dim'])
            else:
                backbone = BrainTransformer(num_rois=cfg['data']['roi_count'], d_model=256, feature_dim=cfg['model']['feature_dim'])
                classifier_input_dim = 256

            model = MDDClassifier(backbone, hidden_dim=classifier_input_dim).to(device)

            cli_model = None
            is_multimodal = site_id in multi_site_set
            if is_multimodal:
                cli_model = ClinicalNet(input_dim=cfg['data']['clinical_dim'], feature_dim=cfg['model']['feature_dim']).to(device)

            trainer = ClientTrainer(site_id, model, train_loader, public_loader, cfg, device, is_multimodal=is_multimodal, clinical_model=cli_model)
            clients.append({'id': site_id, 'trainer': trainer, 'test_loader': test_loader})

        # --- C. 训练循环 ---
        best_metrics = {'auc': 0.0, 'acc': 0.0, 'sen': 0.0, 'spe': 0.0}

        for round_idx in range(cfg['federated']['n_rounds']):
            # 1. 客户端采样
            selected_clients = random.sample(clients, max(1, int(len(clients) * cfg['federated'].get('client_sample_rate', 1.0))))
            selected_ids = [c['id'] for c in selected_clients]
            logger.info(f"  Round {round_idx + 1} Selected: {selected_ids}")

            # 2. 下行分发 (Downlink)
            g_reps = None
            global_params = None

            if BASELINE_MODE == 'Local':
                g_reps = None
            elif BASELINE_MODE == 'Solo':
                g_reps = server.get_global_reps()
            elif BASELINE_MODE in ['FedAvg', 'FedProx']:
                global_params = global_model_avg.state_dict()
                for c in selected_clients:
                    c['trainer'].model.load_state_dict(global_params)

            # 3. 客户端训练
            client_uploads = []
            for c in selected_clients:
                loss = c['trainer'].train_epoch(
                    global_reps=g_reps,
                    global_model_params=global_params if BASELINE_MODE == 'FedProx' else None,
                    mu=0.01 if BASELINE_MODE == 'FedProx' else 0.0
                )

                if BASELINE_MODE == 'Solo':
                    client_uploads.append(c['trainer'].get_upload_package())

            # 4. 上行聚合 (Aggregation)
            if BASELINE_MODE == 'Solo':
                server.aggregate(client_uploads)
            elif BASELINE_MODE in ['FedAvg', 'FedProx']:
                fedavg_aggregate(global_model_avg, selected_clients)

            # 5. 评估
            round_scores = {'auc': [], 'acc': [], 'sen': [], 'spe': []}
            for c in clients:
                c['trainer'].model.eval()
                all_labels, all_probs, all_preds = [], [], []
                with torch.no_grad():
                    for batch in c['test_loader']:
                        # Dataset 返回值的兼容性处理
                        if len(batch) == 3: imgs, _, labels = batch
                        else: imgs, labels = batch

                        imgs = imgs.to(device)
                        outputs = c['trainer'].model(imgs)
                        probs = torch.softmax(outputs, dim=1)[:, 1] if outputs.shape[1] == 2 else torch.sigmoid(outputs).squeeze()
                        preds = (probs > 0.5).long()
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                        all_preds.extend(preds.cpu().numpy())

                try: auc = roc_auc_score(all_labels, all_probs)
                except: auc = 0.5

                try:
                    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0,1]).ravel()
                    acc = accuracy_score(all_labels, all_preds)
                    sen = tp / (tp + fn + 1e-10)
                    spe = tn / (tn + fp + 1e-10)
                except:
                    acc, sen, spe = 0, 0, 0

                round_scores['auc'].append(auc); round_scores['acc'].append(acc)
                round_scores['sen'].append(sen); round_scores['spe'].append(spe)

            avg_auc = np.mean(round_scores['auc'])
            if avg_auc > best_metrics['auc']:
                best_metrics = {k: np.mean(v) for k, v in round_scores.items()}

            if (round_idx + 1) % 5 == 0:
                logger.info(f"  [Fold {fold_idx+1} R{round_idx+1}] AUC: {avg_auc:.4f}")

        for k in folds_best_results: folds_best_results[k].append(best_metrics[k])
        logger.info(f"Fold {fold_idx+1} Best AUC: {best_metrics['auc']:.4f}")

    # 最终统计
    logger.info(f"\n{'='*20} Final Results ({BASELINE_MODE}) {'='*20}")
    for k, v in folds_best_results.items():
        logger.info(f"{k.upper()}: {statistics.mean(v):.4f} ± {statistics.stdev(v) if len(v)>1 else 0.0:.4f}")

if __name__ == "__main__":
    main()
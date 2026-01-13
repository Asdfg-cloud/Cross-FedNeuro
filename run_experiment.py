import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml
import logging
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

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
    # 存储每一折的最佳结果
    folds_best_auc = []

    for fold_idx in range(n_folds):
        logger.info(f"\n{'=' * 20} Start Fold {fold_idx + 1} / {n_folds} {'=' * 20}")

        # --- 步骤 A: 初始化异构服务端 (必须重置) ---
        # s_backbone = ResNetFC(input_dim=cfg['data']['roi_count']**2,
        #                       hidden_dim=cfg['model']['hidden_dim'],
        #                       feature_dim=cfg['model']['feature_dim']).to(device)
        # s_clinical = ClinicalNet(input_dim=cfg['data']['clinical_dim'],
        #                          feature_dim=cfg['model']['feature_dim']).to(device)
        # server = ServerTrainer(s_backbone, s_clinical, public_loader, cfg, device)
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
            # skf.split 需要 X 和 y。X 可以是任意等长数组，y 必须是标签
            dummy_X = np.zeros(len(full_ds))
            site_labels = full_ds.labels  # 假设 Dataset 里有 self.labels

            # list(generator) 会生成 [(train_idx, test_idx), ...] 共5个元组
            # 我们取第 fold_idx 个
            train_idx, test_idx = list(skf.split(dummy_X, site_labels))[fold_idx]

            # 创建 Subset
            train_subset = Subset(full_ds, train_idx)
            test_subset = Subset(full_ds, test_idx)

            # 创建 DataLoader
            # 只有训练集需要 shuffle=True
            train_loader = DataLoader(train_subset, batch_size=cfg['data']['batch_size'], shuffle=True, drop_last=True)
            test_loader = DataLoader(test_subset, batch_size=cfg['data']['batch_size'], shuffle=False)

            # backbone = ResNetFC(input_dim=cfg['data']['roi_count']**2,
            #                     hidden_dim=cfg['model']['hidden_dim'],
            #                     feature_dim=cfg['model']['feature_dim'])
            # --- 异构模型分配 ---
            # 随机给客户端分配不同的模型结构
            model_choice = random.choice(['resnet', 'transformer'])
            classifier_input_dim = 0
            backbone = None
            if model_choice == 'resnet':
                # 标准 ResNetFC
                classifier_input_dim = cfg['model']['hidden_dim']
                backbone = ResNetFC(input_dim=cfg['data']['roi_count'] ** 2,
                                    hidden_dim=classifier_input_dim,
                                    feature_dim=cfg['model']['feature_dim'])  # Proj Head Dim 统一
            elif model_choice == 'transformer':
                backbone = BrainTransformer(num_rois=cfg['data']['roi_count'],
                                            d_model=256,
                                            feature_dim=cfg['model']['feature_dim'])
                classifier_input_dim = 256  # 与 d_model 一致
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

        for round_idx in range(cfg['federated']['n_rounds']):
            # 1. 服务端分发全局特征 (Round 0 为 None)
            g_reps = server.get_global_reps()

            # 2. 客户端采样 (Client Sampling)
            sample_rate = cfg['federated'].get('client_sample_rate', 1.0)

            # 计算本轮选多少个客户端 (至少选1个)
            num_selected = max(1, int(len(clients) * sample_rate))

            # 随机抽取
            selected_clients = random.sample(clients, num_selected)

            # 打印日志看看选中了谁，防止出错
            selected_ids = [c['id'] for c in selected_clients]
            logger.info(f"  Round {round_idx + 1} Selected: {selected_ids}")

            # 3. 客户端训练 & 上传
            client_uploads = []
            for c in selected_clients:
                loss = c['trainer'].train_epoch(g_reps)

                # 获取上传包 (特征表示)
                pkg = c['trainer'].get_upload_package()
                client_uploads.append(pkg)

            # 4. 服务端聚合
            server.aggregate(client_uploads)

            # 5. 评估 (Evaluation) - 在本折的测试集上评估
            # 通常我们在所有客户端的测试集上评估，取平均 AUC
            auc_list = []

            for c in clients:
                trainer = c['trainer']
                loader = c['test_loader']  # 注意：这是本折划分出来的 20% 数据
                trainer.model.eval()

                all_labels, all_probs = [], []
                with torch.no_grad():
                    for batch_data in loader:
                        if (len(batch_data) == 3):
                            # 多模态客户端返回: (img, clinical, label)
                            imgs, _, labels = batch_data
                        else:
                            # 单模态客户端返回: (img, label)
                            imgs, labels = batch_data

                        imgs = imgs.to(device)
                        outputs = trainer.model(imgs)

                        # 简单的二分类概率获取
                        if outputs.shape[1] == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        else:
                            probs = torch.sigmoid(outputs).squeeze()

                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                # 计算 AUC (防止只有一类样本报错)
                try:
                    score = roc_auc_score(all_labels, all_probs)
                except ValueError:
                    score = 0.5  # 极端情况：测试集只有正例或只有负例
                auc_list.append(score)

            # 计算本轮平均 AUC
            round_avg_auc = np.mean(auc_list)
            if round_avg_auc > best_fold_auc:
                best_fold_auc = round_avg_auc

            if (round_idx + 1) % 5 == 0:
                logger.info(f"  [Fold {fold_idx + 1}] Round {round_idx + 1} Avg AUC: {round_avg_auc:.4f}")

        # 本折结束，记录最佳结果
        logger.info(f"Fold {fold_idx + 1} Best AUC: {best_fold_auc:.4f}")
        folds_best_auc.append(best_fold_auc)

    # ================= 实验结束，输出统计 =================
    import statistics

    if len(folds_best_auc) > 0:
        mean_auc = statistics.mean(folds_best_auc)
        # 如果只有一折，stdev 会报错，做个判断
        std_auc = statistics.stdev(folds_best_auc) if len(folds_best_auc) > 1 else 0.0
    else:
        mean_auc = 0.0
        std_auc = 0.0

    logger.info(f"\n{'=' * 20} 5-Fold Cross-Validation Results {'=' * 20}")
    logger.info(f"AUCs per fold: {folds_best_auc}")
    logger.info(f"Final Result: {mean_auc:.4f} ± {std_auc:.4f}")


if __name__ == "__main__":
    main()

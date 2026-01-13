import torch
import torch.nn as nn
import itertools
import numpy as np
from tqdm import tqdm

class ClientTrainer:
    def __init__(self, client_id, model, train_loader, public_loader, config, device,
                 is_multimodal=False, clinical_model=None, ablation_config=None):
        """
        :param model: 本地影像模型 (可以是异构的，但投影头输出维度需统一)
        :param clinical_model: 临床网络 (仅多模态客户端持有)
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.config = config
        self.device = device

        self.is_multimodal = is_multimodal
        self.clinical_model = clinical_model

        # 优化器配置：多模态客户端同时优化影像和临床网络
        if self.is_multimodal and self.clinical_model is not None:
            params = list(self.model.parameters()) + list(self.clinical_model.parameters())
        else:
            params = self.model.parameters()

        self.optimizer = torch.optim.Adam(params,
                                          lr=config['train']['lr'],
                                          weight_decay=config['train']['weight_decay'])
        self.criterion_ce = nn.CrossEntropyLoss()

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        # 默认开启所有 Loss
        self.ablation = ablation_config if ablation_config else {'use_intra': True, 'use_inter': True}

    def contrastive_loss(self, feat1, feat2):
        """ InfoNCE Loss """
        feat1 = torch.nn.functional.normalize(feat1, dim=1)
        feat2 = torch.nn.functional.normalize(feat2, dim=1)
        logits = torch.matmul(feat1, feat2.T) / self.config['train']['temperature']
        labels = torch.arange(logits.size(0)).to(self.device)
        return self.criterion_ce(logits, labels)

    def train_epoch(self, global_reps):
        """
        :param global_reps: (global_img_reps, global_cli_reps)
                            来自服务端聚合的公共数据特征共识。
                            Round 0 时可能为 None。
        """
        self.model.train()
        if self.is_multimodal:
            self.clinical_model.train()

        # 解析全局特征
        g_img_reps, g_cli_reps = None, None
        if global_reps is not None:
            g_img_reps, g_cli_reps = global_reps
            if g_img_reps is not None: g_img_reps = g_img_reps.to(self.device)
            if g_cli_reps is not None: g_cli_reps = g_cli_reps.to(self.device)

        if len(self.train_loader) == 0: return 0.0

        public_iter = itertools.cycle(self.public_loader)
        all_epochs_loss = []
        num_epochs = self.config['train'].get('local_epochs', 1)

        with tqdm(range(num_epochs), desc=f"Client {self.client_id} ({'Multi' if self.is_multimodal else 'Uni'})", leave=False, dynamic_ncols=True) as pbar:
            for epoch in pbar:
                batch_losses = []

                for batch_data in self.train_loader:
                    # --- 1. 数据解包 (适配混合模态) ---
                    if self.is_multimodal:
                        pvt_imgs, pvt_clin, pvt_labels = batch_data
                        pvt_clin = pvt_clin.to(self.device)
                    else:
                        pvt_imgs, pvt_labels = batch_data
                        pvt_clin = None

                    pvt_imgs, pvt_labels = pvt_imgs.to(self.device), pvt_labels.to(self.device)

                    # --- 2. Task Loss ---
                    preds = self.model(pvt_imgs)
                    loss_task = self.criterion_ce(preds, pvt_labels)

                    # --- 3. Local Multimodal Contrast (仅多模态) ---
                    loss_local_mm = 0.0
                    if self.is_multimodal:
                        local_img_feat = self.model.get_projection(pvt_imgs)
                        local_cli_feat = self.clinical_model(pvt_clin)
                        loss_local_mm = self.contrastive_loss(local_img_feat, local_cli_feat)

                    # --- 4. LCR Regularization (依赖全局特征) ---
                    loss_lcr = 0.0
                    # 只有当全局特征存在时才计算 LCR
                    if g_img_reps is not None:
                        # 获取公共数据
                        pub_imgs, _ = next(public_iter)
                        pub_imgs = pub_imgs.to(self.device)

                        # 确保 batch size 对齐
                        curr_bs = pub_imgs.size(0)

                        # 本地提取公共数据特征
                        pub_local_proj = self.model.get_projection(pub_imgs)

                        # Intra: 本地影像 <-> 全局影像共识
                        # 注意：这里假设 public_loader 是 shuffle=False 的，顺序一致
                        if self.ablation['use_intra']:
                            target_img = g_img_reps[:curr_bs]
                            loss_intra = self.contrastive_loss(pub_local_proj, target_img)
                            loss_lcr += loss_intra

                        # Inter: 本地影像 <-> 全局临床共识
                        # 单模态客户端在这里通过 global_cli_reps 间接学习临床知识
                        if self.ablation['use_inter'] and g_cli_reps is not None:
                            target_cli = g_cli_reps[:curr_bs]
                            loss_inter = self.contrastive_loss(pub_local_proj, target_cli)
                            loss_lcr += loss_inter

                    # --- Total Loss ---
                    gamma = self.config['train']['gamma']
                    loss = loss_task + gamma * (loss_lcr + loss_local_mm)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_losses.append(loss.item())

                if len(batch_losses) > 0:
                    epoch_avg_loss = sum(batch_losses) / len(batch_losses)
                    all_epochs_loss.append(epoch_avg_loss)
                    pbar.set_postfix({'loss': f'{epoch_avg_loss:.4f}'})

        self.scheduler.step()
        if len(all_epochs_loss) == 0: return 0.0
        return sum(all_epochs_loss) / len(all_epochs_loss)

    def get_upload_package(self):
        """
        【核心修改】
        不再上传模型参数，而是上传公共数据集在当前本地模型下的特征表示。
        这允许客户端使用完全不同的模型结构。
        """
        self.model.eval()
        if self.is_multimodal:
            self.clinical_model.eval()

        img_reps_list = []
        cli_reps_list = []

        with torch.no_grad():
            for batch in self.public_loader:
                # 兼容 Dataset 返回 2 个或 3 个值的情况
                if len(batch) == 3: imgs, clins, _ = batch
                else: imgs, clins = batch

                imgs = imgs.to(self.device)

                # 1. 提取影像特征
                img_proj = self.model.get_projection(imgs)
                img_reps_list.append(img_proj)

                # 2. 提取临床特征 (仅多模态)
                if self.is_multimodal:
                    clins = clins.to(self.device)
                    cli_proj = self.clinical_model(clins)
                    cli_reps_list.append(cli_proj)

        # 拼接并转至 CPU 以前往 Server
        pkg = {
            'n_samples': len(self.train_loader.dataset), # 权重依据
            'img_reps': torch.cat(img_reps_list, dim=0).cpu(),
            'cli_reps': torch.cat(cli_reps_list, dim=0).cpu() if self.is_multimodal else None
        }
        return pkg
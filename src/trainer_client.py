import torch
import torch.nn as nn
import itertools
import numpy as np
from tqdm import tqdm

class ClientTrainer:
    def __init__(self, client_id, model, train_loader, public_loader, config, device,
                 is_multimodal=False, clinical_model=None, ablation_config=None):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.public_loader = public_loader
        self.config = config
        self.device = device

        self.is_multimodal = is_multimodal
        self.clinical_model = clinical_model

        # 消融实验配置 (默认全开)
        self.ablation = ablation_config if ablation_config else {'use_intra': True, 'use_inter': True}

        if self.is_multimodal and self.clinical_model is not None:
            params = list(self.model.parameters()) + list(self.clinical_model.parameters())
        else:
            params = self.model.parameters()

        self.optimizer = torch.optim.Adam(params,
                                          lr=config['train']['lr'],
                                          weight_decay=config['train']['weight_decay'])
        self.criterion_ce = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['train'].get('step_size', 50), gamma=0.9)

    def contrastive_loss(self, feat1, feat2):
        """ [原始版本] 计算特征间的对比损失 (基于余弦相似度) """
        feat1 = torch.nn.functional.normalize(feat1, dim=1)
        feat2 = torch.nn.functional.normalize(feat2, dim=1)
        logits = torch.matmul(feat1, feat2.T) / self.config['train']['temperature']
        labels = torch.arange(logits.size(0)).to(self.device)
        return self.criterion_ce(logits, labels)

    def train_epoch(self, global_reps=None, global_model_params=None, mu=0.0):
        """
        :param global_reps: (img_reps, cli_reps) 原始的特征元组
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

        with tqdm(range(num_epochs), desc=f"Client {self.client_id} ({'Multi' if self.is_multimodal else 'Uni'})", leave=True, dynamic_ncols=True) as pbar:
            for epoch in pbar:
                batch_losses = []
                for batch_data in self.train_loader:
                    # 数据解包
                    if self.is_multimodal:
                        pvt_imgs, pvt_clin, pvt_labels = batch_data
                        pvt_clin = pvt_clin.to(self.device)
                    else:
                        pvt_imgs, pvt_labels = batch_data
                        pvt_clin = None

                    pvt_imgs, pvt_labels = pvt_imgs.to(self.device), pvt_labels.to(self.device)

                    # 1. Task Loss
                    preds = self.model(pvt_imgs)
                    loss_task = self.criterion_ce(preds, pvt_labels)
                    loss = loss_task

                    # 2. FedProx Loss (仅当传入 global_model_params 时计算)
                    if global_model_params is not None and mu > 0:
                        loss_prox = 0.0
                        for name, param in self.model.named_parameters():
                            if name in global_model_params:
                                target = global_model_params[name].to(self.device)
                                loss_prox += torch.norm(param - target) ** 2
                        loss += (mu / 2) * loss_prox

                    # 3. H2-FedNeuro Losses (原始版本核心逻辑)
                    if g_img_reps is not None:
                        loss_local_mm = 0.0
                        # 模态内对比 (Intra-client Multimodal Contrastive)
                        if self.is_multimodal and self.is_multimodal:
                            local_img_feat = self.model.get_projection(pvt_imgs)
                            local_cli_feat = self.clinical_model(pvt_clin)
                            loss_local_mm = self.contrastive_loss(local_img_feat, local_cli_feat)

                        # 全局-本地正则化 (LCR Loss)
                        loss_lcr = 0.0
                        # 获取公共数据 (2返回值: img, cli)
                        pub_imgs, _ = next(public_iter)
                        pub_imgs = pub_imgs.to(self.device)

                        curr_bs = pub_imgs.size(0)
                        target_img = g_img_reps[:curr_bs]
                        pub_local_proj = self.model.get_projection(pub_imgs)

                        if self.ablation['use_intra']:
                            loss_lcr += self.contrastive_loss(pub_local_proj, target_img)

                        if self.ablation['use_inter'] and g_cli_reps is not None:
                            target_cli = g_cli_reps[:curr_bs]
                            loss_lcr += self.contrastive_loss(pub_local_proj, target_cli)

                        gamma = self.config['train']['gamma']
                        loss += gamma * (loss_lcr + loss_local_mm)

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
        """ 上传公共数据上的特征投影 (Representations) """
        self.model.eval()
        if self.is_multimodal: self.clinical_model.eval()
        img_reps, cli_reps = [], []
        with torch.no_grad():
            for batch in self.public_loader:
                if len(batch) == 3: imgs, clins, _ = batch
                else: imgs, clins = batch # dataset 返回的是 img, cli

                imgs = imgs.to(self.device)
                img_reps.append(self.model.get_projection(imgs))
                if self.is_multimodal:
                    clins = clins.to(self.device)
                    cli_reps.append(self.clinical_model(clins))
        return {
            'n_samples': len(self.train_loader.dataset),
            'img_reps': torch.cat(img_reps, dim=0).cpu(),
            'cli_reps': torch.cat(cli_reps, dim=0).cpu() if self.is_multimodal else None
        }
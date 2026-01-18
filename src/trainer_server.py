import torch
import torch.nn.functional as F
from src.utils import get_logger

class ServerTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logger = get_logger('Cross-FedNeuro', config['experiment']['save_dir'])

        # 存储全局共识特征 (Global Consensus Representations)
        self.global_img_reps = None
        self.global_cli_reps = None

    def _gca_aggregate(self, local_reps_list, global_anchor_reps):
        """
        核心 GCA 聚合算法 (对应论文 Eq 4, 5, 6)

        :param local_reps_list: List[Tensor], 每个 Tensor 形状为 [N_pub, Dim]
        :param global_anchor_reps: Tensor [N_pub, Dim] 或 None (来自上一轮的另一模态特征)
        :return: Aggregated Tensor [N_pub, Dim]
        """
        # 1. 堆叠所有客户端的特征 -> [Client, N, Dim]
        stacked_reps = torch.stack(local_reps_list).to(self.device)

        # --- 冷启动/降级处理 ---
        # 如果没有锚点 (第一轮) 或者锚点数量不匹配 (极其罕见), 回退到平均聚合
        if global_anchor_reps is None:
            return torch.mean(stacked_reps, dim=0)

        global_anchor_reps = global_anchor_reps.to(self.device)
        temp = self.config['train'].get('temperature', 0.07)

        # 2. 归一化特征 (Cosine Similarity 前置步骤)
        # [Client, N, Dim]
        normalized_local = F.normalize(stacked_reps, dim=2)
        # [N, Dim]
        normalized_anchor = F.normalize(global_anchor_reps, dim=1)

        # 3. 计算对比分数 (Eq 4)
        # score = pos_sim - log(sum(exp(neg_sim)))

        scores_list = []
        n_clients = stacked_reps.size(0)
        n_samples = stacked_reps.size(1)

        # 逐个客户端计算以节省显存 (也可以完全向量化，但这取决于 N_pub 大小)
        for i in range(n_clients):
            loc_rep = normalized_local[i]  # [N, Dim]

            # A. 分子项 (Positive Pair): 对角线相似度
            # sim(i_local^k, t_global^k)
            pos_sim = (loc_rep * normalized_anchor).sum(dim=1) / temp  # [N]

            # B. 分母项 (Negative Pairs): 与所有非配对 anchor 的相似度
            # sim(i_local^k, t_global^j) for all j
            # Matrix multiplication: [N, Dim] @ [Dim, N] -> [N, N]
            all_sim_matrix = torch.matmul(loc_rep, normalized_anchor.T) / temp

            # Mask out diagonal (self-pair is positive, others are negatives)
            # Eq 4 的分母是 sum_{j != k} exp(...)
            mask = torch.eye(n_samples, device=self.device).bool()
            exp_all = torch.exp(all_sim_matrix)

            # 减去对角线上的项 (即正样本)
            sum_neg_exp = exp_all.sum(dim=1) - torch.exp(pos_sim)

            # 加上 epsilon 防止 log(0)
            sum_neg_exp = torch.clamp(sum_neg_exp, min=1e-9)

            # C. 最终分数
            # s^(k,c) = pos_term - log(neg_term)
            score = pos_sim - torch.log(sum_neg_exp)
            scores_list.append(score)

        # 堆叠分数 -> [Client, N]
        scores_tensor = torch.stack(scores_list)

        # 4. 计算权重 (Eq 5)
        # 对客户端维度 (dim=0) 做 Softmax
        # alpha^(k,c)
        weights = F.softmax(scores_tensor, dim=0)

        # 5. 加权聚合 (Eq 6)
        # weights: [Client, N] -> unsqueeze -> [Client, N, 1]
        # stacked_reps: [Client, N, Dim]
        weighted_reps = (stacked_reps * weights.unsqueeze(-1)).sum(dim=0)

        return weighted_reps

    def aggregate(self, client_uploads):
        """
        基于 GCA 的特征聚合
        """
        if len(client_uploads) == 0:
            return

        # 准备数据容器
        img_reps_list = []
        cli_reps_list = []

        # 提取数据
        for pkg in client_uploads:
            # 确保数据在 CPU 或指定 Device，这里建议统一处理
            img_reps_list.append(pkg['img_reps'])

            if pkg['cli_reps'] is not None:
                cli_reps_list.append(pkg['cli_reps'])

        # --- A. 聚合影像特征 (Image Aggregation) ---
        # 使用上一轮的 "Global Clinical" 作为 Anchor 来筛选高质量的 "Local Image"
        if len(img_reps_list) > 0:
            new_global_img = self._gca_aggregate(
                img_reps_list,
                self.global_cli_reps  # Cross-Modal Anchor
            )
            # 动量更新 (可选，防止剧烈波动，Paper中是直接替换，这里可以保持直接替换)
            self.global_img_reps = new_global_img.cpu()

        # --- B. 聚合临床特征 (Clinical Aggregation) ---
        # 使用上一轮的 "Global Image" 作为 Anchor 来筛选高质量的 "Local Clinical"
        # 仅多模态客户端参与
        if len(cli_reps_list) > 0:
            new_global_cli = self._gca_aggregate(
                cli_reps_list,
                self.global_img_reps # Cross-Modal Anchor
            )
            self.global_cli_reps = new_global_cli.cpu()
            self.logger.info(f"GCA Aggregated Clinical Reps from {len(cli_reps_list)} clients.")
        else:
            self.logger.warning("No Multimodal clients this round! Keeping previous Clinical Reps.")

    def get_global_reps(self):
        """ 分发全局特征给客户端 """
        return self.global_img_reps, self.global_cli_reps
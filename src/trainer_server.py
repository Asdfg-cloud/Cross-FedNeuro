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

    def aggregate(self, client_uploads):
        """
        基于特征的聚合 (Prototype Aggregation)

        :param client_uploads: list of dicts, 每个包含:
               - 'n_samples': int
               - 'img_reps': Tensor [N_pub, Dim]
               - 'cli_reps': Tensor [N_pub, Dim] or None
        """
        if len(client_uploads) == 0:
            return

        # 1. 获取基础维度信息
        sample_pkg = client_uploads[0]
        n_pub = sample_pkg['img_reps'].size(0)
        dim = sample_pkg['img_reps'].size(1)

        # 初始化聚合容器 (CPU上计算)
        agg_img = torch.zeros(n_pub, dim)
        agg_cli = torch.zeros(n_pub, dim)

        total_img_weight = 0
        total_cli_weight = 0

        # 2. 遍历聚合
        for pkg in client_uploads:
            n = pkg['n_samples'] # 权重

            # --- A. 聚合影像特征 (所有人都贡献) ---
            agg_img += pkg['img_reps'] * n
            total_img_weight += n

            # --- B. 聚合临床特征 (只有多模态客户端贡献) ---
            if pkg['cli_reps'] is not None:
                agg_cli += pkg['cli_reps'] * n
                total_cli_weight += n

        # 3. 计算加权平均
        if total_img_weight > 0:
            self.global_img_reps = agg_img / total_img_weight

        if total_cli_weight > 0:
            self.global_cli_reps = agg_cli / total_cli_weight
            self.logger.info(f"Aggregated Clinical Reps from {total_cli_weight} weighted samples.")
        else:
            # 极端情况：本轮没有多模态客户端被选中
            # 策略：保持上一轮的临床特征 (如果存在)，否则为 None
            self.logger.warning("No Multimodal clients this round! Keeping previous Clinical Reps.")

    def get_global_reps(self):
        """ 分发全局特征给客户端 """
        return self.global_img_reps, self.global_cli_reps
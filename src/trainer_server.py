import torch
import torch.nn.functional as F
from src.utils import get_logger

class ServerTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logger = get_logger('Cross-FedNeuro', config['experiment']['save_dir'])

        # 存储全局共识特征
        self.global_img_reps = None
        self.global_cli_reps = None

    def aggregate(self, client_uploads):
        """
        基于质量评分(Accuracy)的加权聚合
        """
        if len(client_uploads) == 0:
            return

        sample_pkg = client_uploads[0]
        n_pub = sample_pkg['img_reps'].size(0)
        dim = sample_pkg['img_reps'].size(1)

        # --- 步骤 A: 计算每个客户端的权重 ---
        # 提取所有客户端的 public_acc
        accuracies = [pkg['public_acc'] for pkg in client_uploads]
        accuracies_tensor = torch.tensor(accuracies, dtype=torch.float32)

        # [关键策略] 使用 Softmax 放大差异
        # temperature 控制区分度：越小(如0.1)，好模型权重越大；越大(如1.0)，越接近平均
        temperature = 0.5
        weights = torch.softmax(accuracies_tensor / temperature, dim=0)

        # 记录日志，观察权重分配情况
        self.logger.info(f"Aggregation Weights (Acc based): {weights.tolist()}")

        # --- 步骤 B: 加权聚合 ---
        agg_img = torch.zeros(n_pub, dim)
        agg_cli = torch.zeros(n_pub, dim)

        total_cli_weight = 0.0 # 临床模态可能只有部分客户端有，需要单独归一化

        for idx, pkg in enumerate(client_uploads):
            w = weights[idx].item() # 获取归一化后的权重

            # 聚合影像特征
            # 注意：这里不进行 L2 Normalize，保留模长蕴含的置信度信息
            agg_img += pkg['img_reps'] * w

            # 聚合临床特征 (如果存在)
            if pkg['cli_reps'] is not None:
                agg_cli += pkg['cli_reps'] * w
                total_cli_weight += w

        # --- 步骤 C: 更新全局状态 ---
        # 影像特征权重之和已经是 1.0 (Softmax保证)，直接赋值
        self.global_img_reps = agg_img

        # 临床特征需要除以实际参与的权重之和 (因为部分客户端可能没有临床数据)
        if total_cli_weight > 0:
            self.global_cli_reps = agg_cli / total_cli_weight
        else:
            self.logger.warning("No Multimodal clients this round!")

    def get_global_reps(self):
        return self.global_img_reps, self.global_cli_reps
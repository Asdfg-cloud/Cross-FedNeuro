import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class ResNetFC(nn.Module):
    """ 处理 FC 矩阵的 ResNet-MLP 架构 """
    def __init__(self, input_dim, hidden_dim=1024, feature_dim=256, num_blocks=2):
        super().__init__()
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Residual Backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Representation Head (Output for Aggregation)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim) # Proj to common space
        )

    def forward(self, x, return_proj=False):
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)

        # Project to contrastive space
        proj = self.projection_head(out)

        if return_proj:
            return proj
        return out # Return features for classifier

class BrainTransformer(nn.Module):
    """
    基于 Transformer 的 fMRI 脑网络分析模型
    输入: (Batch, N*N) -> 内部 reshape 为 (Batch, N, N)
    将每个 ROI 的连接图谱视为一个 Token
    """
    def __init__(self, num_rois=116, d_model=256, nhead=8, num_layers=2, feature_dim=256):
        super().__init__()
        self.num_rois = num_rois

        # 1. Embedding Layer: 将每个 ROI 的原始连接特征 (dim=116) 映射到高维 (d_model)
        self.embedding = nn.Sequential(
            nn.Linear(num_rois, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 2. Positional Encoding (可学习的位置编码，虽然 FC 是无向图，但加上有时有助于区分 ROI)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_rois, d_model))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Projection Head (用于对比学习和分类)
        # 我们使用 CLS token 或者是 Mean Pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feature_dim)
        )

    def forward(self, x, return_proj=False):
        # x shape: (Batch, 116*116) -> Flattened
        batch_size = x.size(0)

        # Reshape back to (Batch, ROI, ROI)
        # 每个 ROI 作为一个 Token，其特征是它与其他所有 ROI 的连接强度
        x = x.view(batch_size, self.num_rois, self.num_rois)

        # Embedding
        x = self.embedding(x) # (Batch, 116, d_model)

        # Add CLS Token (类似 BERT，用于汇总全局信息)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (Batch, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1) # (Batch, 117, d_model)

        # Add Positional Embedding
        # 注意位置编码需要扩展维度匹配
        x = x + torch.cat((torch.zeros(batch_size, 1, x.shape[2], device=x.device),
                           self.pos_embedding.expand(batch_size, -1, -1)), dim=1)

        # Transformer Forward
        out = self.transformer(x) # (Batch, 117, d_model)

        # 取出 CLS token 对应的输出作为全局特征
        global_feat = out[:, 0, :] # (Batch, d_model)

        # Projection
        proj = self.projection_head(global_feat)

        if return_proj:
            return proj
        return global_feat

class MDDClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Extract features (before projection head)
        features = self.backbone(x, return_proj=False)
        return self.fc(features)

    def get_projection(self, x):
        return self.backbone(x, return_proj=True)

class ClinicalNet(nn.Module):
    """ 处理临床数据的简单网络 """
    def __init__(self, input_dim, hidden_dim=64, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim) # Align with Image Dim
        )
    def forward(self, x):
        return self.net(x)

class SimpleCNN(torch.nn.Module):
    """ 简单的 CNN 骨干 """
    def __init__(self, input_dim, feature_dim=256):
        super().__init__()
        # 假设输入已经被 flatten，我们先 reshape 回去 (仅用于演示异构逻辑)
        # 这里简单起见，还是处理 FC，但是层数和结构不同
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )
        self.proj = torch.nn.Linear(256, feature_dim) # 必须统一输出维度

    def forward(self, x, return_proj=False):
        feat = self.net(x)
        proj = self.proj(feat)
        if return_proj:
            return proj
        return feat
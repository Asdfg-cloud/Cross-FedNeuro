# src/__init__.py

# 从各个子模块导入关键类
from .dataset import RestMetaDataset
from .models import ResNetFC, MDDClassifier, ClinicalNet, BrainTransformer
from .trainer_client import ClientTrainer
from .trainer_server import ServerTrainer
from .utils import seed_everything, get_logger, calculate_metrics


__all__ = [
    'RestMetaDataset',
    'ResNetFC',
    'MDDClassifier',
    'ClinicalNet',
    'BrainTransformer',
    'ClientTrainer',
    'ServerTrainer',
    'seed_everything',
    'get_logger',
    'calculate_metrics'
]
import torch
import numpy as np
import random
import os
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred, y_prob):
    """
    计算医学图像分析所需的完整指标
    y_true: 真实标签 (0/1)
    y_pred: 预测类别 (0/1)
    y_prob: 预测概率 (for AUC)
    """
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5  # 处理只有一个类别的极端情况

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8) # Recall
    specificity = tn / (tn + fp + 1e-8)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "auc": auc,
        "f1": f1_score(y_true, y_pred),
        "sens": sensitivity,
        "spec": specificity
    }

def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

        # File Handler
        fh = logging.FileHandler(os.path.join(log_dir, 'experiment.log'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Stream Handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger
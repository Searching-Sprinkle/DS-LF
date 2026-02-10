#损失函数方案一
import torch
import torch.nn as nn
import torchvision.models as models

def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_mse_loss_raw(loss, x, y, k = 1):
    """
    Calculate mse loss for raw.
    """
    # Compute loss for raw
    loss_mse_raw = torch.mean((x-y)**2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss

def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss

# --- 新增：感知损失模块 ---
class PerceptualLoss(nn.Module):
    """
    感知损失模块，基于 VGG16 的特征图。
    """
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        # 加载预训练的 VGG16，并截取部分层
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        # 冻结 VGG 的参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 定义要提取特征的层索引
        self.layers = [3, 8, 15, 22] # 对应 'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'
        self.vgg_layers = nn.ModuleList([vgg[i] for i in range(max(self.layers)+1)])
        self.device = device

    def forward(self, pred, target):
        """
        计算感知损失。
        Args:
            pred (Tensor): 预测的投影图, [B, C, H, W]
            target (Tensor): 真实的投影图, [B, C, H, W]
        Returns:
            loss (Tensor): 感知损失标量
        """
        # VGG 需要 3 通道输入，将单通道复制成 3 通道
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0.0
        x_pred, x_target = pred, target
        
        for i, layer in enumerate(self.vgg_layers):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.layers:
                # 使用 L1 损失，通常比 L2 更稳定
                loss += torch.mean(torch.abs(x_pred - x_target))
        
        return loss

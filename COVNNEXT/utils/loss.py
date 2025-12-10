import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.modules import *
from torchvision.transforms.functional import rgb_to_grayscale
import imageio

'''https://github.com/ntcongvn/CCBANet/blob/main/libraries/CCBANet/utils/loss.py'''

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pred_prob, target):
        size = pred_prob.size(0)
        pred_flat = pred_prob.view(size, -1)
        target_flat = target.view(size, -1)
        return F.binary_cross_entropy(pred_flat, target_flat, reduction='mean')


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #pred = F.sigmoid(pred)       
        
        #flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, pred_prob, target):
        # pred_prob: sigmoid uygulanmış olmalı (sen zaten öyle yapıyorsun)
        bce = F.binary_cross_entropy(pred_prob, target, reduction='mean')
        dice = self.dice(pred_prob, target)
        return bce + dice




"""BCE + IoU Loss"""


class BceIoULoss(nn.Module):
    def __init__(self):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss()
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss = self.iou(pred, target)

        loss = iouloss + bceloss

        return loss

""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """
class StructureLoss_old(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()


# StructureLoss sınıfının DÜZELTİLMİŞ hali
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        # Ağırlık haritası (değişiklik yok)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

        # Ağırlıklı BCE kısmı (değişiklik yok, hala logit kullanıyor)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # --- DÜZELTME BURADA ---
        # IoU hesaplamasından önce logit'i olasılığa çevir
        pred_prob = torch.sigmoid(pred)

        # IoU hesaplamasında artık olasılık değerini ('pred_prob') kullan
        inter = ((pred_prob * mask) * weit).sum(dim=(2, 3))
        union = ((pred_prob + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)  # paydadaki -inter olmalı

        return (wbce + wiou).mean()

""" Deep Supervision Loss"""

class DeepSupervisionLoss(nn.Module):
    def __init__(self, typeloss="BceDiceLoss"):
        super(DeepSupervisionLoss, self).__init__()

        if typeloss=="BceDiceLoss":
            self.criterion = BceDiceLoss()
        elif typeloss=="BceIoULoss":
            self.criterion = BceIoULoss()
        elif typeloss=="StructureLoss":
            self.criterion = StructureLoss()
        else:
            raise Exception("Loss name is unvalid.")

    def forward(self, pred, gt):
        d0, d1, d2, d3, d4= pred[0:]
        loss0 = self.criterion(torch.sigmoid(d0), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss1 = self.criterion(torch.sigmoid(d1), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss2 = self.criterion(torch.sigmoid(d2), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss3 = self.criterion(torch.sigmoid(d3), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss4 = self.criterion(torch.sigmoid(d4), gt)

        return loss0 + loss1 + loss2 + loss3 + loss4


class MultiTaskDeepSupervisionLoss(nn.Module):
    """
    Hem çok seviyeli maske hem de çok seviyeli sınır kaybını hesaplayan
    ve logit/olasılık ayrımını doğru yapan çok görevli bir kayıp fonksiyonu.
    """

    def __init__(self,
                 mask_loss_type="StructureLoss",
                 boundary_loss_type="BceDiceLoss",
                 boundary_weight=0.8,
                 mask_ds_weights=None,        # NEW
                 boundary_ds_weights=None):   # NEW
        super(MultiTaskDeepSupervisionLoss, self).__init__()

        self.boundary_weight = boundary_weight

        # default deep supervision weights (yüksek çözünürlüğe daha çok ağırlık)
        self.mask_ds_weights = mask_ds_weights or [1.0, 0.5, 0.25, 0.125, 0.125]
        self.boundary_ds_weights = boundary_ds_weights or [1.0, 0.5, 0.25, 0.125, 0.125]

        # Maske kaybı için kriteri seç
        if mask_loss_type == "StructureLoss":
            self.criterion_mask = StructureLoss()
        elif mask_loss_type == "BceDiceLoss":
            self.criterion_mask = BceDiceLoss()
        else:
            raise NotImplementedError

        # Sınır kaybı için kriteri seç
        if boundary_loss_type == "BceDiceLoss":
            self.criterion_boundary = BceDiceLoss()
        else:
            raise NotImplementedError

    def forward(self, pred_masks, pred_boundaries, gt_mask, gt_boundary):
        assert len(pred_masks) == len(self.mask_ds_weights), \
            f"mask scales={len(pred_masks)} ile mask_ds_weights={len(self.mask_ds_weights)} uyuşmuyor"
        assert len(pred_boundaries) == len(self.boundary_ds_weights), \
            f"boundary scales={len(pred_boundaries)} ile boundary_ds_weights={len(self.boundary_ds_weights)} uyuşmuyor"

        total_mask_loss = 0.0
        total_boundary_loss = 0.0

        # --- Maske kaybı (logit → StructureLoss, prob → BceDice) ---
        temp_gt_mask = gt_mask
        for w, pred_mask in zip(self.mask_ds_weights, pred_masks):
            if isinstance(self.criterion_mask, StructureLoss):
                loss_m = self.criterion_mask(pred_mask, temp_gt_mask)
            else:
                loss_m = self.criterion_mask(torch.sigmoid(pred_mask), temp_gt_mask)
            total_mask_loss += w * loss_m
            # Downsample GT (NEAREST)
            temp_gt_mask = F.interpolate(temp_gt_mask, scale_factor=0.5, mode='nearest')

        # --- Boundary kaybı (prob bekler) ---
        for w, pred_boundary in zip(self.boundary_ds_weights, pred_boundaries):
            gt_boundary_resized = F.interpolate(gt_boundary, size=pred_boundary.shape[2:], mode='nearest')
            loss_b = self.criterion_boundary(torch.sigmoid(pred_boundary), gt_boundary_resized)
            total_boundary_loss += w * loss_b

        return total_mask_loss + self.boundary_weight * total_boundary_loss

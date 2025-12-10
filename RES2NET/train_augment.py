import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import argparse
from lib.EGANet import EGANetModel
from utils.dataloader_augment import get_loader, ValidationDataset
from utils.loss import MultiTaskDeepSupervisionLoss
from utils.utils import AvgMeter, clip_gradient
from utils.scheduler import WarmupCosineScheduler
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


def calculate_metrics_with_threshold(pred, gt, threshold=0.5):
    """
    Belirli threshold ile segmentasyon metriklerini hesaplar
    """
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    # Dice Score
    intersection = (pred_binary * gt_binary).sum()
    dice = (2.0 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)

    # IoU Score
    union = pred_binary.sum() + gt_binary.sum() - intersection
    iou = intersection / (union + 1e-8)

    # MAE
    mae = torch.mean(torch.abs(gt_binary - pred))

    return dice.item(), iou.item(), mae.item()


def threshold_sweep(pred_probs, gts, thresholds=None):
    """
    Threshold sweeping - tüm threshold'larda Dice hesaplar
    pred_probs: (N, 1, H, W) - sigmoid uygulanmış tahminler
    gts: (N, 1, H, W) - ground truth masks
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)  # Hız için 0.05 adım

    threshold_results = []

    for threshold in thresholds:
        dice_scores = []
        iou_scores = []
        mae_scores = []

        # Tüm örnekler için bu threshold'da metrik hesapla
        for i in range(pred_probs.shape[0]):
            dice, iou, mae = calculate_metrics_with_threshold(pred_probs[i], gts[i], threshold)
            dice_scores.append(dice)
            iou_scores.append(iou)
            mae_scores.append(mae)

        # Bu threshold için ortalama metrikler
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_mae = np.mean(mae_scores)

        threshold_results.append({
            'threshold': threshold,
            'dice': avg_dice,
            'iou': avg_iou,
            'mae': avg_mae
        })

    # En iyi Dice score'u ve threshold'u bul
    best_result = max(threshold_results, key=lambda x: x['dice'])
    best_threshold = best_result['threshold']
    best_dice = best_result['dice']

    return best_dice, best_threshold, threshold_results


def validate_model(model, val_dataset, device, epoch, writer=None):
    """
    Model validation fonksiyonu - threshold sweeping ile en iyi threshold bulur
    """
    model.eval()

    dataset_loaders = val_dataset.get_dataset_loaders(batchsize=4, num_workers=2)

    all_results = {}
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for dataset_name, loader in dataset_loaders.items():
            print(f"\n--- Validating on {dataset_name} ---")

            dataset_predictions = []
            dataset_gts = []

            for batch_idx, (images, gts, gts_boundary) in enumerate(loader):
                images = images.to(device)
                gts = gts.to(device)

                # Model prediction
                pred_masks, pred_boundaries = model(images)
                pred = pred_masks[0]  # En yüksek çözünürlüklü tahmin

                # Sigmoid uygula
                pred = torch.sigmoid(pred)

                # CPU'ya taşı ve topla
                dataset_predictions.append(pred.cpu())
                dataset_gts.append(gts.cpu())

            # Dataset tahminlerini birleştir
            dataset_pred_tensor = torch.cat(dataset_predictions, dim=0)
            dataset_gt_tensor = torch.cat(dataset_gts, dim=0)

            # Threshold sweeping yapıp en iyi threshold bul
            print(f"Running threshold sweep for {dataset_name}...")
            best_dice, best_threshold, threshold_curve = threshold_sweep(dataset_pred_tensor, dataset_gt_tensor)

            # 0.5 threshold ile de karşılaştırmalı metrik hesapla
            # 0.5 threshold ile örnek-bazlı metrik hesapla (batch mean değil!)
            dice_05s, iou_05s, mae_05s = [], [], []
            for i in range(dataset_pred_tensor.shape[0]):
                d, j, m = calculate_metrics_with_threshold(dataset_pred_tensor[i], dataset_gt_tensor[i], 0.5)
                dice_05s.append(d); iou_05s.append(j); mae_05s.append(m)
            dice_05, iou_05, mae_05 = np.mean(dice_05s), np.mean(iou_05s), np.mean(mae_05s)

            all_results[dataset_name] = {
                'best_dice': best_dice,
                'best_threshold': best_threshold,
                'dice_05': dice_05,
                'iou_05': iou_05,
                'mae_05': mae_05,
                'threshold_curve': threshold_curve,
                'samples': dataset_pred_tensor.shape[0]
            }

            print(f"{dataset_name} Results:")
            print(f"  Best - Dice: {best_dice:.4f} @ threshold: {best_threshold:.2f}")
            print(f"  @0.5 - Dice: {dice_05:.4f}, IoU: {iou_05:.4f}, MAE: {mae_05:.4f}")
            print(f"  Samples: {dataset_pred_tensor.shape[0]}")

            # TensorBoard logging
            if writer:
                writer.add_scalar(f"Validation/{dataset_name}/Best_Dice", best_dice, epoch)
                writer.add_scalar(f"Validation/{dataset_name}/Best_Threshold", best_threshold, epoch)
                writer.add_scalar(f"Validation/{dataset_name}/Dice_05", dice_05, epoch)
                writer.add_scalar(f"Validation/{dataset_name}/IoU_05", iou_05, epoch)
                writer.add_scalar(f"Validation/{dataset_name}/MAE_05", mae_05, epoch)

            # Global predictions için topla
            all_predictions.append(dataset_pred_tensor)
            all_ground_truths.append(dataset_gt_tensor)

    # Tüm dataset'leri birleştirip global threshold sweep
    if all_predictions:
        global_predictions = torch.cat(all_predictions, dim=0)
        global_gts = torch.cat(all_ground_truths, dim=0)

        print(f"\n--- Global Threshold Sweep ---")
        global_best_dice, global_best_threshold, global_curve = threshold_sweep(global_predictions, global_gts)

        # Global 0.5 threshold metrikleri
        gd05, gi05, gm05 = [], [], []
        for i in range(global_predictions.shape[0]):
            d, j, m = calculate_metrics_with_threshold(global_predictions[i], global_gts[i], 0.5)
            gd05.append(d); gi05.append(j); gm05.append(m)
        global_dice_05, global_iou_05, global_mae_05 = np.mean(gd05), np.mean(gi05), np.mean(gm05)

        print(f"=== OVERALL VALIDATION ===")
        print(f"Best Global - Dice: {global_best_dice:.4f} @ threshold: {global_best_threshold:.2f}")
        print(f"@0.5 Global - Dice: {global_dice_05:.4f}, IoU: {global_iou_05:.4f}, MAE: {global_mae_05:.4f}")
        print(f"Total samples: {global_predictions.shape[0]}")

        if writer:
            writer.add_scalar("Validation/Global/Best_Dice", global_best_dice, epoch)
            writer.add_scalar("Validation/Global/Best_Threshold", global_best_threshold, epoch)
            writer.add_scalar("Validation/Global/Dice_05", global_dice_05, epoch)
            writer.add_scalar("Validation/Global/IoU_05", global_iou_05, epoch)
            writer.add_scalar("Validation/Global/MAE_05", global_mae_05, epoch)

        # İki kez aynı değer döndürülüyordu, düzeltildi
        # Eski: return global_best_dice, global_best_threshold, global_best_dice, all_results
        return global_best_dice, global_best_threshold, global_iou_05, all_results

    return 0.0, 0.5, 0.0, all_results


def train(train_loader, model, optimizer, epoch, criteria_loss, total_step, opt, writer):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    epoch_loss = 0.0

    # Global step hesaplaması için
    global_step = epoch * total_step

    pbar = tqdm(enumerate(train_loader, start=1), total=total_step,
                desc=f"Epoch {epoch+1}/{opt.epoch}", ncols=120, leave=True)
    for i, pack in pbar:
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, gts_boundary = pack
            images = Variable(images).to(device)
            gts = Variable(gts).to(device)
            gts_boundary = Variable(gts_boundary).to(device)

            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # NEAREST interpolasyon mask ve boundary için
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')
                gts_boundary = F.interpolate(gts_boundary, size=(trainsize, trainsize), mode='nearest')

            pred_masks, pred_boundaries = model(images)
            loss = criteria_loss(pred_masks, pred_boundaries, gts, gts_boundary)

            # TensorBoard step düzeltmesi: yalnızca rate==1 iken log at
            if rate == 1:
                writer.add_scalar("Loss/train", loss.item(), global_step + i)

            loss.backward()
            clip_gradient(optimizer, opt.grad_norm)
            optimizer.step()

            if rate == 1:
                loss_record.update(loss.item(), opt.batchsize)
                epoch_loss += loss.item()

        # tqdm postfix'e anlık ortalama loss'u yaz
        try:
            pbar.set_postfix({"loss": f"{loss_record.show():.4f}"})
        except Exception:
            pass

    return loss_record.show()

if __name__ == '__main__':
    writer = SummaryWriter()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--workers', type=int, default=4, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='gradient clipping norm')
    parser.add_argument('--batchsize', type=int,default=12, help='training batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--power', type=float, default=0.9)
    parser.add_argument('--trainsize', type=int,default=352, help='training dataset size')
    parser.add_argument('--train_path', type=str,default='./dataset/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,default='bb-Res2Net')
    parser.add_argument('--resume', type=int, default=False, help='Resume training from checkpoint')
    parser.add_argument('--resume_path', type=str, default='checkpoints/bb/bb-Res2Net/bb.pth', help='Path to the checkpoint to resume from')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequency (epochs)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs for scheduler')

    opt = parser.parse_args()

    model = EGANetModel()
    model.to(device)
    
    # Backbone ve Head modüllerini ayır - farklı LR için
    backbone_modules = [
        model.net.encoder1_conv, model.net.encoder1_bn, model.net.encoder1_relu,
        model.net.encoder2, model.net.encoder3, model.net.encoder4, model.net.encoder5
    ]

    head_modules = [
        model.net.x5_dem_1, model.net.x4_dem_1, model.net.x3_dem_1, model.net.x2_dem_1,
        model.net.up5, model.net.up4, model.net.up3, model.net.up2, model.net.up1,
        model.net.out1, model.net.out2, model.net.out3, model.net.out4, model.net.out5,
        model.net.out_boundary1, model.net.out_boundary2, model.net.out_boundary3,
        model.net.out_boundary4, model.net.out_boundary5,
        model.net.ega1, model.net.ega2, model.net.ega3, model.net.ega4,
        model.net.pat_att1, model.net.pat_att2, model.net.pat_att3, model.net.pat_att4,
        model.net.gabor_layer
    ]

    # Parametreleri topla
    backbone_params = []
    for module in backbone_modules:
        backbone_params.extend(list(module.parameters()))

    head_params = []
    for module in head_modules:
        head_params.extend(list(module.parameters()))

    # Weight decay filtreleme fonksiyonu
    def wd_filter(p):
        return p.ndim >= 2  # conv/linear ağırlıkları için weight decay, bias/norm için yok

    # Parameter grupları: backbone 0.33x LR, head 1.0x LR, bias/norm WD=0
    param_groups = [
        {"params": [p for p in backbone_params if wd_filter(p)], "lr": opt.lr*0.33, "weight_decay": opt.weight_decay},
        {"params": [p for p in backbone_params if not wd_filter(p)], "lr": opt.lr*0.33, "weight_decay": 0.0},
        {"params": [p for p in head_params if wd_filter(p)], "lr": opt.lr*1.0, "weight_decay": opt.weight_decay},
        {"params": [p for p in head_params if not wd_filter(p)], "lr": opt.lr*1.0, "weight_decay": 0.0},
    ]

    # AdamW optimizer - parameter grupları ile
    optimizer = torch.optim.AdamW(param_groups, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    # Her parametre grubu için kaç parametre olduğunu ve nasıl optimize edileceğini yazdır
    backbone_wd_params = len([p for p in backbone_params if wd_filter(p)])
    backbone_no_wd_params = len([p for p in backbone_params if not wd_filter(p)])
    head_wd_params = len([p for p in head_params if wd_filter(p)])
    head_no_wd_params = len([p for p in head_params if not wd_filter(p)])

    print(f"Backbone params: {len(backbone_params)} (WD: {backbone_wd_params}, no WD: {backbone_no_wd_params})")
    print(f"Head params: {len(head_params)} (WD: {head_wd_params}, no WD: {head_no_wd_params})")
    print(f"Backbone LR: {opt.lr*0.33:.6f} | Head LR: {opt.lr*1.0:.6f}")
    print(f"Weight decay: conv/linear={opt.weight_decay} | bias/norm=0.0")

    # Warmup + Cosine Scheduler - AdamW ile uyumlu ayarlar
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=opt.warmup_epochs,  # 10 epoch warmup
        total_epochs=opt.epoch,           # 300 epoch
        min_lr=1e-7                       # Daha düşük min_lr
    )

    # Deep supervision ağırlıkları - yüksek çözünürlüğe daha çok ağırlık
    mask_ds_weights = [1.0, 0.5, 0.3, 0.2, 0.15]        # Maske için ağırlıklar
    boundary_ds_weights = [1.0, 0.5, 0.3, 0.2, 0.15]    # Boundary için ağırlıklar

    criteria_loss = MultiTaskDeepSupervisionLoss(
        mask_loss_type="StructureLoss",       # Ana maske için Structure Loss
        boundary_loss_type="BceDiceLoss",     # Sınır için BceDice Loss
        boundary_weight=1.0,                  # Sınır kaybının ağırlığı (0.8→1.0)
        mask_ds_weights=mask_ds_weights,      # Maske deep supervision ağırlıkları
        boundary_ds_weights=boundary_ds_weights  # Boundary deep supervision ağırlıkları
    )

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.workers)

    # Validation dataset setup
    val_dataset_root = opt.train_path.replace('TrainDataset', 'TestDataset')
    print(f"Setting up validation datasets from: {val_dataset_root}")
    val_dataset = ValidationDataset(val_dataset_root, ['Kvasir', 'CVC-ClinicDB'], trainsize=opt.trainsize)

    # Kayıt yolu ve en iyi kayıp değişkeni
    save_path = './checkpoints/bb/{}/'.format(opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_step = len(train_loader)
    best_loss = float('inf')
    best_dice = 0.0
    best_val_epoch = 0

    print("#"*20, "Start Training", "#"*20)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {pytorch_total_params:,}")
    start_epoch = 0

    if opt.resume and os.path.exists(opt.resume_path):
        print(f"==> Loading checkpoint from {opt.resume_path}")
        checkpoint = torch.load(opt.resume_path, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_dice = checkpoint.get('best_dice', 0.0)
        best_val_epoch = checkpoint.get('best_val_epoch', 0)
        print(f"==> Resuming training from epoch {start_epoch} with best_loss {best_loss:.4f}, best_dice {best_dice:.4f}")

    # --- ANA EĞİTİM DÖNGÜSÜ ---
    val_dice = 0.0
    best_threshold = 0.5
    for epoch in range(start_epoch, opt.epoch):
        # Scheduler update (epoch başında) – warmup ilk epoch'tan itibaren uygulansın
        scheduler.step()

        # Tüm parametre gruplarının LR'lerini al
        current_lrs = scheduler.get_last_lr()
        backbone_conv_lr = current_lrs[0]  # Backbone conv/linear params
        backbone_bias_lr = current_lrs[1]  # Backbone bias/norm params
        head_conv_lr = current_lrs[2]      # Head conv/linear params
        head_bias_lr = current_lrs[3]      # Head bias/norm params

        # Learning rate'leri TensorBoard'a kaydet
        writer.add_scalar("LearningRate/Backbone_Conv", backbone_conv_lr, epoch)
        writer.add_scalar("LearningRate/Backbone_BiasNorm", backbone_bias_lr, epoch)
        writer.add_scalar("LearningRate/Head_Conv", head_conv_lr, epoch)
        writer.add_scalar("LearningRate/Head_BiasNorm", head_bias_lr, epoch)

        # Ortalama LR'leri de ekle
        avg_backbone_lr = (backbone_conv_lr + backbone_bias_lr) / 2
        avg_head_lr = (head_conv_lr + head_bias_lr) / 2
        writer.add_scalar("LearningRate/Backbone_Average", avg_backbone_lr, epoch)
        writer.add_scalar("LearningRate/Head_Average", avg_head_lr, epoch)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{opt.epoch}")
        print(f"Backbone LR: conv={backbone_conv_lr:.6f}, bias/norm={backbone_bias_lr:.6f}")
        print(f"Head LR:     conv={head_conv_lr:.6f}, bias/norm={head_bias_lr:.6f}")
        print(f"{'='*60}")

        # 1. Training
        avg_loss = train(train_loader, model, optimizer, epoch, criteria_loss, total_step, opt, writer)
        print(f'\n🛠️  Training Loss at epoch {epoch+1}: {avg_loss:.4f}')

        # 2. (Scheduler güncellemesi epoch başında yapıldı)

        # 3. Validation (her val_freq epoch'ta bir)
        if (epoch + 1) % opt.val_freq == 0 or epoch == opt.epoch - 1:
            print(f"\n🔍 Running validation at epoch {epoch + 1}...")
            val_dice, best_threshold, _, detailed_results = validate_model(model, val_dataset, device, epoch, writer)

            # Best validation check ve model kaydetme
            if val_dice > best_dice:
                best_dice = val_dice
                best_val_epoch = epoch + 1
                print(f"🎯 NEW BEST VALIDATION! Dice: {best_dice:.4f} @ threshold: {best_threshold:.2f} at epoch {best_val_epoch}")

                # En iyi validation model'ini kaydet - threshold bilgisiyle birlikte
                checkpoint = {
                    'epoch': epoch + 1,
                    'best_loss': avg_loss,
                    'best_dice': best_dice,
                    'best_threshold': best_threshold,  # En iyi threshold'u da kaydet
                    'best_val_epoch': best_val_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_results': detailed_results,  # Tüm threshold sweep sonuçları
                    'threshold_curve': detailed_results  # Dice(t) eğrisi
                }

                # Dosya adında epoch ve loss bilgisi
                filename = f'bb_res2net_ep{epoch+1:03d}_loss{avg_loss:.4f}_dice{best_dice:.4f}.pth'
                torch.save(checkpoint, os.path.join(save_path, filename))
                print(f"💾 Best Validation Model Saved!")
                print(f"   File: {filename}")
                print(f"   Dice: {best_dice:.4f} @ threshold: {best_threshold:.2f}")
                print(f"   Threshold curve saved in checkpoint")

        # Training loss tracking (sadece monitoring için)
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Training Loss: {avg_loss:.4f} (best: {best_loss:.4f})")
        if val_dice > 0:
            print(f"   Validation - Best Dice: {val_dice:.4f} @ threshold: {best_threshold:.2f}")
        print(f"   Best Validation: {best_dice:.4f} @ t={best_threshold:.2f} (epoch {best_val_epoch})")

    # Final summary
    print(f"\n🏁 Training Completed!")
    print(f"📈 Best Validation Dice: {best_dice:.4f} @ threshold: {best_threshold:.2f} at epoch {best_val_epoch}")
    print(f"📉 Best Training Loss: {best_loss:.4f}")

    writer.close()
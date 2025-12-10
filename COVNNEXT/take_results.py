# test.py
import os, argparse
import torch
import torch.nn.functional as F  # torch F (upsample/interpolate vs)
import numpy as np
import imageio
from PIL import Image
import torchvision.transforms as transforms

from lib.EGANet import EGANetModel


# -------------------------
# Minimal, sağlam test dataset
# -------------------------
class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = int(testsize)

        self.images = [os.path.join(image_root, f)
                       for f in os.listdir(image_root)
                       if f.lower().endswith(('.jpg', '.png'))]
        self.gts = [os.path.join(gt_root, f)
                    for f in os.listdir(gt_root)
                    if f.lower().endswith(('.tif', '.png'))]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # ✅ Resize’ı buradan kaldırdık
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def __len__(self):
        return self.size

    def load_data(self):
        img_path = self.images[self.index]
        gt_path  = self.gts[self.index]

        image = self.rgb_loader(img_path)
        gt    = self.binary_loader(gt_path)

        # ✅ Model giriş boyutunu burada PIL ile ayarlıyoruz
        image = image.resize((self.testsize, self.testsize), Image.BILINEAR)

        # sonra tensöre çevirip normalize et
        image = self.transform(image).unsqueeze(0)  # (1,3,Ht,Wt)

        name = os.path.basename(img_path)
        self.index += 1
        return image, gt, name

    @staticmethod
    def rgb_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def binary_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



# -------------------------
# Güvenli state_dict yükleme (PyTorch 2.6 uyumlu)
# -------------------------
from collections import OrderedDict
def load_state_dict_safely(model: torch.nn.Module, ckpt_obj):
    """
    ckpt -> 'state_dict' olabilir ya da direkt ağırlık sözlüğü olabilir.
    DataParallel 'module.' prefix'lerini temizler ve strict=False ile yükler.
    """
    if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj:
        state_dict = ckpt_obj['state_dict']
    elif isinstance(ckpt_obj, dict):
        # Direkt state_dict gibi davran
        state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith('module.') else k
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print('[warn] missing keys:', missing)
    if unexpected:
        print('[warn] unexpected keys:', unexpected)


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./checkpoints/bb/bb-Res2Net/bb_res2net_ep046_loss1.1422_dice0.9279.pth')
    parser.add_argument('--test_root_base', type=str, default='./dataset/TestDataset')
    parser.add_argument('--save_root_bin', type=str, default='./results/covnnext_az_augment_bin')
    parser.add_argument('--save_root_prob', type=str, default='./results/covnnext_az_augment_prob')
    parser.add_argument('--datasets', nargs='*',
                        default=['CVC-300','CVC-ClinicDB','Kvasir','CVC-ColonDB','ETIS-LaribPolypDB'])
    parser.add_argument('--save_prob', default=True , help='also save probability maps (_prob.png)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model + checkpoint (tek kez yükle)
    model = EGANetModel().to(device)

    # PyTorch 2.6 güvenlik değişikliği: weights_only=False (güvendiğin ckpt ise)
    ckpt = torch.load(args.pth_path, map_location='cpu', weights_only=False)
    load_state_dict_safely(model, ckpt)
    model.eval()

    best_t = float(ckpt.get('best_threshold', 0.5)) if isinstance(ckpt, dict) else 0.5
    print(f"Using threshold = {best_t:.2f}")

    for dname in args.datasets:
        data_path = os.path.join(args.test_root_base, dname)
        image_root = os.path.join(data_path, 'images')
        gt_root    = os.path.join(data_path, 'masks')

        save_path_bin = os.path.join(args.save_root_bin, dname)
        save_path_prob = os.path.join(args.save_root_prob, dname)
        os.makedirs(save_path_bin, exist_ok=True)
        os.makedirs(save_path_prob, exist_ok=True)

        loader = test_dataset(image_root, gt_root, args.testsize)
        print(f"[{dname}] {loader.size} images")

        for _ in range(loader.size):
            image, gt, name = loader.load_data()  # image: (1,3,Ht,Wt), gt: PIL(L)
            # Hedef boyut (orijinal GT boyutu)
            if isinstance(gt, Image.Image):
                W, H = gt.size  # PIL: (W,H)
            else:
                gt_np = np.asarray(gt, np.float32)
                H, W = gt_np.shape[:2]

            image = image.to(device)

            # forward -> logits
            pred_masks, _ = model(image)
            logit = pred_masks[0]                  # (1,1,h,w)
            prob  = torch.sigmoid(logit)           # (1,1,h,w)

            # orijinal GT boyutuna getir (prob üzerinde)
            prob = F.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)
            prob_np = prob.squeeze(0).squeeze(0).cpu().numpy()  # (H,W) [0,1]

            # 1) ikili maske (adı aynen koru)
            bin_mask = (prob_np >= best_t).astype(np.uint8) * 255
            imageio.imwrite(os.path.join(save_path_bin, name), bin_mask)

            # 2) opsiyonel: olasılık haritası
            if args.save_prob:
                root, _ = os.path.splitext(name)
                prob_name = f"{root}_prob.png"
                imageio.imwrite(os.path.join(save_path_prob, prob_name),
                                (prob_np * 255).astype(np.uint8))

if __name__ == '__main__':
    main()

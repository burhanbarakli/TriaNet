import os
from PIL import Image, ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numbers
import numpy as np
import cv2
import math
import torch


def mask_to_boundary(mask_pil, kernel_size=(3, 3)):
    """
    Bir PIL maske görüntüsünden sınır haritası üretir.
    """
    # PIL görüntüsünü numpy array'e çevir
    mask_np = np.array(mask_pil, dtype=np.uint8)

    # Morfolojik işlemlerle sınırı bul
    dilated = cv2.dilate(mask_np, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size), iterations=1)
    eroded = cv2.erode(mask_np, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size), iterations=1)

    boundary_np = dilated - eroded

    # Numpy array'i tekrar PIL görüntüsüne çevir
    return Image.fromarray(boundary_np)

# ------- Paired (img, mask) helper'lar -------

class RandomResizedCropPaired:
    def __init__(self, size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5):
        self.size, self.scale, self.ratio, self.p = size, scale, ratio, p

    def __call__(self, img, mask):
        if random.random() >= self.p:
            return img, mask
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, interpolation=Image.BILINEAR)
        mask = F.resized_crop(mask, i, j, h, w, self.size, interpolation=Image.NEAREST)
        return img, mask

class RandomAffinePaired:
    def __init__(self, degrees=7, translate=(0.02,0.02), scale=(0.95,1.05), shear=(-3,3), p=0.5):
        self.degrees, self.translate, self.scale, self.shear, self.p = degrees, translate, scale, shear, p

    def __call__(self, img, mask):
        if random.random() >= self.p:
            return img, mask
        params = transforms.RandomAffine.get_params((-self.degrees, self.degrees),
                    self.translate, self.scale, self.shear, img.size)
        angle, translations, scale, shear = params
        img = F.affine(img, angle, translations, scale, shear, interpolation=Image.BILINEAR)
        mask = F.affine(mask, angle, translations, scale, shear, interpolation=Image.NEAREST)
        return img, mask

# Elastic (çok küçük, stabil)
def _elastic(img, alpha=8.0, sigma=6.0, is_mask=False):
    # pillow tabanlı hızlı approx: hafif Gaussian blur + küçük perspektif jitter
    # (ağır sahte deformasyondan kaçınmak için mikro düzeyde)
    if not is_mask:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return img

class RandomElasticTiny:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, mask):
        if random.random() >= self.p:
            return img, mask
        return _elastic(img, is_mask=False), _elastic(mask, is_mask=True)

# ------- Photometric (yalnızca image) -------

class RandomGamma:
    def __init__(self, gmin=0.9, gmax=1.1, p=0.5):
        self.gmin, self.gmax, self.p = gmin, gmax, p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        gamma = random.uniform(self.gmin, self.gmax)
        return F.adjust_gamma(img, gamma)

class AddGaussianNoise:
    def __init__(self, std=0.01, p=0.3):
        self.std, self.p = std, p

    def __call__(self, tensor):
        if random.random() >= self.p:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask

class RandomRotate(object):
    def __init__(self, degrees, p=0.5, resample=False, expand=False, center=None):
        if isinstance(degrees,numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = self.get_params(self.degrees)
            return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(mask, angle, self.resample, self.expand, self.center)
        return img, mask    

class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations=True):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.augmentations = augmentations

        # Geometrik Augmentation'ları burada tanımla - p003.md'deki gelişmiş augmentations
        if self.augmentations:
            # Paired geometrik augmentations (image + mask)
            self.rrc = RandomResizedCropPaired(size=(self.trainsize, self.trainsize), scale=(0.8,1.0), ratio=(0.9,1.1), p=0.4)
            self.raff = RandomAffinePaired(degrees=7, translate=(0.02,0.02), scale=(0.95,1.05), shear=(-3,3), p=0.4)
            self.relast = RandomElasticTiny(p=0.25)  # Küçük datada p değerini düşürdüm
            self.rotate = RandomRotate(degrees=90, p=0.4)
            self.vf = RandomVerticalFlip(p=0.5)
            self.hf = RandomHorizontalFlip(p=0.5)

        # Fotometrik zinciri - p003.md'deki gelişmiş fotometrik augmentations
        if self.augmentations:
            self.img_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
                RandomGamma(0.9, 1.1, p=0.5),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Gaussian blur
                transforms.ToTensor(),
                AddGaussianNoise(std=0.01, p=0.3),  # Küçük Gaussian noise
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # Augmentation'sız versiyon (validation için)
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.gt_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # Resize işlemi
        image, gt = self.resize(image, gt)

        # Geometrik augmentations - p003.md'deki sıralama: önce crop/scale, sonra affine
        if self.augmentations:
            image, gt = self.rrc(image, gt)      # RandomResizedCrop first
            image, gt = self.raff(image, gt)     # Affine transforms second
            image, gt = self.rotate(image, gt)   # Rotation
            image, gt = self.vf(image, gt)       # Vertical flip
            image, gt = self.hf(image, gt)       # Horizontal flip
            image, gt = self.relast(image, gt)   # Elastic deformation last

        # Boundary üretimi tüm geometrik augmentation'dan sonra
        gt_boundary = mask_to_boundary(gt)

        # Transform'ları uygula (fotometrik sadece image'a)
        image = self.img_transform(image)        # Fotometrik augmentations burada
        gt = self.gt_transform(gt)
        gt_boundary = self.gt_transform(gt_boundary)

        return image, gt, gt_boundary

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        # Doğrudan trainsize'a resize et (aspect ratio korunmaz ama tutarlı boyut garantisi)
        return img.resize((self.trainsize, self.trainsize), Image.BILINEAR), gt.resize((self.trainsize, self.trainsize), Image.NEAREST)

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=False, augmentation=True):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  persistent_workers=(num_workers>1))
    return data_loader


def get_val_loader(image_root, gt_root, batchsize, trainsize, num_workers=4, pin_memory=False):
    """
    Validation için dataloader - augmentation'sız, shuffle'sız
    """
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentations=False)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batchsize,
                             shuffle=False,  # Validation için shuffle yok
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return data_loader


class ValidationDataset:
    """
    Validation için özel dataset sınıfı - multiple dataset'leri handle eder
    """
    def __init__(self, dataset_root, dataset_names=['Kvasir', 'CVC-ClinicDB'], trainsize=352):
        self.dataset_root = dataset_root
        self.dataset_names = dataset_names
        self.trainsize = trainsize
        self.datasets = {}
        self.total_size = 0

        # Her dataset için PolypDataset oluştur
        for name in dataset_names:
            image_root = os.path.join(dataset_root, name, 'images/')
            gt_root = os.path.join(dataset_root, name, 'masks/')

            if os.path.exists(image_root) and os.path.exists(gt_root):
                dataset = PolypDataset(image_root, gt_root, trainsize, augmentations=False)
                self.datasets[name] = dataset
                self.total_size += len(dataset)
                print(f"Loaded {name}: {len(dataset)} images")
            else:
                print(f"Warning: {name} dataset not found at {image_root}")

    def get_dataset_loaders(self, batchsize=8, num_workers=4):
        """
        Her dataset için ayrı dataloader döndür
        """
        loaders = {}
        for name, dataset in self.datasets.items():
            loader = DataLoader(dataset=dataset,
                               batch_size=batchsize,
                               shuffle=False,
                               num_workers=num_workers,
                               pin_memory=True)
            loaders[name] = loader
        return loaders

    def get_combined_loader(self, batchsize=8, num_workers=4):
        """
        Tüm dataset'leri birleştirip tek dataloader döndür
        """
        if not self.datasets:
            return None

        # Tüm dataset'leri birleştir
        combined_images = []
        combined_gts = []

        for dataset in self.datasets.values():
            combined_images.extend(dataset.images)
            combined_gts.extend(dataset.gts)

        # Yeni birleşik dataset oluştur
        combined_dataset = PolypDataset.__new__(PolypDataset)
        combined_dataset.trainsize = self.trainsize
        combined_dataset.images = combined_images
        combined_dataset.gts = combined_gts
        combined_dataset.size = len(combined_images)
        combined_dataset.augmentations = False

        # Transform'ları ekle
        combined_dataset.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        combined_dataset.gt_transform = transforms.ToTensor()

        # Metod binding
        combined_dataset.rgb_loader = self.datasets[list(self.datasets.keys())[0]].rgb_loader
        combined_dataset.binary_loader = self.datasets[list(self.datasets.keys())[0]].binary_loader
        combined_dataset.resize = self.datasets[list(self.datasets.keys())[0]].resize
        combined_dataset.__getitem__ = PolypDataset.__getitem__.__get__(combined_dataset, PolypDataset)

        loader = DataLoader(dataset=combined_dataset,
                           batch_size=batchsize,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=True)
        return loader

    def __len__(self):
        return self.total_size

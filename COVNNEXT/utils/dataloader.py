import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numbers
import numpy as np
import cv2


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

        # Transform için 'resize_transform' adını kullanıyorum, böylece 'resize' metodu ile karıştırılmaz
        self.resize_transform = transforms.Resize((self.trainsize, self.trainsize))

        # Geometrik Augmentation'ları burada tanımla
        if self.augmentations:
            self.rotate = RandomRotate(degrees=90, p=0.5) # Dereceyi 90 gibi daha makul bir değere çektim
            self.vf = RandomVerticalFlip(p=0.5)
            self.hf = RandomHorizontalFlip(p=0.5)

        # Tensöre çevirme ve normalizasyon işlemleri
        self.img_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # Özel resize metodunu kullan (iki parametre bekliyor ve iki değer döndürüyor)
        image, gt = self.resize(image, gt)

        if self.augmentations:
            image, gt = self.rotate(image, gt)
            image, gt = self.vf(image, gt)
            image, gt = self.hf(image, gt)

        # Boundary üretimi resize'dan sonra
        gt_boundary = mask_to_boundary(gt)

        image = self.img_transform(image)
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

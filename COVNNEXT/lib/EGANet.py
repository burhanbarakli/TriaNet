import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import *
from torchvision.transforms.functional import rgb_to_grayscale
from .nn_backbone_5_layer import ConvNeXtBackbone

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv_block(x) + self.shortcut(x)


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up_ori(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)

        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)


class Up(nn.Module):
    """
    Önce birleştirme, sonra ResidualConv ve en son ConvTranspose2d ile yukarı örnekleme.
    Bu yapı, EGANet'in orijinal mantığına sadık kalır.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1. Evrişim Bloğu:
        # Bu blok, birleştirilmiş özellik haritasını işleyecek.
        # Girdi kanalı, x1 ve x2 birleştiğinde oluşacak toplam kanal sayısıdır.
        # Örneğin up4 için bu 768'dir.
        # Çıktı kanalı ise bu katmandan sonra istediğimiz özellik sayısıdır (örn: 256).
        self.conv = ResidualConv(in_channels, out_channels)

        # 2. Yukarı Örnekleme Katmanı:
        # Bu katman, evrişimden çıkan 'out_channels' sayısındaki özelliği alır
        # ve boyutunu 2 katına çıkarır. Kanal sayısı değişmez.
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        # x1: Derin katmandan gelen özellik haritası.
        # x2: Kodlayıcıdan (encoder) gelen skip-connection.

        # 1. ADIM: Önce birleştir (Orijinal mantığı koru)
        x = torch.cat([x2, x1], dim=1)

        # 2. ADIM: Sonra evrişim uygula (ResidualConv ile daha güçlü)
        x_conv = self.conv(x)

        # 3. ADIM: En son yukarı örnekle (ConvTranspose2d ile daha modern)
        return self.up(x_conv)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Bu yeni sınıfı EGANet.py dosyanıza ekleyin
class FinalUpBlock(nn.Module):
    """
    Decoder'ın son adımı için özel blok.
    d2 özelliğini (H/2) yukarı örnekler ve bunu tam çözünürlüklü
    edge_feature ile birleştirerek nihai d1 özelliğini (H) üretir.
    """

    def __init__(self, in_channels_d2, in_channels_edge, out_channels_d1):
        super(FinalUpBlock, self).__init__()

        # d2'yi yukarı örnekleyecek katman. Çıktı kanalı out_channels_d1 olsun.
        self.up = nn.ConvTranspose2d(in_channels_d2, out_channels_d1, kernel_size=2, stride=2)

        # Birleştirilmiş özellikleri (d1_upsampled + edge_feature) işleyecek evrişim bloğu.
        # Girdi kanalı, iki özelliğin kanal toplamı olacak.
        self.conv = ResidualConv(out_channels_d1 + in_channels_edge, out_channels_d1)

    def forward(self, d2, edge_feature):
        # 1. Decoder'dan gelen d2 özelliğini (H/2) yukarı örnekle -> d1_upsampled (H)
        d1_upsampled = self.up(d2)

        # 2. Tam çözünürlüklü d1_upsampled ile tam çözünürlüklü edge_feature'ı birleştir
        x = torch.cat([d1_upsampled, edge_feature], dim=1)

        # 3. Birleştirilmiş özellikleri işleyerek nihai d1'i üret
        d1 = self.conv(x)

        return d1


#Multi-Scale Edge-Guided Attention Network
class EGANet(nn.Module):
    def __init__(self, n_channels, n_classes, backbone_name='convnext_tiny'):
        super(EGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.backbone = ConvNeXtBackbone(model_name=backbone_name, pretrained=True)

        # --- ConvNeXt KANAL SAYILARINA GÖRE GÜNCELLENMİŞ DEM KATMANLARI ---
        # convnext_tiny: [96, 192, 384, 768]
        # convnext_small: [96, 192, 384, 768] -> aynı
        # convnext_base: [128, 256, 512, 1024]
        # Bu değerler seçilen modele göre ayarlanmalıdır. Biz 'tiny' varsayıyoruz.
        dims = {'convnext_tiny': [96, 192, 384, 768], 'convnext_small': [96, 192, 384, 768]}
        in_dims = dims.get(backbone_name, dims['convnext_tiny']) # Varsayılan olarak tiny

        self.x5_dem_1 = nn.Sequential(nn.Conv2d(in_dims[3], 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(in_dims[2], 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(in_dims[1], 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(in_dims[0], 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.up5 = nn.Sequential(Conv(512, 512), nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.final_up = FinalUpBlock(in_channels_d2=64, in_channels_edge=1, out_channels_d1=64)
        # EGA1 için uygun katman eklenmeli - init fonksiyonunda tanımlanmalı
        self.ega1 = EGA(64)
        self.ega2 = EGA(64)
        self.ega3 = EGA(128)
        self.ega4 = EGA(256)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

        # YENİ EK: Sınır haritasını tahmin edecek olan çıkış katmanı.
        # Bu katman, ana maske çıkışı olan out1 ile aynı girdiyi (d1) alacağı için
        # aynı Out sınıfını ve aynı girdi kanal sayısını (64) kullanabiliriz.
        # Çıktı kanalı 1 olmalı (sınır/değil).
        # Sınır Çıkış Katmanları (YENİ)
        # Her bir 'd' özellik haritasından bir sınır tahmini üreteceğiz.
        # Kanal sayıları ilgili 'd' haritasının kanal sayılarıyla eşleşmeli.
        self.out_boundary5 = Out(512, 1) # d5'ten gelen çıktı için
        self.out_boundary4 = Out(256, 1) # d4'ten gelen çıktı için
        self.out_boundary3 = Out(128, 1) # d3'ten gelen çıktı için
        self.out_boundary2 = Out(64, 1)  # d2'den gelen çıktı için
        self.out_boundary1 = Out(64, 1)  # d1'den gelen çıktı için

        # GaborConv burada tanımlanmalı
        self.gabor_layer = GaborConv(n_filters=4)  # Varsayılan 4 filtre
        num_gabor_filters = self.gabor_layer.n_filters

        # Fusion conv katmanları - resnet versiyonundan eklendi
        self.fusion_conv4 = nn.Conv2d(256 * 2, 256, kernel_size=1)
        # 128 + 128 -> 128
        self.fusion_conv3 = nn.Conv2d(128 * 2, 128, kernel_size=1)
        # 64 + 64 -> 64
        self.fusion_conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1)
        self.fusion_conv1 = nn.Conv2d(64 * 2, 64, kernel_size=1)

        # PatternAttention katmanları - resnet versiyonundan eklendi
        self.pat_att1 = PatternAttention(64)
        self.pat_att2 = PatternAttention(64)
        self.pat_att3 = PatternAttention(128)
        self.pat_att4 = PatternAttention(256)

        self.feature_fusion_conv = nn.Conv2d(1 + num_gabor_filters, 1, kernel_size=1, padding=0) # attention eklenebilir.

    def forward(self, x):
        # Normalize'sız kopya - Laplacian ve Gabor için raw piksel değerleri
        # ImageNet normalizasyonunu tersine çevir
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_raw = x * std + mean  # [0,1] aralığına getir
        x_raw = torch.clamp(x_raw, 0, 1)  # Güvenlik için sınırla

        # Raw görüntüden edge feature üret
        grayscale_img = rgb_to_grayscale(x_raw)
        laplacian = make_laplace_pyramid(grayscale_img, 5, 1)
        laplacian_feature = laplacian[1]

        # Gabor features da raw görüntüden
        gabor_features = self.gabor_layer(grayscale_img)

        #Encoder (normalize'lu x ile)
        encoder_features = self.backbone(x)
        e5 = encoder_features['stage5'] #768*8*8
        e4 = encoder_features['stage4'] #384*16*16
        e3 = encoder_features['stage3'] #192*32*32
        e2 = encoder_features['stage2'] #96*64*64

        e5_dem_1 = self.x5_dem_1(e5)
        e4_dem_1 = self.x4_dem_1(e4)
        e3_dem_1 = self.x3_dem_1(e3)
        e2_dem_1 = self.x2_dem_1(e2)

        #Decoder - ÇALIŞAN VERSİYON MANTIĞI (resnet benzeri)
        d5 = self.up5(e5_dem_1)
        out5 = self.out5(d5)
        out_b5 = self.out_boundary5(d5)
        ega4 = self.ega4(laplacian_feature, e4_dem_1, out5)
        pattern_out4 = self.pat_att4(gabor_features, e4_dem_1)
        fused_features4 = torch.cat([ega4, pattern_out4], dim=1)
        fused_out4 = self.fusion_conv4(fused_features4)

        d4 = self.up4(d5, fused_out4)
        out4 = self.out4(d4)
        out_b4 = self.out_boundary4(d4)
        ega3 = self.ega3(laplacian_feature, e3_dem_1, out4)
        pattern_out3 = self.pat_att3(gabor_features, e3_dem_1)
        fused_features3 = torch.cat([ega3, pattern_out3], dim=1)
        fused_out3 = self.fusion_conv3(fused_features3)

        d3 = self.up3(d4, fused_out3)
        out3 = self.out3(d3)
        out_b3 = self.out_boundary3(d3)
        ega2 = self.ega2(laplacian_feature, e2_dem_1, out3)
        pattern_out2 = self.pat_att2(gabor_features, e2_dem_1)
        fused_features2 = torch.cat([ega2, pattern_out2], dim=1)
        fused_out2 = self.fusion_conv2(fused_features2)

        d2 = self.up2(d3, fused_out2)
        out2 = self.out2(d2)
        out_b2 = self.out_boundary2(d2)

        # ConvNeXt'te e1 olmadığı için, ega1 için d2'yi kullanıyoruz (out2 ile aynı boyutta)
        # Bu, d2'nin 64 kanalı olduğunu ve out2 ile uyumlu olduğunu garanti eder
        ega1 = self.ega1(laplacian_feature, d2, out2)  # e2_dem_1 yerine d2 kullan
        pattern_out1 = self.pat_att1(gabor_features, d2)  # e2_dem_1 yerine d2 kullan
        fused_features1 = torch.cat([ega1, pattern_out1], dim=1)
        fused_out1 = self.fusion_conv1(fused_features1)

        d1 = self.up1(d2, fused_out1)
        out1 = self.out1(d1)
        out_b1 = self.out_boundary1(d1)

        pred_masks = [out1, out2, out3, out4, out5]
        pred_boundaries = [out_b1, out_b2, out_b3, out_b4, out_b5]

        return pred_masks, pred_boundaries


class EGANetModel(nn.Module): 
    def __init__(self, n_channels=3, n_classes=1):
        super(EGANetModel,self).__init__()
        self.channel = n_channels
        self.num_classes = n_classes
        self.net = EGANet(self.channel, self.num_classes)

    def forward(self, images):
        pred_masks, pred_boundaries= self.net(images)
        return pred_masks, pred_boundaries

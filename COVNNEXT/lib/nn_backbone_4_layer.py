import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List

#bunlar 4 katmanlı

# ==============================================================================
# 1. Resnet s  Backbone
# ==============================================================================

class ResNetBackbone(nn.Module):
    """
    Herhangi bir Torchvision ResNet modelini bir özellik çıkarıcı (backbone)
    olarak kullanan bir nn.Module.

    Bu sınıf, kendi özel modellerinize (U-Net, vb.) kolayca entegre edilmek
    üzere tasarlanmıştır. Ön işleme adımlarını içermez; girdinin zaten
    normalize edilmiş bir tensor olduğunu varsayar.

    Args:
        model_name (str): Yüklenecek ResNet modelinin adı.
                          'resnet18', 'resnet34', 'resnet50', vb.
        pretrained (bool): True ise ImageNet üzerinde eğitilmiş ağırlıklar
                           kullanılır.
    """

    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True):
        super().__init__()  # nn.Module'ün __init__ metodunu çağırmak zorunludur.

        # Modern torchvision'da ağırlıkları yüklemenin en doğru yolu
        weights = "DEFAULT" if pretrained else None

        # Orijinal ResNet modelini yükle
        resnet_model = models.get_model(model_name, weights=weights)

        # ResNet'in ihtiyacımız olan katmanlarını kopyalayıp bu sınıfa
        # alt modüller olarak kaydediyoruz. Bu, hook kullanmaktan daha
        # verimli ve PyTorch standartlarına daha uygundur.

        # İlk konvolüsyon bloğu ("stem")
        self.stem = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool
        )

        # ResNet'in 4 ana katmanı
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Girdi tensor'ünü ağdan geçirir ve 4 ana katmanın çıktılarını bir
        sözlük olarak döndürür.

        Args:
            x (torch.Tensor): Girdi tensörü. Boyut: [N, 3, H, W]

        Returns:
            Dict[str, torch.Tensor]: Katman isimleri ve çıktı tensörlerini
                                     içeren bir sözlük.
        """
        features = {}

        # Girdiyi katmanlardan sırayla geçir
        x = self.stem(x)
        x = self.layer1(x)
        features['layer1'] = x

        x = self.layer2(x)
        features['layer2'] = x

        x = self.layer3(x)
        features['layer3'] = x

        x = self.layer4(x)
        features['layer4'] = x

        return features

# ==============================================================================
# 2. MobileNetV2 Backbone
# ==============================================================================
class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 modelini bir özellik çıkarıcı olarak kullanır."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        mobilenet_v2 = models.mobilenet_v2(weights=weights)

        # MobileNetV2'nin 'features' bloğu ana özellik çıkarıcıdır.
        # Bu bloğu, farklı çözünürlük seviyelerinden çıktı alacak şekilde 4 parçaya ayırıyoruz.
        all_features = mobilenet_v2.features
        self.stage1 = nn.Sequential(*all_features[0:4])
        self.stage2 = nn.Sequential(*all_features[4:7])
        self.stage3 = nn.Sequential(*all_features[7:14])
        self.stage4 = nn.Sequential(*all_features[14:19])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stage1(x);
        features['stage1'] = x
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x
        x = self.stage4(x);
        features['stage4'] = x
        return features


# ==============================================================================
# 3. ShuffleNetV2 Backbone
# ==============================================================================
class ShuffleNetV2Backbone(nn.Module):
    """ShuffleNetV2 x1.0 modelini bir özellik çıkarıcı olarak kullanır."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        shufflenet_v2 = models.shufflenet_v2_x1_0(weights=weights)

        # ShuffleNet'in yapısı ResNet'e çok benzer, bu da işimizi kolaylaştırır.
        self.stem = nn.Sequential(shufflenet_v2.conv1, shufflenet_v2.maxpool)
        self.stage1 = shufflenet_v2.stage2
        self.stage2 = shufflenet_v2.stage3
        self.stage3 = shufflenet_v2.stage4
        # Son konvolüsyon katmanını da 4. özellik olarak alıyoruz.
        self.stage4 = shufflenet_v2.conv5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stem(x)
        x = self.stage1(x);
        features['stage1'] = x
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x
        x = self.stage4(x);
        features['stage4'] = x
        return features


# ==============================================================================
# 4. EfficientNet Backbone (B0-B4 için tek bir sınıf)
# ==============================================================================
class EfficientNetBackbone(nn.Module):
    """EfficientNet (B0-B7) modellerini özellik çıkarıcı olarak kullanır."""

    def __init__(self, model_name: str = 'efficientnet_b0', pretrained: bool = True):
        super().__init__()

        # Her model varyantı için katmanların bittiği indeksler. Bu, sınıfı dinamik yapar.
        stage_indices = {
            'efficientnet_b0': [1, 2, 4, 8],
            'efficientnet_b1': [2, 4, 7, 15],
            'efficientnet_b2': [2, 4, 7, 15],
            'efficientnet_b3': [2, 4, 7, 17],
            'efficientnet_b4': [3, 5, 9, 21],
            'efficientnet_b5': [3, 6, 11, 26],
            'efficientnet_b6': [4, 8, 14, 30],
            'efficientnet_b7': [5, 10, 17, 37],
        }

        if model_name not in stage_indices:
            raise ValueError(f"Model adı {model_name} desteklenmiyor veya geçersiz.")

        weights = "DEFAULT" if pretrained else None
        efficientnet = models.get_model(model_name, weights=weights)

        all_features = efficientnet.features
        indices = stage_indices[model_name]

        # İndekslere göre katmanları dinamik olarak ayır
        self.stage1 = nn.Sequential(*all_features[0:indices[0] + 1])
        self.stage2 = nn.Sequential(*all_features[indices[0] + 1: indices[1] + 1])
        self.stage3 = nn.Sequential(*all_features[indices[1] + 1: indices[2] + 1])
        self.stage4 = nn.Sequential(*all_features[indices[2] + 1: indices[3] + 1])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stage1(x)
        features['stage1'] = x
        x = self.stage2(x)
        features['stage2'] = x
        x = self.stage3(x)
        features['stage3'] = x
        x = self.stage4(x)
        features['stage4'] = x
        return features


# ==============================================================================
# 5. VGG19 (Batch Norm) Backbone
# ==============================================================================
class VGG19Backbone(nn.Module):
    """VGG19 (Batch Norm ile) modelini özellik çıkarıcı olarak kullanır."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        vgg19_bn = models.vgg19_bn(weights=weights)

        all_features = vgg19_bn.features
        # VGG'de her boyut küçültme (MaxPool2d) bir "stage" sonu olarak kabul edilir.
        # İndeksler, ilk 4 MaxPool2d katmanının sonrasını işaret eder.
        self.stage1 = nn.Sequential(*all_features[0:7])  # 1. MaxPool sonrası
        self.stage2 = nn.Sequential(*all_features[7:14])  # 2. MaxPool sonrası
        self.stage3 = nn.Sequential(*all_features[14:27])  # 3. MaxPool sonrası
        self.stage4 = nn.Sequential(*all_features[27:40])  # 4. MaxPool sonrası

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stage1(x);
        features['stage1'] = x
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x
        x = self.stage4(x);
        features['stage4'] = x
        return features


# ==============================================================================
# 6. InceptionV3 Backbone
# ==============================================================================
class InceptionV3Backbone(nn.Module):
    """InceptionV3 modelini özellik çıkarıcı olarak kullanır."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # --- FİNAL DÜZELTME ---
        # `pretrained` durumuna göre tüm parametreleri dinamik olarak ayarlıyoruz.
        if pretrained:
            weights = "DEFAULT"
            aux_logits_setting = True
            init_weights_setting = False  # Hazır ağırlık varken rastgele başlatma YAPMA
        else:
            weights = None
            aux_logits_setting = False
            init_weights_setting = True  # Sıfırdan eğitim için rastgele başlatma YAP

        inception_v3 = models.inception_v3(
            weights=weights,
            aux_logits=aux_logits_setting,
            init_weights=init_weights_setting
        )

        # InceptionV3'ün yapısı sıralı olmadığı için katmanları tek tek alıyoruz.
        self.Conv2d_1a_3x3 = inception_v3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception_v3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception_v3.Conv2d_2b_3x3
        self.maxpool1 = inception_v3.maxpool1
        self.Conv2d_3b_1x1 = inception_v3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception_v3.Conv2d_4a_3x3
        self.maxpool2 = inception_v3.maxpool2
        self.Mixed_5b = inception_v3.Mixed_5b
        self.Mixed_5c = inception_v3.Mixed_5c
        self.Mixed_5d = inception_v3.Mixed_5d
        self.Mixed_6a = inception_v3.Mixed_6a
        self.Mixed_6b = inception_v3.Mixed_6b
        self.Mixed_6c = inception_v3.Mixed_6c
        self.Mixed_6d = inception_v3.Mixed_6d
        self.Mixed_6e = inception_v3.Mixed_6e
        self.Mixed_7a = inception_v3.Mixed_7a
        self.Mixed_7b = inception_v3.Mixed_7b
        self.Mixed_7c = inception_v3.Mixed_7c

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        # Forward pass aynı kalabilir, çünkü aux_logits'i hiç kullanmıyoruz.
        x = self.Conv2d_1a_3x3(x);
        x = self.Conv2d_2a_3x3(x);
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x);
        x = self.Conv2d_3b_1x1(x);
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x);
        features['stage1'] = x
        x = self.Mixed_5b(x);
        x = self.Mixed_5c(x);
        x = self.Mixed_5d(x);
        features['stage2'] = x
        x = self.Mixed_6a(x);
        x = self.Mixed_6b(x);
        x = self.Mixed_6c(x);
        x = self.Mixed_6d(x);
        x = self.Mixed_6e(x);
        features['stage3'] = x
        x = self.Mixed_7a(x);
        x = self.Mixed_7b(x);
        x = self.Mixed_7c(x);
        features['stage4'] = x
        return features


# ==============================================================================
# 7: ResNeXt Backbone
# ==============================================================================
class ResNeXtBackbone(nn.Module):
    """ResNeXt-101 32x8d modelini bir özellik çıkarıcı olarak kullanır."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None

        # ResNeXt'in yapısı ResNet ile neredeyse aynıdır.
        resnext_model = models.resnext101_32x8d(weights=weights)

        self.stem = nn.Sequential(
            resnext_model.conv1,
            resnext_model.bn1,
            resnext_model.relu,
            resnext_model.maxpool
        )
        self.layer1 = resnext_model.layer1
        self.layer2 = resnext_model.layer2
        self.layer3 = resnext_model.layer3
        self.layer4 = resnext_model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stem(x)
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features


# ==============================================================================
# 8. RegNet Backbone
# ==============================================================================
class RegNetBackbone(nn.Module):
    """RegNet modellerini özellik çıkarıcı olarak kullanır."""

    def __init__(self, model_name: str = 'regnet_y_400mf', pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        regnet = models.get_model(model_name, weights=weights)

        # RegNet'in yapısı ResNet gibi çok temiz ve aşamalara ayrılmış.
        self.stem = regnet.stem
        self.stage1 = regnet.trunk_output.block1
        self.stage2 = regnet.trunk_output.block2
        self.stage3 = regnet.trunk_output.block3
        self.stage4 = regnet.trunk_output.block4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stem(x)
        x = self.stage1(x);
        features['stage1'] = x
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x
        x = self.stage4(x);
        features['stage4'] = x
        return features


# ==============================================================================
# 9. ConvNeXt Backbone (Tüm varyantlar için tek bir sınıf)
# ==============================================================================
class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt (tiny, small, base, large) modellerini özellik çıkarıcı
    olarak kullanan bir nn.Module.
    """

    def __init__(self, model_name: str = 'convnext_tiny', pretrained: bool = True):
        super().__init__()

        # Desteklenen ConvNeXt modellerini kontrol et
        supported_models = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']
        if model_name not in supported_models:
            raise ValueError(f"Model adı {model_name} desteklenmiyor. Lütfen {supported_models} arasından seçin.")

        weights = "DEFAULT" if pretrained else None
        convnext = models.get_model(model_name, weights=weights)

        # ConvNeXt'in `features` bloğu zaten doğal aşamalara ayrılmıştır.
        all_features = convnext.features

        # Her aşama bir downsampling bloğu ve bir dizi ana bloktan oluşur.
        self.stage1 = nn.Sequential(all_features[0], all_features[1])
        self.stage2 = nn.Sequential(all_features[2], all_features[3])
        self.stage3 = nn.Sequential(all_features[4], all_features[5])
        self.stage4 = nn.Sequential(all_features[6], all_features[7])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        # ConvNeXt katmanları doğrudan doğru formatta (B, C, H, W) çıktı verir.
        x = self.stage1(x)
        features['stage1'] = x

        x = self.stage2(x)
        features['stage2'] = x

        x = self.stage3(x)
        features['stage3'] = x

        x = self.stage4(x)
        features['stage4'] = x

        return features

# ==============================================================================
# TEST FONKSİYONU
# ==============================================================================
if __name__ == '__main__':
    # Test edilecek modellerin ve sınıflarının listesi
    backbone_tests = {
        "ResNet-50": (ResNetBackbone, {'model_name': 'resnet50'}),
        "MobileNetV2": MobileNetV2Backbone,
        "ShuffleNetV2": ShuffleNetV2Backbone,
        "EfficientNet-B0": (EfficientNetBackbone, {'model_name': 'efficientnet_b0'}),
        "EfficientNet-B1": (EfficientNetBackbone, {'model_name': 'efficientnet_b1'}),
        "EfficientNet-B2": (EfficientNetBackbone, {'model_name': 'efficientnet_b2'}),
        "EfficientNet-B3": (EfficientNetBackbone, {'model_name': 'efficientnet_b3'}),
        "EfficientNet-B4": (EfficientNetBackbone, {'model_name': 'efficientnet_b4'}),
        "EfficientNet-B5": (EfficientNetBackbone, {'model_name': 'efficientnet_b5'}),
        "EfficientNet-B6": (EfficientNetBackbone, {'model_name': 'efficientnet_b6'}),
        "EfficientNet-B7": (EfficientNetBackbone, {'model_name': 'efficientnet_b7'}),
        "VGG19_BN": VGG19Backbone,
        "InceptionV3": InceptionV3Backbone,
        "ResNeXt101": ResNeXtBackbone,
        "RegNetY_3_2_GF": (RegNetBackbone, {'model_name': 'regnet_y_3_2gf'}),
        "RegNetY_16GF": (RegNetBackbone, {'model_name': 'regnet_y_16gf'}),
        "RegNetY_32GF": (RegNetBackbone, {'model_name': 'regnet_y_32gf'}),
        "RegNetY_8GF": (RegNetBackbone, {'model_name': 'regnet_y_8gf'}),
        "ConvNeXt_Small": (ConvNeXtBackbone, {'model_name': 'convnext_small'}),
        "ConvNeXt_Base": (ConvNeXtBackbone, {'model_name': 'convnext_base'}),
        "ConvNeXt_Tiny": (ConvNeXtBackbone, {'model_name': 'convnext_tiny'}),
        "ConvNeXt_Large": (ConvNeXtBackbone, {'model_name': 'convnext_large'}),
    }

    # EfficientNet B6 ve B7 daha büyük girdi boyutları bekler.
    inputs = {
        "default": torch.randn(1, 3, 224, 224),
        "inception": torch.randn(1, 3, 299, 299),
        "effnet_b6": torch.randn(1, 3, 528, 528),
        "effnet_b7": torch.randn(1, 3, 600, 600),
    }

    for name, model_info in backbone_tests.items():
        print(f"{'=' * 25}\nTEST EDİLİYOR: {name}\n{'=' * 25}")
        try:
            if isinstance(model_info, tuple):
                model_class, model_kwargs = model_info
            else:
                model_class, model_kwargs = model_info, {}

            # Ağırlıkları indirmemek için pretrained=False, indirmek için True yapın.
            backbone = model_class(pretrained=True, **model_kwargs)
            backbone.eval()

            # Modele göre doğru girdi boyutunu seç
            if name == "InceptionV3":
                input_tensor = inputs["inception"]
            elif name == "EfficientNet-B6":
                input_tensor = inputs["effnet_b6"]
            elif name == "EfficientNet-B7":
                input_tensor = inputs["effnet_b7"]
            else:
                input_tensor = inputs["default"]

            with torch.no_grad():
                features = backbone(input_tensor)

            print(f"Girdi Boyutu: {input_tensor.shape}")
            print(f"{name} için Çıktı Özellik Haritaları:")
            for stage_name, fmap in features.items():
                print(f" -> {stage_name}: {fmap.shape}")
            print("\n")

        except Exception as e:
            print(f"HATA: {name} modeli test edilirken bir sorun oluştu: {e}\n")

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List


# ==============================================================================
# 1. ResNet Backbone (5 Aşamalı)
# ==============================================================================
class ResNetBackbone(nn.Module):
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        resnet_model = models.get_model(model_name, weights=weights)
        self.stem = nn.Sequential(resnet_model.conv1, resnet_model.bn1, resnet_model.relu)
        self.stage1_pool = resnet_model.maxpool
        self.stage2_block = resnet_model.layer1
        self.stage3_block = resnet_model.layer2
        self.stage4_block = resnet_model.layer3
        self.stage5_block = resnet_model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        x = self.stem(x);
        features['stage1'] = x
        x_pooled = self.stage1_pool(x)
        x = self.stage2_block(x_pooled);
        features['stage2'] = x
        x = self.stage3_block(x);
        features['stage3'] = x
        x = self.stage4_block(x);
        features['stage4'] = x
        x = self.stage5_block(x);
        features['stage5'] = x
        return features


# ==============================================================================
# 2. MobileNetV2 Backbone (5 Aşamalı)
# ==============================================================================
class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        mobilenet_v2 = models.mobilenet_v2(weights=weights)
        all_features = mobilenet_v2.features
        self.stage1 = nn.Sequential(*all_features[0:2]);
        self.stage2 = nn.Sequential(*all_features[2:4])
        self.stage3 = nn.Sequential(*all_features[4:7]);
        self.stage4 = nn.Sequential(*all_features[7:14])
        self.stage5 = nn.Sequential(*all_features[14:19])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stage1(x);
        features['stage1'] = x;
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x;
        x = self.stage4(x);
        features['stage4'] = x
        x = self.stage5(x);
        features['stage5'] = x;
        return features


# ==============================================================================
# 3. ShuffleNetV2 Backbone (5 Aşamalı)
# ==============================================================================
class ShuffleNetV2Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        shufflenet_v2 = models.shufflenet_v2_x1_0(weights=weights)
        self.stage1 = shufflenet_v2.conv1;
        self.stage2 = nn.Sequential(shufflenet_v2.maxpool, shufflenet_v2.stage2)
        self.stage3 = shufflenet_v2.stage3;
        self.stage4 = shufflenet_v2.stage4;
        self.stage5 = shufflenet_v2.conv5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stage1(x);
        features['stage1'] = x;
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x;
        x = self.stage4(x);
        features['stage4'] = x
        x = self.stage5(x);
        features['stage5'] = x;
        return features


# ==============================================================================
# 4. EfficientNet Backbone (5 Aşamalı, B0-B7)
# ==============================================================================
class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name: str = 'efficientnet_b0', pretrained: bool = True):
        super().__init__()
        stage_endpoints = {
            'efficientnet_b0': [0, 1, 2, 4, 8], 'efficientnet_b1': [1, 2, 4, 7, 15],
            'efficientnet_b2': [1, 2, 4, 7, 15],
            'efficientnet_b3': [1, 2, 4, 7, 17], 'efficientnet_b4': [1, 3, 5, 9, 21],
            'efficientnet_b5': [1, 3, 6, 11, 26],
            'efficientnet_b6': [2, 4, 8, 14, 30], 'efficientnet_b7': [2, 5, 10, 17, 37],
        }
        if model_name not in stage_endpoints: raise ValueError(f"Desteklenmeyen model: {model_name}")
        weights = "DEFAULT" if pretrained else None
        efficientnet = models.get_model(model_name, weights=weights)
        all_features = efficientnet.features;
        indices = stage_endpoints[model_name]
        self.stage1 = nn.Sequential(*all_features[0:indices[0] + 1]);
        self.stage2 = nn.Sequential(*all_features[indices[0] + 1: indices[1] + 1])
        self.stage3 = nn.Sequential(*all_features[indices[1] + 1: indices[2] + 1]);
        self.stage4 = nn.Sequential(*all_features[indices[2] + 1: indices[3] + 1])
        self.stage5 = nn.Sequential(*all_features[indices[3] + 1: indices[4] + 1])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stage1(x);
        features['stage1'] = x;
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x;
        x = self.stage4(x);
        features['stage4'] = x
        x = self.stage5(x);
        features['stage5'] = x;
        return features


# ==============================================================================
# 5. VGG19 (Batch Norm) Backbone (5 Aşamalı)
# ==============================================================================
class VGG19Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        vgg19_bn = models.vgg19_bn(weights=weights)
        all_features = vgg19_bn.features
        self.stage1 = nn.Sequential(*all_features[0:7]);
        self.stage2 = nn.Sequential(*all_features[7:14])
        self.stage3 = nn.Sequential(*all_features[14:27]);
        self.stage4 = nn.Sequential(*all_features[27:40])
        self.stage5 = nn.Sequential(*all_features[40:53])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stage1(x);
        features['stage1'] = x;
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x;
        x = self.stage4(x);
        features['stage4'] = x
        x = self.stage5(x);
        features['stage5'] = x;
        return features


# ==============================================================================
# 6. InceptionV3 Backbone (5 Aşamalı)
# ==============================================================================
class InceptionV3Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights, aux_logits, init_weights = "DEFAULT", True, False
        else:
            weights, aux_logits, init_weights = None, False, True
        inception_v3 = models.inception_v3(weights=weights, aux_logits=aux_logits, init_weights=init_weights)
        self.stage1_block = nn.Sequential(inception_v3.Conv2d_1a_3x3, inception_v3.Conv2d_2a_3x3,
                                          inception_v3.Conv2d_2b_3x3, inception_v3.maxpool1)
        self.stage2_block = nn.Sequential(inception_v3.Conv2d_3b_1x1, inception_v3.Conv2d_4a_3x3, inception_v3.maxpool2)
        self.stage3_block = nn.Sequential(inception_v3.Mixed_5b, inception_v3.Mixed_5c, inception_v3.Mixed_5d)
        self.stage4_block = nn.Sequential(inception_v3.Mixed_6a, inception_v3.Mixed_6b, inception_v3.Mixed_6c,
                                          inception_v3.Mixed_6d, inception_v3.Mixed_6e)
        self.stage5_block = nn.Sequential(inception_v3.Mixed_7a, inception_v3.Mixed_7b, inception_v3.Mixed_7c)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stage1_block(x);
        features['stage1'] = x;
        x = self.stage2_block(x);
        features['stage2'] = x
        x = self.stage3_block(x);
        features['stage3'] = x;
        x = self.stage4_block(x);
        features['stage4'] = x
        x = self.stage5_block(x);
        features['stage5'] = x;
        return features


# ==============================================================================
# 7. ResNeXt Backbone (5 Aşamalı)
# ==============================================================================
class ResNeXtBackbone(ResNetBackbone):
    def __init__(self, model_name: str = 'resnext101_32x8d', pretrained: bool = True):
        super().__init__(model_name=model_name, pretrained=pretrained)


# ==============================================================================
# 8. RegNet Backbone (5 Aşamalı)
# ==============================================================================
class RegNetBackbone(nn.Module):
    def __init__(self, model_name: str = 'regnet_y_400mf', pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        regnet = models.get_model(model_name, weights=weights)
        self.stem = regnet.stem;
        self.stage2_block = regnet.trunk_output.block1;
        self.stage3_block = regnet.trunk_output.block2
        self.stage4_block = regnet.trunk_output.block3;
        self.stage5_block = regnet.trunk_output.block4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stem(x);
        features['stage1'] = x;
        x = self.stage2_block(x);
        features['stage2'] = x
        x = self.stage3_block(x);
        features['stage3'] = x;
        x = self.stage4_block(x);
        features['stage4'] = x
        x = self.stage5_block(x);
        features['stage5'] = x;
        return features


# ==============================================================================
# 9. ConvNeXt Backbone (5 Aşamalı)
# ==============================================================================
class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name: str = 'convnext_tiny', pretrained: bool = True):
        super().__init__()
        supported_models = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']
        if model_name not in supported_models: raise ValueError(f"Desteklenmeyen model: {model_name}")
        weights = "DEFAULT" if pretrained else None;
        convnext = models.get_model(model_name, weights=weights)
        all_features = convnext.features
        self.stage1 = all_features[0];
        self.stage2 = all_features[1];
        self.stage3 = nn.Sequential(all_features[2], all_features[3])
        self.stage4 = nn.Sequential(all_features[4], all_features[5]);
        self.stage5 = nn.Sequential(all_features[6], all_features[7])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {};
        x = self.stage1(x);
        features['stage1'] = x;
        x = self.stage2(x);
        features['stage2'] = x
        x = self.stage3(x);
        features['stage3'] = x;
        x = self.stage4(x);
        features['stage4'] = x
        x = self.stage5(x);
        features['stage5'] = x;
        return features



# ==============================================================================
# TEST FONKSİYONU
# ==============================================================================
# if __name__ == '__main__':
#     backbone_tests = {
#         "ResNet-50": (ResNetBackbone, {'model_name': 'resnet50'}),
#         "MobileNetV2": MobileNetV2Backbone,
#         "ShuffleNetV2": ShuffleNetV2Backbone,
#         "EfficientNet-B0": (EfficientNetBackbone, {'model_name': 'efficientnet_b0'}),
#         "EfficientNet-B1": (EfficientNetBackbone, {'model_name': 'efficientnet_b1'}),
#         "EfficientNet-B2": (EfficientNetBackbone, {'model_name': 'efficientnet_b2'}),
#         "EfficientNet-B3": (EfficientNetBackbone, {'model_name': 'efficientnet_b3'}),
#         "EfficientNet-B4": (EfficientNetBackbone, {'model_name': 'efficientnet_b4'}),
#         "EfficientNet-B5": (EfficientNetBackbone, {'model_name': 'efficientnet_b5'}),
#         "EfficientNet-B6": (EfficientNetBackbone, {'model_name': 'efficientnet_b6'}),
#         "EfficientNet-B7": (EfficientNetBackbone, {'model_name': 'efficientnet_b7'}),
#         "VGG19_BN": VGG19Backbone,
#         "InceptionV3": InceptionV3Backbone,
#         "ResNeXt101": ResNeXtBackbone,
#         "RegNetY_3_2_GF": (RegNetBackbone, {'model_name': 'regnet_y_3_2gf'}),
#         "RegNetY_16GF": (RegNetBackbone, {'model_name': 'regnet_y_16gf'}),
#         "RegNetY_32GF": (RegNetBackbone, {'model_name': 'regnet_y_32gf'}),
#         "RegNetY_8GF": (RegNetBackbone, {'model_name': 'regnet_y_8gf'}),
#         "ConvNeXt_Small": (ConvNeXtBackbone, {'model_name': 'convnext_small'}),
#         "ConvNeXt_Base": (ConvNeXtBackbone, {'model_name': 'convnext_base'}),
#         "ConvNeXt_Tiny": (ConvNeXtBackbone, {'model_name': 'convnext_tiny'}),
#         "ConvNeXt_Large": (ConvNeXtBackbone, {'model_name': 'convnext_large'}),
#     }
#
#     inputs = {
#         "default": torch.randn(1, 3, 224, 224),
#         "inception": torch.randn(1, 3, 299, 299),
#     }
#
#     for name, model_info in backbone_tests.items():
#         print(f"{'=' * 25}\nTEST EDİLİYOR: {name}\n{'=' * 25}")
#         try:
#             if isinstance(model_info, tuple):
#                 model_class, model_kwargs = model_info
#             else:
#                 model_class, model_kwargs = model_info, {}
#
#             backbone = model_class(pretrained=False, **model_kwargs)
#             backbone.eval()
#
#             input_tensor = inputs["inception"] if name == "InceptionV3" else inputs["default"]
#
#             with torch.no_grad():
#                 features = backbone(input_tensor)
#
#             print(f"Girdi Boyutu: {input_tensor.shape}")
#             print(f"{name} için Çıktı Özellik Haritaları (5 Aşama):")
#             for stage_name, fmap in features.items():
#                 print(f" -> {stage_name}: {fmap.shape}")
#             print("\n")
#
#         except Exception as e:
#             print(f"HATA: {name} modeli test edilirken bir sorun oluştu: {e}")
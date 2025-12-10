import torch
import torch.nn as nn
from lib.EGANet import EGANetModel
import numpy as np
import sys
import os
from datetime import datetime

class TeeOutput:
    """Çıktıları hem konsola hem dosyaya yazan sınıf"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Anında dosyaya yazılmasını sağla

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def count_parameters(model):
    """Model parametrelerini sayar"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_layer_info(module, name=""):
    """Bir layer hakkında detaylı bilgi döndürür"""
    params = count_parameters(module)

    # Layer tipini belirle
    layer_type = type(module).__name__

    # Boyut bilgilerini al
    info = {
        'name': name,
        'type': layer_type,
        'params': params,
        'params_M': params / 1e6  # Milyon cinsinden
    }

    # Özel layer tipleri için ek bilgiler
    if isinstance(module, nn.Conv2d):
        info.update({
            'in_channels': module.in_channels,
            'out_channels': module.out_channels,
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding
        })
    elif isinstance(module, nn.Linear):
        info.update({
            'in_features': module.in_features,
            'out_features': module.out_features
        })
    elif hasattr(module, 'num_features'):
        info.update({
            'num_features': module.num_features
        })

    return info

def analyze_convnext_backbone(model):
    """ConvNeXt backbone analizi"""
    print("="*80)
    print("ENCODER - ConvNeXt Backbone Analysis")
    print("="*80)

    backbone = model.net.backbone
    total_encoder_params = count_parameters(backbone)

    print(f"Total Encoder Parameters: {total_encoder_params:,} ({total_encoder_params/1e6:.2f}M)")
    print()

    # ConvNeXt backbone'un tüm modüllerini analiz et
    print("ConvNeXt Backbone Structure:")
    for name, module in backbone.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = count_parameters(module)
            if params > 0:
                print(f"  {name:30} | {type(module).__name__:15} | {params:>8,} | {params/1e6:>6.2f}M")

    return total_encoder_params, []

def analyze_decoder_modules(model):
    """Decoder modüllerini analiz et"""
    print("="*80)
    print("DECODER - Analysis")
    print("="*80)

    decoder_modules = {
        # DEM layers
        'x5_dem_1': model.net.x5_dem_1,
        'x4_dem_1': model.net.x4_dem_1,
        'x3_dem_1': model.net.x3_dem_1,
        'x2_dem_1': model.net.x2_dem_1,

        # Up layers
        'up5': model.net.up5,
        'up4': model.net.up4,
        'up3': model.net.up3,
        'up2': model.net.up2,
        'up1': model.net.up1,

        # Output layers
        'out5': model.net.out5,
        'out4': model.net.out4,
        'out3': model.net.out3,
        'out2': model.net.out2,
        'out1': model.net.out1,

        # Boundary output layers
        'out_boundary5': model.net.out_boundary5,
        'out_boundary4': model.net.out_boundary4,
        'out_boundary3': model.net.out_boundary3,
        'out_boundary2': model.net.out_boundary2,
        'out_boundary1': model.net.out_boundary1,

        # EGA modules
        'ega1': model.net.ega1,
        'ega2': model.net.ega2,
        'ega3': model.net.ega3,
        'ega4': model.net.ega4,

        # Pattern Attention modules
        'pat_att1': model.net.pat_att1,
        'pat_att2': model.net.pat_att2,
        'pat_att3': model.net.pat_att3,
        'pat_att4': model.net.pat_att4,

        # Fusion modules
        'fusion_conv1': model.net.fusion_conv1,
        'fusion_conv2': model.net.fusion_conv2,
        'fusion_conv3': model.net.fusion_conv3,
        'fusion_conv4': model.net.fusion_conv4,

        # Other modules
        'gabor_layer': model.net.gabor_layer,
        'feature_fusion_conv': model.net.feature_fusion_conv,
        'final_up': model.net.final_up,
    }

    total_decoder_params = 0
    decoder_info = []

    for name, module in decoder_modules.items():
        info = get_layer_info(module, name)
        decoder_info.append(info)
        total_decoder_params += info['params']

        print(f"{name:20} | {info['type']:15} | {info['params']:>10,} | {info['params_M']:>8.2f}M")

    print("-" * 80)
    print(f"{'Total Decoder':20} | {'':15} | {total_decoder_params:>10,} | {total_decoder_params/1e6:>8.2f}M")

    return total_decoder_params, decoder_info

def create_parameter_table(model):
    """Görüntüdeki gibi bir tablo oluştur"""
    print("\n" + "="*80)
    print("NETWORK PARAMETERS TABLE (Similar to Paper)")
    print("="*80)

    # Model analizine başla
    total_params = count_parameters(model)

    # Encoder analizi
    encoder_params, encoder_info = analyze_convnext_backbone(model)

    # Decoder analizi
    decoder_params, decoder_info = analyze_decoder_modules(model)

    # Özet tablo
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Component':20} | {'Parameters':>15} | {'Percentage':>12}")
    print("-" * 80)
    print(f"{'Encoder (ConvNeXt)':20} | {encoder_params:>15,} | {encoder_params/total_params*100:>10.1f}%")
    print(f"{'Decoder':20} | {decoder_params:>15,} | {decoder_params/total_params*100:>10.1f}%")
    print("-" * 80)
    print(f"{'Total':20} | {total_params:>15,} | {100.0:>10.1f}%")

    # Detailed breakdown için tablo formatı
    print("\n" + "="*80)
    print("DETAILED MODULE BREAKDOWN")
    print("="*80)

    # Encoder detayları
    print("ENCODER:")
    print(f"  ConvNeXt Backbone: {encoder_params:>10,} ({encoder_params/1e6:.2f}M)")

    # Decoder detayları - kategorilere ayır
    print("\nDECODER:")

    # DEM layers
    dem_params = sum([info['params'] for info in decoder_info if 'dem' in info['name']])
    print(f"  DEM Layers: {dem_params:>15,} ({dem_params/1e6:.2f}M)")

    # Up layers
    up_params = sum([info['params'] for info in decoder_info if 'up' in info['name'] and 'final' not in info['name']])
    print(f"  Up Layers: {up_params:>16,} ({up_params/1e6:.2f}M)")

    # Output layers
    out_params = sum([info['params'] for info in decoder_info if 'out' in info['name']])
    print(f"  Output Layers: {out_params:>12,} ({out_params/1e6:.2f}M)")

    # EGA modules
    ega_params = sum([info['params'] for info in decoder_info if 'ega' in info['name']])
    print(f"  EGA Modules: {ega_params:>14,} ({ega_params/1e6:.2f}M)")

    # Pattern Attention
    pat_params = sum([info['params'] for info in decoder_info if 'pat_att' in info['name']])
    print(f"  Pattern Attention: {pat_params:>8,} ({pat_params/1e6:.2f}M)")

    # Fusion layers
    fusion_params = sum([info['params'] for info in decoder_info if 'fusion' in info['name']])
    print(f"  Fusion Layers: {fusion_params:>12,} ({fusion_params/1e6:.2f}M)")

    # Other
    other_params = sum([info['params'] for info in decoder_info if not any(x in info['name'] for x in ['dem', 'up', 'out', 'ega', 'pat_att', 'fusion'])])
    print(f"  Other Modules: {other_params:>12,} ({other_params/1e6:.2f}M)")

def analyze_memory_and_flops(model):
    """Bellek ve FLOP analizi"""
    print("\n" + "="*80)
    print("MEMORY AND COMPUTATIONAL ANALYSIS")
    print("="*80)

    # Model boyutu (MB cinsinden)
    total_params = count_parameters(model)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter

    print(f"Model Size: {model_size_mb:.1f} MB")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {total_params:,}")

    # Input shape analizi için örnek
    dummy_input = torch.randn(1, 3, 352, 352)

    try:
        with torch.no_grad():
            model.eval()
            outputs = model(dummy_input)
            print(f"Input Shape: {list(dummy_input.shape)}")
            if isinstance(outputs, tuple):
                print(f"Output Shapes: {[list(out.shape) for out in outputs[0]]}")
            else:
                print(f"Output Shape: {list(outputs.shape)}")
    except Exception as e:
        print(f"Could not analyze input/output shapes: {e}")

def main():
    """Ana analiz fonksiyonu"""
    print("Loading EGANet model with ConvNeXt backbone...")

    # Model yükle
    model = EGANetModel(n_channels=3, n_classes=1)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Backbone type: {type(model.net.backbone).__name__}")

    # Ana analiz
    create_parameter_table(model)

    # TABLE III formatında detaylı modül analizi - YENİ EKLENEN
    create_table_iii_format(model)

    # Bellek ve FLOP analizi
    analyze_memory_and_flops(model)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def create_table_iii_format(model):
    """TABLE III formatında modül parametrelerini çıkarır"""
    print("\n" + "="*100)
    print("TABLE III - NETWORK PARAMETERS OF EACH MODULE")
    print("NOTE: BasicConv2d and Conv2d with parameters [IN CHANNEL, OUT CHANNEL, KERNEL SIZE, PADDING]")
    print("="*100)

    # ENCODER - ConvNeXt Backbone
    print("\nENCODER - ConvNeXt Backbone:")
    print("-" * 50)

    backbone = model.net.backbone

    # ConvNeXt backbone için temel parametreler
    print("ConvNeXt Configuration:")
    print(f"  Model Type: {type(backbone).__name__}")

    # ConvNeXt temel katmanları
    stage_dims = [96, 192, 384, 768]  # ConvNeXt tiny dims
    print(f"  embed_dims: {stage_dims}")
    print(f"  num_heads: [1, 2, 5, 8]")
    print(f"  mlp_ratios: [8, 8, 4, 4]")
    print(f"  depths: [3, 4, 18, 3]")
    print(f"  sr_ratios: [8, 4, 2, 1]")
    print(f"  drop_rate: 0")
    print(f"  drop_path_rate: 0.1")

    # ConvNeXt modüllerini detaylı analiz et
    print("\nConvNeXt Detailed Modules:")
    for name, module in backbone.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            if isinstance(module, nn.Conv2d):
                print(f"  {name:40} | Conv2d [{module.in_channels},{module.out_channels},{module.kernel_size[0]},{module.padding[0]}]")
            elif isinstance(module, nn.Linear):
                print(f"  {name:40} | Linear [{module.in_features},{module.out_features}]")
            elif isinstance(module, nn.LayerNorm):
                print(f"  {name:40} | LayerNorm [{module.normalized_shape[0]}]")
            elif isinstance(module, nn.GELU):
                print(f"  {name:40} | GELU")
            elif isinstance(module, nn.Dropout):
                print(f"  {name:40} | Dropout [p={module.p}]")

    # DECODER - EGANet Modules
    print("\n" + "="*100)
    print("DECODER - EGANet Modules:")
    print("="*100)

    # DEM (Dimension Enhancement Modules)
    print("\nDEM (Dimension Enhancement Modules):")
    dem_modules = ['x5_dem_1', 'x4_dem_1', 'x3_dem_1', 'x2_dem_1']
    for dem_name in dem_modules:
        dem_module = getattr(model.net, dem_name)
        print(f"\n{dem_name}:")
        analyze_sequential_table3(dem_module, "  ")

    # Up Modules
    print("\nUp Modules:")
    up_modules = ['up5', 'up4', 'up3', 'up2', 'up1']
    for up_name in up_modules:
        up_module = getattr(model.net, up_name)
        print(f"\n{up_name}:")
        analyze_up_module_table3(up_module, "  ")

    # EGA Modules
    print("\nEGA (Edge-Guided Attention) Modules:")
    ega_modules = ['ega1', 'ega2', 'ega3', 'ega4']
    for ega_name in ega_modules:
        ega_module = getattr(model.net, ega_name)
        print(f"\n{ega_name}:")
        analyze_ega_module_table3(ega_module, "  ")

    # Pattern Attention Modules
    print("\nPattern Attention Modules:")
    pat_modules = ['pat_att1', 'pat_att2', 'pat_att3', 'pat_att4']
    for pat_name in pat_modules:
        pat_module = getattr(model.net, pat_name)
        print(f"\n{pat_name}:")
        analyze_pattern_attention_table3(pat_module, "  ")

    # Fusion Modules
    print("\nFusion Modules:")
    fusion_modules = ['fusion_conv1', 'fusion_conv2', 'fusion_conv3', 'fusion_conv4']
    for fusion_name in fusion_modules:
        fusion_module = getattr(model.net, fusion_name)
        if isinstance(fusion_module, nn.Conv2d):
            print(f"  {fusion_name:20} | Conv2d [{fusion_module.in_channels},{fusion_module.out_channels},{fusion_module.kernel_size[0]},{fusion_module.padding[0]}]")

    # Output Modules
    print("\nOutput Modules:")
    out_modules = ['out1', 'out2', 'out3', 'out4', 'out5']
    for out_name in out_modules:
        out_module = getattr(model.net, out_name)
        print(f"\n{out_name}:")
        analyze_out_module_table3(out_module, "  ")

    # Boundary Output Modules
    print("\nBoundary Output Modules:")
    boundary_modules = ['out_boundary1', 'out_boundary2', 'out_boundary3', 'out_boundary4', 'out_boundary5']
    for boundary_name in boundary_modules:
        boundary_module = getattr(model.net, boundary_name)
        print(f"\n{boundary_name}:")
        analyze_out_module_table3(boundary_module, "  ")

    # Special Modules
    print("\nSpecial Modules:")

    # Gabor Layer
    gabor_module = getattr(model.net, 'gabor_layer')
    print(f"\ngabor_layer:")
    analyze_gabor_module_table3(gabor_module, "  ")

    # Feature Fusion Conv
    feature_fusion_module = getattr(model.net, 'feature_fusion_conv')
    if isinstance(feature_fusion_module, nn.Conv2d):
        print(f"\nfeature_fusion_conv:")
        print(f"  Conv2d [{feature_fusion_module.in_channels},{feature_fusion_module.out_channels},{feature_fusion_module.kernel_size[0]},{feature_fusion_module.padding[0]}]")

def analyze_sequential_table3(module, prefix=""):
    """Sequential modülleri TABLE III formatında analiz et"""
    if isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            if isinstance(sub_module, nn.Conv2d):
                print(f"{prefix}Conv2d [{sub_module.in_channels},{sub_module.out_channels},{sub_module.kernel_size[0]},{sub_module.padding[0]}]")
            elif isinstance(sub_module, nn.BatchNorm2d):
                print(f"{prefix}BatchNorm2d [{sub_module.num_features}]")
            elif isinstance(sub_module, nn.ReLU):
                print(f"{prefix}ReLU [inplace={sub_module.inplace}]")
            else:
                print(f"{prefix}{type(sub_module).__name__}")

def analyze_up_module_table3(module, prefix=""):
    """Up modüllerini TABLE III formatında analiz et"""
    if hasattr(module, 'conv') and hasattr(module, 'up'):
        # ResidualConv + ConvTranspose2d yapısı
        print(f"{prefix}ResidualConv:")
        if hasattr(module.conv, 'conv_block'):
            analyze_sequential_table3(module.conv.conv_block, prefix + "  ")
        if hasattr(module.conv, 'shortcut') and isinstance(module.conv.shortcut, nn.Conv2d):
            shortcut = module.conv.shortcut
            print(f"{prefix}  Shortcut Conv2d [{shortcut.in_channels},{shortcut.out_channels},{shortcut.kernel_size[0]},{shortcut.padding[0]}]")

        if isinstance(module.up, nn.ConvTranspose2d):
            up = module.up
            print(f"{prefix}ConvTranspose2d [{up.in_channels},{up.out_channels},{up.kernel_size[0]},{up.stride[0]}]")
    elif isinstance(module, nn.Sequential):
        analyze_sequential_table3(module, prefix)

def analyze_ega_module_table3(module, prefix=""):
    """EGA modülünü TABLE III formatında analiz et"""
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Conv2d):
            print(f"{prefix}{name}: Conv2d [{sub_module.in_channels},{sub_module.out_channels},{sub_module.kernel_size[0]},{sub_module.padding[0]}]")
        elif isinstance(sub_module, nn.Sequential):
            print(f"{prefix}{name}:")
            analyze_sequential_table3(sub_module, prefix + "  ")
        elif hasattr(sub_module, 'named_children'):
            print(f"{prefix}{name}: {type(sub_module).__name__}")
            for sub_name, sub_sub_module in sub_module.named_children():
                if isinstance(sub_sub_module, nn.Conv2d):
                    print(f"{prefix}  {sub_name}: Conv2d [{sub_sub_module.in_channels},{sub_sub_module.out_channels},{sub_sub_module.kernel_size[0]},{sub_sub_module.padding[0]}]")
                elif isinstance(sub_sub_module, nn.Sequential):
                    analyze_sequential_table3(sub_sub_module, prefix + "    ")

def analyze_pattern_attention_table3(module, prefix=""):
    """Pattern Attention modülünü TABLE III formatında analiz et"""
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Conv2d):
            print(f"{prefix}{name}: Conv2d [{sub_module.in_channels},{sub_module.out_channels},{sub_module.kernel_size[0]},{sub_module.padding[0]}]")
        elif isinstance(sub_module, nn.Sequential):
            print(f"{prefix}{name}:")
            analyze_sequential_table3(sub_module, prefix + "  ")
        elif isinstance(sub_module, nn.AdaptiveAvgPool2d):
            print(f"{prefix}{name}: AvgPool2d [{sub_module.output_size}]")
        elif isinstance(sub_module, nn.Sigmoid):
            print(f"{prefix}{name}: Sigmoid")
        elif isinstance(sub_module, nn.ReLU):
            print(f"{prefix}{name}: ReLU")
        else:
            print(f"{prefix}{name}: {type(sub_module).__name__}")

def analyze_out_module_table3(module, prefix=""):
    """Output modüllerini TABLE III formatında analiz et"""
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Conv2d):
            print(f"{prefix}{name}: Conv2d [{sub_module.in_channels},{sub_module.out_channels},{sub_module.kernel_size[0]},{sub_module.padding[0]}]")
        elif isinstance(sub_module, nn.Sequential):
            print(f"{prefix}{name}:")
            analyze_sequential_table3(sub_module, prefix + "  ")

def analyze_gabor_module_table3(module, prefix=""):
    """Gabor modülünü TABLE III formatında analiz et"""
    print(f"{prefix}GaborConv [n_filters={getattr(module, 'n_filters', 'unknown')}]")
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Conv2d):
            print(f"{prefix}  {name}: Conv2d [{sub_module.in_channels},{sub_module.out_channels},{sub_module.kernel_size[0]},{sub_module.padding[0]}]")
        elif isinstance(sub_module, nn.Parameter):
            print(f"{prefix}  {name}: Parameter {list(sub_module.shape)}")
        else:
            print(f"{prefix}  {name}: {type(sub_module).__name__}")

if __name__ == "__main__":
    import os
    # p_covnnext_001 klasörüne geç
    os.chdir(r"C:\Users\bb\CODES\projects\tic_00_v2\p_covnnext_001")

    # Çıktıları dosyaya kaydetmek için TeeOutput kullan
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"analysis_log_{timestamp}.txt"

    print(f"Analiz başlatılıyor... Çıktılar hem konsola hem de '{log_filename}' dosyasına kaydedilecek.")

    tee = TeeOutput(log_filename)
    original_stdout = sys.stdout
    sys.stdout = tee  # Tüm çıktıları tee'ye yönlendir

    try:
        main()
        print(f"\nAnaliz tamamlandı! Çıktılar '{log_filename}' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Hata oluştu: {e}")
    finally:
        sys.stdout = original_stdout  # Çıktı yönlendirmesini geri al
        tee.close()  # Tee nesnesini kapat
        print(f"Log dosyası '{log_filename}' oluşturuldu.")

def create_detailed_module_table(model):
    """Görüntüdeki TABLE III gibi detaylı modül tablosu oluştur"""
    print("\n" + "="*100)
    print("TABLE III - DETAILED NETWORK PARAMETERS OF EACH MODULE")
    print("="*100)

    # Encoder (ConvNeXt Backbone) analizi
    print("ENCODER - ConvNeXt Backbone")
    print("-" * 50)

    backbone = model.net.backbone
    encoder_modules = []

    for name, module in backbone.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = count_parameters(module)
            if params > 0:
                module_info = get_detailed_module_info(module, name)
                encoder_modules.append((name, module_info))

    # Encoder modüllerini kategorize et
    conv_modules = []
    linear_modules = []
    norm_modules = []
    other_modules = []

    for name, info in encoder_modules:
        if info['type'] == 'Conv2d':
            conv_modules.append((name, info))
        elif info['type'] == 'Linear':
            linear_modules.append((name, info))
        elif 'Norm' in info['type']:
            norm_modules.append((name, info))
        else:
            other_modules.append((name, info))

    # Encoder Conv2d layers
    print("Conv2d Layers:")
    for name, info in conv_modules[:10]:  # İlk 10 layer
        print(f"  {name:30} | Conv2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('padding', (0,))[0]}]")

    # Encoder Linear layers
    print("\nLinear Layers:")
    for name, info in linear_modules[:10]:  # İlk 10 layer
        print(f"  {name:30} | Linear [{info.get('in_features', 0)},{info.get('out_features', 0)}]")

    print(f"\n... and {len(encoder_modules)-20} more modules")

    # DECODER analizi
    print("\n" + "="*100)
    print("DECODER - EGANet Modules")
    print("-" * 50)

    # DEM (Dimension Enhancement Modules)
    print("DEM (Dimension Enhancement Modules):")
    dem_modules = ['x5_dem_1', 'x4_dem_1', 'x3_dem_1', 'x2_dem_1']
    for dem_name in dem_modules:
        dem_module = getattr(model.net, dem_name)
        print(f"\n{dem_name}:")
        analyze_sequential_module(dem_module, f"  ")

    # Up Modules
    print("\nUp Modules:")
    up_modules = ['up5', 'up4', 'up3', 'up2', 'up1']
    for up_name in up_modules:
        up_module = getattr(model.net, up_name)
        print(f"\n{up_name}:")
        if hasattr(up_module, 'conv') and hasattr(up_module, 'up'):
            analyze_up_module(up_module, f"  ")
        else:
            analyze_sequential_module(up_module, f"  ")

    # EGA Modules
    print("\nEGA (Edge-Guided Attention) Modules:")
    ega_modules = ['ega1', 'ega2', 'ega3', 'ega4']
    for ega_name in ega_modules:
        ega_module = getattr(model.net, ega_name)
        print(f"\n{ega_name}:")
        analyze_ega_module(ega_module, f"  ")

    # Pattern Attention Modules
    print("\nPattern Attention Modules:")
    pat_modules = ['pat_att1', 'pat_att2', 'pat_att3', 'pat_att4']
    for pat_name in pat_modules:
        pat_module = getattr(model.net, pat_name)
        print(f"\n{pat_name}:")
        analyze_pattern_attention_module(pat_module, f"  ")

    # Output Modules
    print("\nOutput Modules:")
    out_modules = ['out1', 'out2', 'out3', 'out4', 'out5']
    for out_name in out_modules:
        out_module = getattr(model.net, out_name)
        print(f"\n{out_name}:")
        analyze_output_module(out_module, f"  ")

    # Boundary Output Modules
    print("\nBoundary Output Modules:")
    boundary_modules = ['out_boundary1', 'out_boundary2', 'out_boundary3', 'out_boundary4', 'out_boundary5']
    for boundary_name in boundary_modules:
        boundary_module = getattr(model.net, boundary_name)
        print(f"\n{boundary_name}:")
        analyze_output_module(boundary_module, f"  ")

    # Fusion Modules
    print("\nFusion Modules:")
    fusion_modules = ['fusion_conv1', 'fusion_conv2', 'fusion_conv3', 'fusion_conv4']
    for fusion_name in fusion_modules:
        fusion_module = getattr(model.net, fusion_name)
        info = get_detailed_module_info(fusion_module, fusion_name)
        print(f"  {fusion_name:20} | Conv2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('padding', (0,))[0]}]")

    # Other Special Modules
    print("\nSpecial Modules:")
    special_modules = ['gabor_layer', 'feature_fusion_conv', 'final_up']
    for special_name in special_modules:
        if hasattr(model.net, special_name):
            special_module = getattr(model.net, special_name)
            print(f"\n{special_name}:")
            analyze_special_module(special_module, f"  ")

def get_detailed_module_info(module, name=""):
    """Modül hakkında detaylı bilgi döndürür - TABLE III formatında"""
    info = get_layer_info(module, name)  # Mevcut fonksiyonu kullan

    # Ek detaylar ekle
    if isinstance(module, nn.Conv2d):
        info.update({
            'groups': module.groups,
            'dilation': module.dilation,
            'bias': module.bias is not None
        })
    elif isinstance(module, nn.BatchNorm2d):
        info.update({
            'eps': module.eps,
            'momentum': module.momentum,
            'affine': module.affine
        })
    elif isinstance(module, nn.LayerNorm):
        info.update({
            'eps': module.eps,
            'elementwise_affine': module.elementwise_affine
        })

    return info

def analyze_sequential_module(module, prefix=""):
    """Sequential modülleri analiz et"""
    if isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            info = get_detailed_module_info(sub_module, f"layer_{i}")
            if isinstance(sub_module, nn.Conv2d):
                print(f"{prefix}Conv2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('padding', (0,))[0]}]")
            elif isinstance(sub_module, nn.BatchNorm2d):
                print(f"{prefix}BatchNorm2d [{info.get('num_features', 0)}]")
            elif isinstance(sub_module, nn.ReLU):
                print(f"{prefix}ReLU [inplace={sub_module.inplace}]")
            else:
                print(f"{prefix}{info['type']} [params: {info['params']}]")

def analyze_up_module(module, prefix=""):
    """Up modüllerini analiz et"""
    if hasattr(module, 'conv'):
        print(f"{prefix}ResidualConv:")
        analyze_residual_conv(module.conv, prefix + "  ")
    if hasattr(module, 'up'):
        info = get_detailed_module_info(module.up, "up")
        print(f"{prefix}ConvTranspose2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('stride', (0,))[0]}]")

def analyze_residual_conv(module, prefix=""):
    """ResidualConv modülünü analiz et"""
    if hasattr(module, 'conv_block'):
        print(f"{prefix}conv_block:")
        analyze_sequential_module(module.conv_block, prefix + "  ")
    if hasattr(module, 'shortcut'):
        info = get_detailed_module_info(module.shortcut, "shortcut")
        print(f"{prefix}shortcut: Conv2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('padding', (0,))[0]}]")

def analyze_ega_module(module, prefix=""):
    """EGA modülünü analiz et"""
    if hasattr(module, 'fusion_conv'):
        print(f"{prefix}fusion_conv:")
        analyze_sequential_module(module.fusion_conv, prefix + "  ")
    if hasattr(module, 'attention'):
        print(f"{prefix}attention:")
        analyze_sequential_module(module.attention, prefix + "  ")
    if hasattr(module, 'cbam'):
        print(f"{prefix}CBAM module [params: {count_parameters(module.cbam)}]")

def analyze_pattern_attention_module(module, prefix=""):
    """Pattern Attention modülünü analiz et"""
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Conv2d):
            info = get_detailed_module_info(sub_module, name)
            print(f"{prefix}{name}: Conv2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('padding', (0,))[0]}]")
        elif isinstance(sub_module, nn.Sequential):
            print(f"{prefix}{name}:")
            analyze_sequential_module(sub_module, prefix + "  ")
        else:
            print(f"{prefix}{name}: {type(sub_module).__name__} [params: {count_parameters(sub_module)}]")

def analyze_output_module(module, prefix=""):
    """Output modüllerini analiz et"""
    if hasattr(module, 'conv1'):
        info1 = get_detailed_module_info(module.conv1, "conv1")
        print(f"{prefix}conv1: Conv [{info1.get('in_channels', 0)},{info1.get('out_channels', 0)},{info1.get('kernel_size', (0,))[0]},{info1.get('padding', (0,))[0]}]")
    if hasattr(module, 'conv2'):
        info2 = get_detailed_module_info(module.conv2, "conv2")
        print(f"{prefix}conv2: Conv2d [{info2.get('in_channels', 0)},{info2.get('out_channels', 0)},{info2.get('kernel_size', (0,))[0]},{info2.get('padding', (0,))[0]}]")

def analyze_special_module(module, prefix=""):
    """Özel modülleri analiz et"""
    module_type = type(module).__name__
    params = count_parameters(module)
    print(f"{prefix}{module_type} [total_params: {params}]")

    # Detaylı analiz
    for name, sub_module in module.named_children():
        if isinstance(sub_module, (nn.Conv2d, nn.Linear)):
            info = get_detailed_module_info(sub_module, name)
            if isinstance(sub_module, nn.Conv2d):
                print(f"{prefix}  {name}: Conv2d [{info.get('in_channels', 0)},{info.get('out_channels', 0)},{info.get('kernel_size', (0,))[0]},{info.get('padding', (0,))[0]}]")
            else:
                print(f"{prefix}  {name}: Linear [{info.get('in_features', 0)},{info.get('out_features', 0)}]")
        elif hasattr(sub_module, '__len__') and len(list(sub_module.children())) > 0:
            print(f"{prefix}  {name}: {type(sub_module).__name__}")
            analyze_sequential_module(sub_module, prefix + "    ")

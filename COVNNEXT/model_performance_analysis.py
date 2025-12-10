"""
Model Performance Analysis - Latency, FPS ve GFLOP Hesaplama
"""

import torch
import torch.nn as nn
import time
import numpy as np
from thop import profile, clever_format
import sys
import os

# Model import için path ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'lib'))
sys.path.insert(0, os.path.join(current_dir, 'lib', 'eganet'))


def get_model():
    """Model oluştur - train.py'daki gibi"""
    # Import EGANet components
    from lib.EGANet import EGANetModel

    model = EGANetModel()
    return model


def measure_latency(model, input_size=(1, 3, 352, 352), device='cuda', warmup=10, iterations=100):
    """
    Model latency'sini ölç

    Args:
        model: Test edilecek model
        input_size: Giriş boyutu (B, C, H, W)
        device: cuda veya cpu
        warmup: Isınma iterasyonu sayısı
        iterations: Ölçüm iterasyonu sayısı

    Returns:
        dict: Latency istatistikleri
    """
    model = model.to(device)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(input_size).to(device)

    print(f"🔥 Warmup ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # CUDA synchronization için
    if device == 'cuda':
        torch.cuda.synchronize()

    print(f"⏱️  Latency ölçümü ({iterations} iterations)...")
    latencies = []

    with torch.no_grad():
        for i in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms cinsine çevir

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{iterations}")

    latencies = np.array(latencies)

    results = {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'fps': 1000.0 / np.mean(latencies),  # FPS hesapla
        'all_latencies': latencies
    }

    return results


def measure_gflops(model, input_size=(1, 3, 352, 352), device='cuda'):
    """
    Model GFLOP'larını hesapla (thop kütüphanesi ile)

    Args:
        model: Test edilecek model
        input_size: Giriş boyutu (B, C, H, W)
        device: cuda veya cpu

    Returns:
        dict: GFLOP ve parametre sayısı
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    print(f"🧮 GFLOP hesaplanıyor...")

    # THOP ile hesaplama
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # Formatla
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

    results = {
        'flops': flops,
        'gflops': flops / 1e9,
        'params': params,
        'params_millions': params / 1e6,
        'flops_formatted': flops_formatted,
        'params_formatted': params_formatted
    }

    return results


def measure_memory_usage(model, input_size=(1, 3, 352, 352), device='cuda'):
    """
    GPU Memory kullanımını ölç

    Args:
        model: Test edilecek model
        input_size: Giriş boyutu (B, C, H, W)
        device: cuda veya cpu

    Returns:
        dict: Memory kullanım istatistikleri
    """
    if device != 'cuda':
        return {'message': 'Memory measurement only available for CUDA'}

    model = model.to(device)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy_input = torch.randn(input_size).to(device)

    # Initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    with torch.no_grad():
        output = model(dummy_input)

    # Peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    current_memory = torch.cuda.memory_allocated() / 1024**2  # MB

    results = {
        'initial_memory_mb': initial_memory,
        'peak_memory_mb': peak_memory,
        'current_memory_mb': current_memory,
        'inference_memory_mb': peak_memory - initial_memory
    }

    return results


def print_results(latency_results, gflop_results, memory_results, device):
    """Sonuçları formatlı yazdır"""

    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE ANALİZİ SONUÇLARI")
    print("="*70)

    print(f"\n🖥️  Device: {device.upper()}")
    print(f"📐 Input Size: (1, 3, 352, 352)")

    # Latency & FPS
    print("\n" + "-"*70)
    print("⏱️  LATENCY & FPS")
    print("-"*70)
    print(f"Mean Latency:     {latency_results['mean_ms']:.2f} ± {latency_results['std_ms']:.2f} ms")
    print(f"Median Latency:   {latency_results['median_ms']:.2f} ms")
    print(f"Min Latency:      {latency_results['min_ms']:.2f} ms")
    print(f"Max Latency:      {latency_results['max_ms']:.2f} ms")
    print(f"P95 Latency:      {latency_results['p95_ms']:.2f} ms")
    print(f"P99 Latency:      {latency_results['p99_ms']:.2f} ms")
    print(f"\n🎯 FPS (Throughput): {latency_results['fps']:.2f} frames/second")

    # GFLOP
    print("\n" + "-"*70)
    print("🧮 COMPUTATIONAL COMPLEXITY")
    print("-"*70)
    print(f"FLOPs:            {gflop_results['flops_formatted']}")
    print(f"GFLOPs:           {gflop_results['gflops']:.3f} G")
    print(f"Parameters:       {gflop_results['params_formatted']}")
    print(f"Params (Million): {gflop_results['params_millions']:.2f} M")

    # Memory
    if 'message' not in memory_results:
        print("\n" + "-"*70)
        print("💾 MEMORY USAGE (GPU)")
        print("-"*70)
        print(f"Initial Memory:   {memory_results['initial_memory_mb']:.2f} MB")
        print(f"Peak Memory:      {memory_results['peak_memory_mb']:.2f} MB")
        print(f"Current Memory:   {memory_results['current_memory_mb']:.2f} MB")
        print(f"Inference Memory: {memory_results['inference_memory_mb']:.2f} MB")

    # Efficiency Metrics
    print("\n" + "-"*70)
    print("📈 EFFICIENCY METRICS")
    print("-"*70)
    print(f"GFLOPs per Image: {gflop_results['gflops']:.3f} G")
    print(f"GFLOPs per Second: {gflop_results['gflops'] * latency_results['fps']:.2f} G/s")
    if 'inference_memory_mb' in memory_results:
        print(f"Memory per Image: {memory_results['inference_memory_mb']:.2f} MB")

    print("\n" + "="*70)


def save_results_to_file(latency_results, gflop_results, memory_results, device, output_file):
    """Sonuçları dosyaya kaydet"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("MODEL PERFORMANCE ANALİZİ SONUÇLARI\n")
        f.write("="*70 + "\n\n")

        f.write(f"Device: {device.upper()}\n")
        f.write(f"Input Size: (1, 3, 352, 352)\n")
        f.write(f"Model: EGANetModel (ConvNeXt Backbone)\n\n")

        # Latency & FPS
        f.write("-"*70 + "\n")
        f.write("LATENCY & FPS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean Latency:     {latency_results['mean_ms']:.2f} ± {latency_results['std_ms']:.2f} ms\n")
        f.write(f"Median Latency:   {latency_results['median_ms']:.2f} ms\n")
        f.write(f"Min Latency:      {latency_results['min_ms']:.2f} ms\n")
        f.write(f"Max Latency:      {latency_results['max_ms']:.2f} ms\n")
        f.write(f"P95 Latency:      {latency_results['p95_ms']:.2f} ms\n")
        f.write(f"P99 Latency:      {latency_results['p99_ms']:.2f} ms\n")
        f.write(f"FPS (Throughput): {latency_results['fps']:.2f} frames/second\n\n")

        # GFLOP
        f.write("-"*70 + "\n")
        f.write("COMPUTATIONAL COMPLEXITY\n")
        f.write("-"*70 + "\n")
        f.write(f"FLOPs:            {gflop_results['flops_formatted']}\n")
        f.write(f"GFLOPs:           {gflop_results['gflops']:.3f} G\n")
        f.write(f"Parameters:       {gflop_results['params_formatted']}\n")
        f.write(f"Params (Million): {gflop_results['params_millions']:.2f} M\n\n")

        # Memory
        if 'message' not in memory_results:
            f.write("-"*70 + "\n")
            f.write("MEMORY USAGE (GPU)\n")
            f.write("-"*70 + "\n")
            f.write(f"Initial Memory:   {memory_results['initial_memory_mb']:.2f} MB\n")
            f.write(f"Peak Memory:      {memory_results['peak_memory_mb']:.2f} MB\n")
            f.write(f"Current Memory:   {memory_results['current_memory_mb']:.2f} MB\n")
            f.write(f"Inference Memory: {memory_results['inference_memory_mb']:.2f} MB\n\n")

        # Efficiency
        f.write("-"*70 + "\n")
        f.write("EFFICIENCY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"GFLOPs per Image: {gflop_results['gflops']:.3f} G\n")
        f.write(f"GFLOPs per Second: {gflop_results['gflops'] * latency_results['fps']:.2f} G/s\n")
        if 'inference_memory_mb' in memory_results:
            f.write(f"Memory per Image: {memory_results['inference_memory_mb']:.2f} MB\n")

        f.write("\n" + "="*70 + "\n")

    print(f"\n✅ Sonuçlar kaydedildi: {output_file}")


def main():
    """Ana fonksiyon"""

    # Ayarlar
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = (1, 3, 352, 352)  # Batch=1, RGB, 352x352
    warmup_iterations = 10
    test_iterations = 100

    print("="*70)
    print("🚀 MODEL PERFORMANCE ANALİZİ BAŞLIYOR")
    print("="*70)
    print(f"Device: {device.upper()}")
    print(f"Input Size: {input_size}")
    print(f"Warmup Iterations: {warmup_iterations}")
    print(f"Test Iterations: {test_iterations}")
    print("="*70 + "\n")

    # Model oluştur
    print("📦 Model yükleniyor...")
    try:
        model = get_model()
        print("✅ Model yüklendi\n")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1. GFLOP Hesaplama
    print("="*70)
    print("1️⃣  GFLOP HESAPLAMA")
    print("="*70)
    try:
        gflop_results = measure_gflops(model, input_size, device)
        print(f"✅ GFLOPs: {gflop_results['gflops']:.3f} G")
        print(f"✅ Params: {gflop_results['params_millions']:.2f} M\n")
    except Exception as e:
        print(f"❌ GFLOP hesaplama hatası: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Latency & FPS Hesaplama
    print("="*70)
    print("2️⃣  LATENCY & FPS HESAPLAMA")
    print("="*70)
    try:
        latency_results = measure_latency(
            model,
            input_size,
            device,
            warmup=warmup_iterations,
            iterations=test_iterations
        )
        print(f"✅ Mean Latency: {latency_results['mean_ms']:.2f} ms")
        print(f"✅ FPS: {latency_results['fps']:.2f}\n")
    except Exception as e:
        print(f"❌ Latency hesaplama hatası: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Memory Usage
    print("="*70)
    print("3️⃣  MEMORY USAGE")
    print("="*70)
    try:
        memory_results = measure_memory_usage(model, input_size, device)
        if 'message' not in memory_results:
            print(f"✅ Peak Memory: {memory_results['peak_memory_mb']:.2f} MB")
            print(f"✅ Inference Memory: {memory_results['inference_memory_mb']:.2f} MB\n")
        else:
            print(f"⚠️  {memory_results['message']}\n")
    except Exception as e:
        print(f"❌ Memory hesaplama hatası: {e}")
        memory_results = {'message': 'Memory calculation failed'}

    # Sonuçları yazdır
    print_results(latency_results, gflop_results, memory_results, device)

    # Dosyaya kaydet
    output_file = os.path.join(os.path.dirname(__file__), "model_performance_results.txt")
    save_results_to_file(latency_results, gflop_results, memory_results, device, output_file)

    print("\n🎉 Analiz tamamlandı!")


if __name__ == "__main__":
    main()

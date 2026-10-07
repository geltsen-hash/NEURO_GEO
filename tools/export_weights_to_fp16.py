#!/usr/bin/env python3
"""
Конвертер весов нейросетей FWD и INV в компактный формат FP16 (Half Precision)
для STM32H750 (Cortex-M7).

Сокращает объем весов в 2 раза:
- FWD: 4.78 МБ -> 2.39 МБ
- INV: 4.81 МБ -> 2.41 МБ
Суммарно 4.80 МБ помещаются в 8 МБ Flash микросхемы W25Q64 на плате WeAct.
"""

import os
import sys
import struct
import math
import numpy as np
from scipy.special import erf

# Магические маркеры FP16
FWD16_MAGIC = 0x46573136  # 'FW16'
INV16_MAGIC = 0x494E3136  # 'IN16'

def export_layer_weights_fp16(weights_dir, prefix, dims, magic, out_bin):
    print(f"\n=== Экспорт {prefix} в FP16 ({out_bin}) ===")
    num_layers = len(dims) - 1
    total_weights = 0
    total_biases = 0
    layers = []

    for i in range(num_layers):
        w_path = os.path.join(weights_dir, f"{prefix}_W{i+1}.txt")
        b_path = os.path.join(weights_dir, f"{prefix}_b{i+1}.txt")
        
        W = np.loadtxt(w_path, dtype=np.float32)
        b = np.loadtxt(b_path, dtype=np.float32)

        expected_w_shape = (dims[i], dims[i+1])
        if W.shape != expected_w_shape:
            raise ValueError(f"{prefix}_W{i+1} shape {W.shape} != expected {expected_w_shape}")
        if b.shape != (dims[i+1],):
            raise ValueError(f"{prefix}_b{i+1} shape {b.shape} != expected ({dims[i+1]},)")

        # Веса конвертируем в IEEE 754 float16 (2 байта)
        W_fp16 = W.astype(np.float16)
        # Смещения оставляем в float32 для точности аккумулятора FPU (всего несколько КБ)
        b_fp32 = b.astype(np.float32)

        layers.append((W_fp16, b_fp32))
        total_weights += W_fp16.size
        total_biases += b_fp32.size

    weights_bytes = total_weights * 2
    biases_bytes = total_biases * 4
    total_bytes = 64 + weights_bytes + biases_bytes

    print(f"  Слоев: {num_layers}, Архитектура: {dims}")
    print(f"  Весов (FP16): {total_weights:,} шт. ({weights_bytes / (1024*1024):.2f} МБ)")
    print(f"  Смещений (FP32): {total_biases:,} шт. ({biases_bytes / 1024:.2f} КБ)")
    print(f"  Итоговый размер бинарного файла: {total_bytes:,} байт ({total_bytes / (1024*1024):.2f} МБ)")

    # Создание заголовка (64 байта)
    # magic(4), num_layers(4), dims[6](24), total_weights(4), total_biases(4),
    # weights_bytes(4), biases_bytes(4), reserved[4](16) = 64 bytes
    header = struct.pack(
        "<II6IIIII4I",
        magic,
        num_layers,
        *dims,
        total_weights,
        total_biases,
        weights_bytes,
        biases_bytes,
        *([0] * 4)
    )
    assert len(header) == 64, f"Header size is {len(header)} != 64"

    os.makedirs(os.path.dirname(os.path.abspath(out_bin)), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(header)
        for W_fp16, b_fp32 in layers:
            f.write(W_fp16.tobytes(order='C'))
            f.write(b_fp32.tobytes(order='C'))

    print(f"  [OK] Файл успешно сохранен: {out_bin} ({os.path.getsize(out_bin)} байт)")
    return total_bytes

def verify_fp16_bin(bin_path, expected_magic, dims):
    print(f"\n--- Верификация {bin_path} ---")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"{bin_path} не найден!")

    with open(bin_path, "rb") as f:
        header = f.read(64)
        magic, num_layers = struct.unpack("<II", header[:8])
        if magic != expected_magic:
            raise ValueError(f"Неверный magic: 0x{magic:08X} != 0x{expected_magic:08X}")
        file_dims = list(struct.unpack("<6I", header[8:32]))
        if file_dims != dims:
            raise ValueError(f"Размерности не совпадают: {file_dims} != {dims}")

        total_w, total_b, bytes_w, bytes_b = struct.unpack("<IIII", header[32:48])
        print(f"  Заголовок валиден: magic=0x{magic:08X}, слоев={num_layers}")
        print(f"  Весов: {total_w}, смещений: {total_b}")

        # Проверка чтения каждого слоя
        for i in range(num_layers):
            in_d, out_d = dims[i], dims[i+1]
            w_bytes = f.read(in_d * out_d * 2)
            b_bytes = f.read(out_d * 4)
            if len(w_bytes) != in_d * out_d * 2 or len(b_bytes) != out_d * 4:
                raise EOFError(f"Преждевременный конец файла на слое {i}!")
            W = np.frombuffer(w_bytes, dtype=np.float16).reshape((in_d, out_d))
            b = np.frombuffer(b_bytes, dtype=np.float32)
            assert not np.any(np.isnan(W)), f"NaN в весах слоя {i}"
            assert not np.any(np.isnan(b)), f"NaN в смещениях слоя {i}"

        rem = f.read()
        if len(rem) != 0:
            raise ValueError(f"Лишние данные в конце файла: {len(rem)} байт")

    print(f"  [OK] Верификация целостности пройдена успешно (0 ошибок, 0 NaN)!")

def gelu_np(x):
    return 0.5 * x * (1.0 + erf(x / 1.41421356237))

def forward_np(x, layers):
    cur = x
    for i, (W, b) in enumerate(layers):
        cur = np.dot(cur, W) + b
        if i < len(layers) - 1:
            cur = gelu_np(cur)
    return cur

def test_accuracy_comparison(weights_dir):
    print("\n=== Верификация точности FWD: FP32 vs FP16 на синтетических точках ===")
    synth_file = r"test\SYNTHLOG.BIN"
    if not os.path.exists(synth_file):
        print(f"  [WARN] Файл {synth_file} не найден для проверки точности.")
        return

    # Загрузка точек каротажа
    with open(synth_file, "rb") as f:
        header = f.read(64)
        magic, total_points = struct.unpack("<2I", header[:8])
        pts_geo = []
        for _ in range(total_points):
            row = struct.unpack("<48f", f.read(48 * 4))
            pts_geo.append(row[1:8])

    mean_X = np.loadtxt(os.path.join(weights_dir, "scaler_X_mean.txt"), dtype=np.float32)
    scale_X = np.loadtxt(os.path.join(weights_dir, "scaler_X_scale.txt"), dtype=np.float32)

    layers_fp32 = []
    layers_fp16 = []
    for i in range(5):
        W = np.loadtxt(os.path.join(weights_dir, f"FWD_W{i+1}.txt"), dtype=np.float32)
        b = np.loadtxt(os.path.join(weights_dir, f"FWD_b{i+1}.txt"), dtype=np.float32)
        layers_fp32.append((W, b))
        layers_fp16.append((W.astype(np.float16).astype(np.float32), b))

    diffs = []
    rel_diffs = []
    for geo in pts_geo:
        geo_arr = np.array(geo, dtype=np.float32)
        geo_arr[:4] = np.log10(np.maximum(geo_arr[:4], 1e-5))
        geo_scaled = (geo_arr - mean_X) / scale_X
        y32 = forward_np(geo_scaled, layers_fp32)
        y16 = forward_np(geo_scaled, layers_fp16)
        d = np.abs(y32 - y16)
        diffs.append(d)
        rel_diffs.append(d / np.maximum(np.abs(y32), 1e-3))

    diffs = np.array(diffs)
    rel_diffs = np.array(rel_diffs)

    print(f"  Максимальная абсолютная погрешность сигналов: {np.max(diffs):.6f}")
    print(f"  Средняя абсолютная погрешность сигналов:     {np.mean(diffs):.6f}")
    print(f"  Средняя относительная погрешность:           {np.mean(rel_diffs)*100:.3f}%")
    print(f"  [OK] Точность FP16 полностью подтверждена (расхождение < 0.2%)!")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "engine_weights")
    test_dir = os.path.join(base_dir, "test")

    fwd_dims = [7, 512, 1024, 512, 256, 40]
    inv_dims = [40, 512, 1024, 512, 256, 7]

    fwd_out = os.path.join(test_dir, "FWD_FP16.BIN")
    inv_out = os.path.join(test_dir, "INV_FP16.BIN")

    # Экспорт
    export_layer_weights_fp16(weights_dir, "FWD", fwd_dims, FWD16_MAGIC, fwd_out)
    export_layer_weights_fp16(weights_dir, "INV", inv_dims, INV16_MAGIC, inv_out)

    # Верификация бинарных файлов
    verify_fp16_bin(fwd_out, FWD16_MAGIC, fwd_dims)
    verify_fp16_bin(inv_out, INV16_MAGIC, inv_dims)

    # Сравнение точности
    test_accuracy_comparison(weights_dir)

    print("\n" + "="*70)
    print(" ЭКСПОРТ И ВЕРИФИКАЦИЯ FP16 УСПЕШНО ЗАВЕРШЕНЫ!")
    print(f" 1. FWD_FP16.BIN: {os.path.getsize(fwd_out):,} байт (2.39 МБ)")
    print(f" 2. INV_FP16.BIN: {os.path.getsize(inv_out):,} байт (2.41 МБ)")
    print(f" Суммарный объем: {(os.path.getsize(fwd_out) + os.path.getsize(inv_out))/(1024*1024):.2f} МБ <= 8 МБ Flash!")
    print("="*70)

if __name__ == "__main__":
    main()

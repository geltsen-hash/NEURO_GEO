#!/usr/bin/env python3
"""
Конвертер весов и каротажа в бинарный формат FP32 для STM32H750 и SD-карты.
"""

import os
import struct
import numpy as np
import pandas as pd

def export_fwd_bin(weights_dir, out_file):
    dims = [7, 512, 1024, 512, 256, 40]
    total_floats = 0
    layers = []
    
    for i in range(5):
        w_path = os.path.join(weights_dir, f"FWD_W{i+1}.txt")
        b_path = os.path.join(weights_dir, f"FWD_b{i+1}.txt")
        
        W = np.loadtxt(w_path, dtype=np.float32)
        b = np.loadtxt(b_path, dtype=np.float32)
        
        expected_w_shape = (dims[i], dims[i+1])
        if W.shape != expected_w_shape:
            raise ValueError(f"FWD_W{i+1} shape {W.shape} != expected {expected_w_shape}")
        if b.shape != (dims[i+1],):
            raise ValueError(f"FWD_b{i+1} shape {b.shape} != expected ({dims[i+1]},)")
            
        layers.append((W, b))
        total_floats += W.size + b.size
        
    print(f"FWD total floats: {total_floats} ({total_floats * 4 / (1024*1024):.2f} MB)")
    
    with open(out_file, "wb") as f:
        # Header: magic(4), num_layers(4), dims[6](24), total_floats(4), reserved[7](28) = 64 bytes
        magic = 0x46574431  # 'FWD1'
        header = struct.pack("<II6II7I", magic, 5, *dims, total_floats, *([0]*7))
        f.write(header)
        
        for W, b in layers:
            f.write(W.tobytes(order='C'))
            f.write(b.tobytes(order='C'))
            
    print(f"Сохранён: {out_file} ({os.path.getsize(out_file)} байт)")


def export_inv_bin(weights_dir, out_file):
    dims = [40, 512, 1024, 512, 256, 7]
    total_floats = 0
    layers = []
    
    for i in range(5):
        w_path = os.path.join(weights_dir, f"INV_W{i+1}.txt")
        b_path = os.path.join(weights_dir, f"INV_b{i+1}.txt")
        
        W = np.loadtxt(w_path, dtype=np.float32)
        b = np.loadtxt(b_path, dtype=np.float32)
        
        expected_w_shape = (dims[i], dims[i+1])
        if W.shape != expected_w_shape:
            raise ValueError(f"INV_W{i+1} shape {W.shape} != expected {expected_w_shape}")
        if b.shape != (dims[i+1],):
            raise ValueError(f"INV_b{i+1} shape {b.shape} != expected ({dims[i+1]},)")
            
        layers.append((W, b))
        total_floats += W.size + b.size
        
    print(f"INV total floats: {total_floats} ({total_floats * 4 / (1024*1024):.2f} MB)")
    
    with open(out_file, "wb") as f:
        # Header: magic(4), num_layers(4), dims[6](24), total_floats(4), reserved[7](28) = 64 bytes
        magic = 0x494E5631  # 'INV1'
        header = struct.pack("<II6II7I", magic, 5, *dims, total_floats, *([0]*7))
        f.write(header)
        
        for W, b in layers:
            f.write(W.tobytes(order='C'))
            f.write(b.tobytes(order='C'))
            
    print(f"Сохранён: {out_file} ({os.path.getsize(out_file)} байт)")


def export_scalers_bin(weights_dir, out_file):
    mean_X = np.loadtxt(os.path.join(weights_dir, "scaler_X_mean.txt"), dtype=np.float32)
    scale_X = np.loadtxt(os.path.join(weights_dir, "scaler_X_scale.txt"), dtype=np.float32)
    mean_Y = np.loadtxt(os.path.join(weights_dir, "scaler_Y_mean.txt"), dtype=np.float32)
    scale_Y = np.loadtxt(os.path.join(weights_dir, "scaler_Y_scale.txt"), dtype=np.float32)
    
    assert mean_X.shape == (7,)
    assert scale_X.shape == (7,)
    assert mean_Y.shape == (40,)
    assert scale_Y.shape == (40,)
    
    with open(out_file, "wb") as f:
        # Header: magic(4), x_dim(4), y_dim(4), reserved[13](52) = 64 bytes
        magic = 0x5343414C  # 'SCAL'
        header = struct.pack("<III13I", magic, 7, 40, *([0]*13))
        f.write(header)
        
        f.write(mean_X.tobytes(order='C'))
        f.write(scale_X.tobytes(order='C'))
        f.write(mean_Y.tobytes(order='C'))
        f.write(scale_Y.tobytes(order='C'))
        
    print(f"Сохранён: {out_file} ({os.path.getsize(out_file)} байт)")


def export_synthlog_bin(csv_file, out_file):
    df = pd.read_csv(csv_file)
    data = df.values.astype(np.float32)
    num_points, num_cols = data.shape
    print(f"Synthetic log: {num_points} points, {num_cols} columns")
    
    with open(out_file, "wb") as f:
        # Header: magic(4), num_points(4), num_cols(4), reserved[13](52) = 64 bytes
        magic = 0x53594E31  # 'SYN1'
        header = struct.pack("<III13I", magic, num_points, num_cols, *([0]*13))
        f.write(header)
        f.write(data.tobytes(order='C'))
        
    print(f"Сохранён: {out_file} ({os.path.getsize(out_file)} байт)")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "engine_weights")
    test_csv = os.path.join(base_dir, "test", "synthetic_well_log_40ch.csv")
    
    # 1. Локальный экспорт в engine_weights
    print("--- Экспорт в локальный каталог engine_weights ---")
    fwd_bin = os.path.join(weights_dir, "FWD_FP32.BIN")
    inv_bin = os.path.join(weights_dir, "INV_FP32.BIN")
    scalers_bin = os.path.join(weights_dir, "SCALERS.BIN")
    synth_bin = os.path.join(base_dir, "test", "SYNTHLOG.BIN")
    
    export_fwd_bin(weights_dir, fwd_bin)
    export_inv_bin(weights_dir, inv_bin)
    export_scalers_bin(weights_dir, scalers_bin)
    export_synthlog_bin(test_csv, synth_bin)
    
    # 2. Экспорт на SD-карту (диск E:), если доступен
    sd_dir = "E:\\"
    if os.path.exists(sd_dir):
        print("\n--- Запись на SD-карту (E:) ---")
        import shutil
        shutil.copy2(fwd_bin, os.path.join(sd_dir, "FWD_FP32.BIN"))
        shutil.copy2(inv_bin, os.path.join(sd_dir, "INV_FP32.BIN"))
        shutil.copy2(scalers_bin, os.path.join(sd_dir, "SCALERS.BIN"))
        shutil.copy2(synth_bin, os.path.join(sd_dir, "SYNTHLOG.BIN"))
        shutil.copy2(test_csv, os.path.join(sd_dir, "SYNTHLOG.CSV"))
        print(f"Все 5 файлов успешно записаны на SD-карту ({sd_dir})!")
    else:
        print(f"\nПредупреждение: диск {sd_dir} не найден для прямой записи.")

if __name__ == "__main__":
    main()

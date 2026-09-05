import argparse
import ctypes
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

NOISE_THRESHOLD = 0.045
TV_WEIGHT_GEO = 0.0
TV_WEIGHT_ANG = 0.0


def load_txt_linear(layer, w_file, b_file):
    W = np.loadtxt(w_file, ndmin=2)
    b = np.loadtxt(b_file, ndmin=1)
    if W.shape != (layer.in_features, layer.out_features):
        raise ValueError(f'{w_file}: expected {layer.in_features, layer.out_features}, got {W.shape}')
    if b.shape != (layer.out_features,):
        raise ValueError(f'{b_file}: expected {(layer.out_features,)}, got {b.shape}')
    layer.weight.data = torch.FloatTensor(W.T)
    layer.bias.data = torch.FloatTensor(b)


class ForwardSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 40)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.gelu(self.fc4(x))
        return self.fc5(x)


class InverseNet40CH(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 7)
        self.gelu = nn.GELU()

    def forward(self, y):
        y = self.gelu(self.fc1(y))
        y = self.gelu(self.fc2(y))
        y = self.gelu(self.fc3(y))
        y = self.gelu(self.fc4(y))
        return self.fc5(y)


def run_pgd_for_topology(y_true, forward_net, best_geo_init, min_scaled, max_scaled,
                         scale_X, mean_X, active_mask, topology_mode, m1_res=None,
                         steps=150, lr=0.015):
    """Батчовый PGD из test_PINN_40sig_7p.py / NEURO_40_7p_DLL_test.py."""
    device = y_true.device
    geo_opt = best_geo_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([geo_opt], lr=lr)

    if topology_mode == 2 and m1_res is not None:
        m1_scaled = torch.tensor((m1_res - mean_X) / scale_X, dtype=torch.float32, device=device)
        m1_real_t = torch.tensor(m1_res, dtype=torch.float32, device=device)
        mask_up_closer = m1_real_t[:, 4] <= m1_real_t[:, 5]
        mask_dn_closer = ~mask_up_closer
        with torch.no_grad():
            geo_opt[mask_up_closer, 4] = m1_scaled[mask_up_closer, 4]
            geo_opt[mask_up_closer, 0] = m1_scaled[mask_up_closer, 0]
            geo_opt[mask_dn_closer, 5] = m1_scaled[mask_dn_closer, 5]
            geo_opt[mask_dn_closer, 3] = m1_scaled[mask_dn_closer, 3]

    weights = torch.ones(40, device=device)
    for step in range(steps):
        optimizer.zero_grad()
        logs_recon = forward_net(geo_opt)
        loss_phys = torch.mean(weights * (logs_recon - y_true) ** 2)
        if TV_WEIGHT_GEO > 0.0 and geo_opt.shape[0] > 1:
            loss_tv_geo = torch.mean((active_mask[:-1, 0] *
                (geo_opt[1:, 4:6] - geo_opt[:-1, 4:6])) ** 2)
            loss_phys += TV_WEIGHT_GEO * loss_tv_geo
        if TV_WEIGHT_ANG > 0.0 and geo_opt.shape[0] > 1:
            loss_tv_ang = torch.mean((active_mask[:-1, 0] *
                (geo_opt[1:, 6] - geo_opt[:-1, 6])) ** 2)
            loss_phys += TV_WEIGHT_ANG * loss_tv_ang

        loss_phys.backward()
        geo_opt.grad.data[:, 4:6] *= active_mask

        if topology_mode == 2 and m1_res is not None:
            geo_opt.grad.data[mask_up_closer, 4] = 0.0
            geo_opt.grad.data[mask_up_closer, 0] = 0.0
            geo_opt.grad.data[mask_dn_closer, 5] = 0.0
            geo_opt.grad.data[mask_dn_closer, 3] = 0.0

        optimizer.step()

        with torch.no_grad():
            geo_opt.data = torch.max(min_scaled, torch.min(max_scaled, geo_opt.data))
            geo_real_fix = geo_opt.data * torch.tensor(scale_X, device=device) + torch.tensor(mean_X, device=device)

            if topology_mode == 0:
                geo_real_fix[:, 4] = 4.0
                geo_real_fix[:, 5] = 4.0
                geo_real_fix[:, 0] = geo_real_fix[:, 1]
                geo_real_fix[:, 3] = geo_real_fix[:, 1]
            elif topology_mode == 1:
                mask_up_c = geo_real_fix[:, 4] <= geo_real_fix[:, 5]
                geo_real_fix[mask_up_c, 5] = 4.0
                geo_real_fix[mask_up_c, 3] = geo_real_fix[mask_up_c, 1]
                geo_real_fix[~mask_up_c, 4] = 4.0
                geo_real_fix[~mask_up_c, 0] = geo_real_fix[~mask_up_c, 1]
            elif topology_mode == 2 and m1_res is not None:
                geo_real_fix[mask_up_closer, 4] = m1_real_t[mask_up_closer, 4]
                geo_real_fix[mask_up_closer, 0] = m1_real_t[mask_up_closer, 0]
                geo_real_fix[mask_dn_closer, 5] = m1_real_t[mask_dn_closer, 5]
                geo_real_fix[mask_dn_closer, 3] = m1_real_t[mask_dn_closer, 3]

            geo_opt.data = (geo_real_fix - torch.tensor(mean_X, device=device)) / torch.tensor(scale_X, device=device)

    final_geo_scaled = geo_opt.detach().cpu().numpy()
    return (final_geo_scaled * scale_X) + mean_X


def init_dll(dll_path, weights_dir, max_steps=150, lr=0.015):
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f'DLL not found: {dll_path}')
    dll_dir = os.path.dirname(dll_path)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_dir)
    lib = ctypes.CDLL(dll_path, winmode=0)
    fp = ctypes.POINTER(ctypes.c_float)
    lib.InitEngine.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float]
    lib.InitEngine.restype = ctypes.c_bool
    lib.RunPGDInversionEx.argtypes = [fp, fp, fp, ctypes.c_int, fp, fp]
    lib.RunPGDInversionEx.restype = None
    lib.RunTopologyInversion.argtypes = [fp, fp, fp, fp]
    lib.RunTopologyInversion.restype = None
    if not lib.InitEngine(weights_dir.encode('utf-8'), max_steps, ctypes.c_float(lr)):
        raise RuntimeError('InitEngine failed')
    return lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dll', required=True, help='path to NEURO_40_7P_DLL.dll')
    parser.add_argument('--csv', default='synthetic_well_log_40ch.csv')
    parser.add_argument('--engine-dir', default='engine_weights')
    parser.add_argument('--out-csv', default='dll_vs_pinn.csv')
    parser.add_argument('--out-png', default='dll_vs_pinn.png')
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.015)
    args = parser.parse_args()

    inv_file = os.path.join(args.engine_dir, 'PINN_40sig_7p.pt')
    if not os.path.exists(inv_file):
        raise FileNotFoundError(inv_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[OK] device:', device)

    print('[OK] Loading forward net from text weights...')
    forward_net = ForwardSurrogate()
    for i in range(1, 6):
        load_txt_linear(getattr(forward_net, f'fc{i}'),
                        os.path.join(args.engine_dir, f'FWD_W{i}.txt'),
                        os.path.join(args.engine_dir, f'FWD_b{i}.txt'))
    forward_net = forward_net.to(device).eval()
    for p in forward_net.parameters():
        p.requires_grad = False

    print('[OK] Loading inverse net from .pt...')
    inv_net = InverseNet40CH().to(device)
    inv_net.load_state_dict(torch.load(inv_file, map_location=device, weights_only=True))
    inv_net.eval()

    print('[OK] Reading CSV...')
    data = np.loadtxt(args.csv, delimiter=',', skiprows=1)
    md_array = data[:, 0]
    true = data[:, 1:8]
    logs_raw = data[:, 8:48].astype(np.float32)

    mean_X = np.loadtxt(os.path.join(args.engine_dir, 'scaler_X_mean.txt'), dtype=np.float32)
    scale_X = np.loadtxt(os.path.join(args.engine_dir, 'scaler_X_scale.txt'), dtype=np.float32)
    mean_Y = np.loadtxt(os.path.join(args.engine_dir, 'scaler_Y_mean.txt'), dtype=np.float32)
    scale_Y = np.loadtxt(os.path.join(args.engine_dir, 'scaler_Y_scale.txt'), dtype=np.float32)

    amp_indices = [4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38]
    logs_raw_log = logs_raw.copy()
    logs_raw_log[:, amp_indices] = np.log10(np.maximum(logs_raw_log[:, amp_indices], 1e-8))
    logs_scaled = (logs_raw_log - mean_Y) / scale_Y
    y_true = torch.tensor(logs_scaled, dtype=torch.float32, device=device)

    signal_amplitude = torch.std(y_true[:, amp_indices], dim=1)
    active_mask = (signal_amplitude > NOISE_THRESHOLD).float().unsqueeze(1)
    print(f'[OK] Active fraction: {active_mask.float().mean().item():.2%}')

    with torch.no_grad():
        best_geo_init = inv_net(y_true)

    num_points = y_true.shape[0]
    min_bounds_raw = np.zeros((num_points, 7), dtype=np.float32)
    max_bounds_raw = np.zeros((num_points, 7), dtype=np.float32)
    min_bounds_raw[:, :] = [1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 60.0]
    max_bounds_raw[:, :] = [1000.0, 1000.0, 1000.0, 1000.0, 4.0, 4.0, 120.0]
    min_bounds = min_bounds_raw.copy()
    max_bounds = max_bounds_raw.copy()
    min_bounds[:, 0:4] = np.log10(np.maximum(min_bounds[:, 0:4], 1e-5))
    max_bounds[:, 0:4] = np.log10(np.maximum(max_bounds[:, 0:4], 1e-5))
    min_scaled = torch.tensor((min_bounds - mean_X) / scale_X, dtype=torch.float32, device=device)
    max_scaled = torch.tensor((max_bounds - mean_X) / scale_X, dtype=torch.float32, device=device)
    best_geo_init = torch.max(min_scaled, torch.min(max_scaled, best_geo_init))

    print('\n[OK] Running PyTorch PGD for topology modes...')
    t0 = time.time()
    res_0 = run_pgd_for_topology(y_true, forward_net, best_geo_init, min_scaled, max_scaled,
                                 scale_X, mean_X, active_mask, 0, steps=args.steps, lr=args.lr)
    res_1 = run_pgd_for_topology(y_true, forward_net, best_geo_init, min_scaled, max_scaled,
                                 scale_X, mean_X, active_mask, 1, steps=args.steps, lr=args.lr)
    res_2 = run_pgd_for_topology(y_true, forward_net, best_geo_init, min_scaled, max_scaled,
                                 scale_X, mean_X, active_mask, 2, m1_res=res_1,
                                 steps=args.steps, lr=args.lr)
    print(f'[OK] PyTorch done in {time.time() - t0:.3f} s')

    print('\n[OK] Running DLL inversion...')
    lib = init_dll(args.dll, args.engine_dir, args.steps, args.lr)
    fp = ctypes.POINTER(ctypes.c_float)
    dll_min = (min_bounds_raw[0]).astype(np.float32)
    dll_max = (max_bounds_raw[0]).astype(np.float32)

    # Outputs:
    # - res_dll_top: 14 columns from RunTopologyInversion
    # - res_dll_m2: full 7 params from RunPGDInversionEx mode 2 (for alpha and sanity check)
    res_dll_top = np.zeros((num_points, 14), dtype=np.float32)
    res_dll_m2 = np.zeros((num_points, 7), dtype=np.float32)
    t0 = time.time()
    for i in range(num_points):
        sig_in = np.ascontiguousarray(logs_raw[i])

        out14 = np.zeros(14, dtype=np.float32)
        lib.RunTopologyInversion(sig_in.ctypes.data_as(fp), dll_min.ctypes.data_as(fp),
                                  dll_max.ctypes.data_as(fp), out14.ctypes.data_as(fp))
        res_dll_top[i] = out14

        # explicit mode 1 -> m1, then explicit mode 2 -> full M2 including alpha
        m1 = np.zeros(7, dtype=np.float32)
        lib.RunPGDInversionEx(sig_in.ctypes.data_as(fp), dll_min.ctypes.data_as(fp),
                              dll_max.ctypes.data_as(fp), 1, None, m1.ctypes.data_as(fp))
        m2 = np.zeros(7, dtype=np.float32)
        lib.RunPGDInversionEx(sig_in.ctypes.data_as(fp), dll_min.ctypes.data_as(fp),
                              dll_max.ctypes.data_as(fp), 2, m1.ctypes.data_as(fp),
                              m2.ctypes.data_as(fp))
        res_dll_m2[i] = m2
    print(f'[OK] DLL done in {time.time() - t0:.3f} s')

    # Sanity: RunTopologyInversion M2 should match explicit RunPGDInversionEx M2 resistivities/depths
    topology_versus_explicit = np.column_stack([
        res_dll_top[:, 8],   # Rh_2
        res_dll_m2[:, 1],
        res_dll_top[:, 9],   # Rv_2
        res_dll_m2[:, 2],
        res_dll_top[:, 10],  # Rh_up_2
        res_dll_m2[:, 0],
        res_dll_top[:, 11],  # Rh_dn_2
        res_dll_m2[:, 3],
        res_dll_top[:, 12],  # D_up_2
        res_dll_m2[:, 4],
        res_dll_top[:, 13],  # D_dn_2
        res_dll_m2[:, 5],
    ])
    diff_topology_explicit = np.abs(topology_versus_explicit[:, 0::2] - topology_versus_explicit[:, 1::2])
    print(f'[OK] RunTopologyInversion vs explicit M2 max abs diff: {diff_topology_explicit.max():.6f}')

    header = 'MD,'
    header += 'Rh0_py,Rh0_dll,Rv0_py,Rv0_dll,'
    header += 'Rh1_py,Rh1_dll,Rv1_py,Rv1_dll,Rh_up1_py,Rh_up1_dll,Rh_dn1_py,Rh_dn1_dll,D_up1_py,D_up1_dll,D_dn1_py,D_dn1_dll,'
    header += 'Rh2_py,Rh2_dll,Rv2_py,Rv2_dll,Rh_up2_py,Rh_up2_dll,Rh_dn2_py,Rh_dn2_dll,D_up2_py,D_up2_dll,D_dn2_py,D_dn2_dll,Alpha2_py,Alpha2_dll'

    out_matrix = np.column_stack([
        md_array,
        10**res_0[:, 1], res_dll_top[:, 0], 10**res_0[:, 2], res_dll_top[:, 1],
        10**res_1[:, 1], res_dll_top[:, 2], 10**res_1[:, 2], res_dll_top[:, 3],
        10**res_1[:, 0], res_dll_top[:, 4], 10**res_1[:, 3], res_dll_top[:, 5], res_1[:, 4], res_dll_top[:, 6], res_1[:, 5], res_dll_top[:, 7],
        10**res_2[:, 1], res_dll_top[:, 8], 10**res_2[:, 2], res_dll_top[:, 9],
        10**res_2[:, 0], res_dll_top[:, 10], 10**res_2[:, 3], res_dll_top[:, 11], res_2[:, 4], res_dll_top[:, 12], res_2[:, 5], res_dll_top[:, 13],
        res_2[:, 6], res_dll_m2[:, 6],
    ])
    np.savetxt(args.out_csv, out_matrix, delimiter=',', header=header, comments='', fmt='%.5f', encoding='utf-8')

    def report(name, py_col, dll_col):
        diff = np.abs(py_col - dll_col)
        print(f'{name}: max abs diff = {diff.max():.6f}, mean = {diff.mean():.6f}')

    print('\n=== Per-parameter mismatch (PyTorch vs DLL) ===')
    report('Rh_0', 10**res_0[:, 1], res_dll_top[:, 0])
    report('Rv_0', 10**res_0[:, 2], res_dll_top[:, 1])
    report('Rh_1', 10**res_1[:, 1], res_dll_top[:, 2])
    report('Rv_1', 10**res_1[:, 2], res_dll_top[:, 3])
    report('Rh_up_1', 10**res_1[:, 0], res_dll_top[:, 4])
    report('Rh_dn_1', 10**res_1[:, 3], res_dll_top[:, 5])
    report('D_up_1', res_1[:, 4], res_dll_top[:, 6])
    report('D_dn_1', res_1[:, 5], res_dll_top[:, 7])
    report('Rh_2', 10**res_2[:, 1], res_dll_top[:, 8])
    report('Rv_2', 10**res_2[:, 2], res_dll_top[:, 9])
    report('Rh_up_2', 10**res_2[:, 0], res_dll_top[:, 10])
    report('Rh_dn_2', 10**res_2[:, 3], res_dll_top[:, 11])
    report('D_up_2', res_2[:, 4], res_dll_top[:, 12])
    report('D_dn_2', res_2[:, 5], res_dll_top[:, 13])
    report('Alpha_2', res_2[:, 6], res_dll_m2[:, 6])

    fig, axs = plt.subplots(5, 1, figsize=(14, 20))
    axs[0].plot(md_array, true[:, 4], 'k-', alpha=0.2, linewidth=8, label='Ист. Кровля')
    axs[0].plot(md_array, true[:, 5], 'k-', alpha=0.2, linewidth=8, label='Ист. Подошва')
    axs[0].plot(md_array, res_2[:, 4], 'r--', linewidth=2, label='PyTorch M2 Кровля')
    axs[0].plot(md_array, res_2[:, 5], 'r:', linewidth=2, label='PyTorch M2 Подошва')
    axs[0].plot(md_array, res_dll_top[:, 12], 'gold', linewidth=3, alpha=0.7, label='DLL M2 Кровля')
    axs[0].plot(md_array, res_dll_top[:, 13], 'orange', linewidth=3, alpha=0.7, label='DLL M2 Подошва')
    axs[0].invert_yaxis(); axs[0].grid(True); axs[0].legend(loc='upper right')
    axs[0].set_title('Геометрия границ (PyTorch M2 vs DLL RunTopologyInversion)')

    axs[1].plot(md_array, true[:, 1], 'k-', alpha=0.2, linewidth=8, label='Ист. Rh_pl')
    axs[1].plot(md_array, 10**res_2[:, 1], 'r-', linewidth=2, label='PyTorch M2')
    axs[1].plot(md_array, res_dll_top[:, 8], 'orange', linewidth=3, alpha=0.7, label='DLL')
    axs[1].set_yscale('log'); axs[1].grid(True); axs[1].legend(loc='upper right')
    axs[1].set_title('Горизонтальное сопротивление пласта (Rh_2)')

    axs[2].plot(md_array, true[:, 0], 'k-', alpha=0.2, linewidth=8, label='Ист. Rh_up')
    axs[2].plot(md_array, true[:, 3], 'k-', alpha=0.2, linewidth=8, label='Ист. Rh_dn')
    axs[2].plot(md_array, 10**res_2[:, 0], 'r--', linewidth=2, label='PyTorch M2 Кровля')
    axs[2].plot(md_array, 10**res_2[:, 3], 'r:', linewidth=2, label='PyTorch M2 Подошва')
    axs[2].plot(md_array, res_dll_top[:, 10], 'gold', linewidth=3, alpha=0.7, label='DLL Кровля')
    axs[2].plot(md_array, res_dll_top[:, 11], 'orange', linewidth=3, alpha=0.7, label='DLL Подошва')
    axs[2].set_yscale('log'); axs[2].grid(True); axs[2].legend(loc='upper right')
    axs[2].set_title('Сопротивление смежных пластов (Rh_up_2 / Rh_dn_2)')

    axs[3].plot(md_array, true[:, 2] / true[:, 1], 'k-', alpha=0.2, linewidth=8, label='Ист.')
    axs[3].plot(md_array, 10**res_2[:, 2] / 10**res_2[:, 1], 'r-', linewidth=2, label='PyTorch M2')
    axs[3].plot(md_array, res_dll_top[:, 9] / res_dll_top[:, 8], 'orange', linewidth=3, alpha=0.7, label='DLL')
    axs[3].grid(True); axs[3].legend(loc='upper right')
    axs[3].set_title('Анизотропия (Rv_2 / Rh_2)')

    axs[4].plot(md_array, true[:, 6], 'k-', alpha=0.2, linewidth=8, label='Ист.')
    axs[4].plot(md_array, res_2[:, 6], 'r-', linewidth=2, label='PyTorch M2')
    axs[4].plot(md_array, res_dll_m2[:, 6], 'orange', linewidth=3, alpha=0.7, label='DLL M2 (RunPGDInversionEx)')
    axs[4].grid(True); axs[4].legend(loc='upper right')
    axs[4].set_title('Зенитный угол')

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f'\n[OK] Saved {args.out_csv} and {args.out_png}')


if __name__ == '__main__':
    main()

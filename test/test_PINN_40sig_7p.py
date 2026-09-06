import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import json
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# --- ФИЗИЧЕСКИЕ ГРАНИЦЫ, СОГЛАСОВАННЫЕ С ГЕНЕРАТОРОМ ДАТАСЕТА ---
RO_MIN, RO_MAX = 1.0, 1000.0    # УЭС, Ом·м
D_MIN, D_MAX = 0.05, 4.0        # глубины границ, м
ANGLE_MIN, ANGLE_MAX = 60.0, 120.0  # угол, градусы

TV_WEIGHT_GEO = 0.0
TV_WEIGHT_ANG = 0.0
NOISE_THRESHOLD = 0.045  # legacy-порог в масштабированном пространстве


def main():
    parser = argparse.ArgumentParser(description='End-to-end инверсия по синтетическому каротажу.')
    parser.add_argument('--csv', default='synthetic_well_log_40ch.csv',
                        help='Входной CSV (MD + 7 true params + 40 сигналов)')
    parser.add_argument('--engine-dir', default='engine_weights',
                        help='Каталог с FWD_*, INV_*, скейлерами и preprocess_meta.json')
    parser.add_argument('--out-csv', default='topology_parallel_complete.csv')
    parser.add_argument('--out-png', default='topology_parallel_complete.png')
    args = parser.parse_args()

    engine_dir = os.path.abspath(args.engine_dir)
    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(f'Каталог engine не найден: {engine_dir}')

    meta_path = os.path.join(engine_dir, 'preprocess_meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'Файл {meta_path} не найден!')
    with open(meta_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)

    amp_indices = meta['y_log10_indices']
    x_log10_indices = meta['x_log10_indices']
    inv_file = os.path.join(engine_dir, 'PINN_40sig_7p.pt')
    if not os.path.exists(inv_file):
        raise FileNotFoundError(f'Файл {inv_file} не найден!')

    print('[OK] Старт пайплайна инверсии')

    # --- АРХИТЕКТУРЫ СЕТЕЙ ---
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

        def load_txt_weights(self, weights_dir):
            def assign_layer(layer, w_file, b_file):
                if not os.path.exists(w_file) or not os.path.exists(b_file):
                    raise FileNotFoundError(f'Не найдены {w_file} или {b_file}')
                W = np.loadtxt(w_file, ndmin=2)
                b = np.loadtxt(b_file)
                layer.weight.data = torch.FloatTensor(W.T)
                layer.bias.data = torch.FloatTensor(b)

            assign_layer(self.fc1, os.path.join(weights_dir, 'FWD_W1.txt'),
                         os.path.join(weights_dir, 'FWD_b1.txt'))
            assign_layer(self.fc2, os.path.join(weights_dir, 'FWD_W2.txt'),
                         os.path.join(weights_dir, 'FWD_b2.txt'))
            assign_layer(self.fc3, os.path.join(weights_dir, 'FWD_W3.txt'),
                         os.path.join(weights_dir, 'FWD_b3.txt'))
            assign_layer(self.fc4, os.path.join(weights_dir, 'FWD_W4.txt'),
                         os.path.join(weights_dir, 'FWD_b4.txt'))
            assign_layer(self.fc5, os.path.join(weights_dir, 'FWD_W5.txt'),
                         os.path.join(weights_dir, 'FWD_b5.txt'))

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

    mean_X = np.loadtxt(os.path.join(engine_dir, 'scaler_X_mean.txt'))
    scale_X = np.loadtxt(os.path.join(engine_dir, 'scaler_X_scale.txt'))
    mean_Y = np.loadtxt(os.path.join(engine_dir, 'scaler_Y_mean.txt'))
    scale_Y = np.loadtxt(os.path.join(engine_dir, 'scaler_Y_scale.txt'))

    if np.any(scale_X == 0) or np.any(scale_Y == 0):
        raise ValueError('В скейлерах есть нулевой масштаб')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[OK] device: {device}')

    print('[OK] Загрузка FWD сети...')
    forward_net = ForwardSurrogate()
    forward_net.load_txt_weights(engine_dir)
    forward_net = forward_net.to(device)
    forward_net.eval()
    for param in forward_net.parameters():
        param.requires_grad = False

    print('[OK] Чтение синтетического каротажа...')
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f'CSV {args.csv} не найден')
    data_matrix = np.loadtxt(args.csv, delimiter=',', skiprows=1)

    md_array = data_matrix[:, 0]
    true_rh_up = data_matrix[:, 1]
    true_rh_pl = data_matrix[:, 2]
    true_rv_pl = data_matrix[:, 3]
    true_rh_dn = data_matrix[:, 4]
    true_d_up = data_matrix[:, 5]
    true_d_down = data_matrix[:, 6]
    true_alpha = data_matrix[:, 7]

    logs_raw = data_matrix[:, 8:48].astype(np.float32)
    print('[OK] Масштабирование сигналов...')
    for i in amp_indices:
        logs_raw[:, i] = np.log10(np.maximum(logs_raw[:, i], 1e-8))
    logs_scaled = (logs_raw - mean_Y) / scale_Y
    y_true = torch.tensor(logs_scaled, dtype=torch.float32, device=device)
    num_points = y_true.shape[0]

    print('[OK] Расчет маски шума...')
    signal_amplitude = torch.std(y_true[:, amp_indices], dim=1)
    active_mask = (signal_amplitude > NOISE_THRESHOLD).float().unsqueeze(1)
    active_fraction = active_mask.float().mean().item()
    print(f'   доля активных точек по маске: {active_fraction:.2%}')

    print(f'[OK] Подгрузка обратной нейросети из {inv_file}...')
    inv_net = InverseNet40CH().to(device)
    inv_net.load_state_dict(torch.load(inv_file, map_location=device, weights_only=True))
    inv_net.eval()

    print('[OK] Выполняем первичную оценку среды...')
    with torch.no_grad():
        best_geo_init = inv_net(y_true)

    print('[OK] Применение ограничений (Bounds)...')
    min_bounds = np.zeros((num_points, 7), dtype=np.float32)
    max_bounds = np.zeros((num_points, 7), dtype=np.float32)
    base_min = [RO_MIN, RO_MIN, RO_MIN, RO_MIN, D_MIN, D_MIN, ANGLE_MIN]
    base_max = [RO_MAX, RO_MAX, RO_MAX, RO_MAX, D_MAX, D_MAX, ANGLE_MAX]
    for i in range(num_points):
        min_bounds[i] = base_min
        max_bounds[i] = base_max

    eps = 1e-5
    min_bounds[:, x_log10_indices] = np.log10(np.maximum(min_bounds[:, x_log10_indices], eps))
    max_bounds[:, x_log10_indices] = np.log10(np.maximum(max_bounds[:, x_log10_indices], eps))

    min_scaled = torch.tensor((min_bounds - mean_X) / scale_X, dtype=torch.float32, device=device)
    max_scaled = torch.tensor((max_bounds - mean_X) / scale_X, dtype=torch.float32, device=device)
    best_geo_init = torch.max(min_scaled, torch.min(max_scaled, best_geo_init))

    scale_gpu = torch.tensor(scale_X, dtype=torch.float32, device=device)
    mean_gpu = torch.tensor(mean_X, dtype=torch.float32, device=device)
    weights = torch.ones(40, device=device)

    # --- ПАКЕТНАЯ КАСКАДНАЯ ИНВЕРСИЯ ---
    def run_pgd_for_topology(topology_mode, steps=150, m1_res=None):
        print(f'\n>>> Запуск PGD: Модель {topology_mode} ({topology_mode} границ)...')
        geo_opt = best_geo_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([geo_opt], lr=0.015)

        if topology_mode == 2 and m1_res is not None:
            m1_scaled_np = (m1_res - mean_X) / scale_X
            m1_scaled = torch.tensor(m1_scaled_np, dtype=torch.float32, device=device)
            m1_real = torch.tensor(m1_res, dtype=torch.float32, device=device)
            mask_up_closer = m1_real[:, 4] <= m1_real[:, 5]
            mask_dn_closer = ~mask_up_closer

            with torch.no_grad():
                geo_opt[mask_up_closer, 4] = m1_scaled[mask_up_closer, 4]
                geo_opt[mask_up_closer, 0] = m1_scaled[mask_up_closer, 0]
                geo_opt[mask_dn_closer, 5] = m1_scaled[mask_dn_closer, 5]
                geo_opt[mask_dn_closer, 3] = m1_scaled[mask_dn_closer, 3]

        for step in range(steps):
            optimizer.zero_grad()
            logs_recon = forward_net(geo_opt)
            loss_phys = torch.mean(weights * (logs_recon - y_true)**2)

            geo_real = geo_opt * scale_gpu + mean_gpu
            diff = geo_real[1:, :] - geo_real[:-1, :]

            loss_tv_geo = torch.mean((diff[:, :6] * active_mask[:-1])**2)
            loss_tv_ang = torch.mean(diff[:, 6]**2)
            loss = loss_phys + TV_WEIGHT_GEO * loss_tv_geo + TV_WEIGHT_ANG * loss_tv_ang
            loss.backward()

            geo_opt.grad.data[:, 4:6] *= active_mask

            if topology_mode == 2 and m1_res is not None:
                geo_opt.grad.data[mask_up_closer, 4] = 0.0
                geo_opt.grad.data[mask_up_closer, 0] = 0.0
                geo_opt.grad.data[mask_dn_closer, 5] = 0.0
                geo_opt.grad.data[mask_dn_closer, 3] = 0.0

            optimizer.step()

            with torch.no_grad():
                geo_opt.data = torch.max(min_scaled, torch.min(max_scaled, geo_opt.data))
                geo_real_fix = geo_opt.data * scale_gpu + mean_gpu

                if topology_mode == 0:
                    geo_real_fix[:, 4] = D_MAX
                    geo_real_fix[:, 5] = D_MAX
                    geo_real_fix[:, 0] = geo_real_fix[:, 1]
                    geo_real_fix[:, 3] = geo_real_fix[:, 1]

                elif topology_mode == 1:
                    mask_up_c = geo_real_fix[:, 4] <= geo_real_fix[:, 5]
                    geo_real_fix[mask_up_c, 5] = D_MAX
                    geo_real_fix[mask_up_c, 3] = geo_real_fix[mask_up_c, 1]
                    geo_real_fix[~mask_up_c, 4] = D_MAX
                    geo_real_fix[~mask_up_c, 0] = geo_real_fix[~mask_up_c, 1]

                elif topology_mode == 2 and m1_res is not None:
                    geo_real_fix[mask_up_closer, 4] = m1_real[mask_up_closer, 4]
                    geo_real_fix[mask_up_closer, 0] = m1_real[mask_up_closer, 0]
                    geo_real_fix[mask_dn_closer, 5] = m1_real[mask_dn_closer, 5]
                    geo_real_fix[mask_dn_closer, 3] = m1_real[mask_dn_closer, 3]

                geo_opt.data = (geo_real_fix - mean_gpu) / scale_gpu

        final_geo_scaled = geo_opt.detach().cpu().numpy()
        return (final_geo_scaled * scale_X) + mean_X

    res_0 = run_pgd_for_topology(0)
    res_1 = run_pgd_for_topology(1)
    res_2 = run_pgd_for_topology(2, m1_res=res_1)

    print('\n[OK] Сборка CSV-матрицы...')
    out_matrix = np.column_stack((
        md_array,
        10**res_0[:, 1], 10**res_0[:, 2],
        10**res_1[:, 1], 10**res_1[:, 2], 10**res_1[:, 0], 10**res_1[:, 3],
        res_1[:, 4], res_1[:, 5],
        10**res_2[:, 1], 10**res_2[:, 2], 10**res_2[:, 0], 10**res_2[:, 3],
        res_2[:, 4], res_2[:, 5]
    ))
    header_str = (
        'MD,Rh_0,Rv_0,Rh_1,Rv_1,Rh_up_1,Rh_dn_1,Кровля_1,Подошва_1,'
        'Rh_2,Rv_2,Rh_up_2,Rh_dn_2,Кровля_2,Подошва_2'
    )
    np.savetxt(args.out_csv, out_matrix, delimiter=',', header=header_str,
               comments='', fmt='%.5f', encoding='utf-8')

    print('\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (каждая 5-я точка) ===')
    print(f'{"MD":>4} | {"Ист.D_up":>8} | {"Ист.D_dn":>8} | '
          f'{"М1_D_up":>7} | {"М1_D_dn":>7} | {"М2_D_up":>7} | {"М2_D_dn":>7} | '
          f'{"M1: Rpl":>7} | {"Rup":>6} | {"Rdn":>6} | {"An":>4} | {"Ang_1":>5} | '
          f'{"M2: Rpl":>7} | {"Rup":>6} | {"Rdn":>6} | {"An":>4} | {"Ang_2":>5} | {"Ист.Ang":>7}')

    for i in range(0, num_points, 5):
        md = md_array[i]
        m1_dup, m1_ddn = res_1[i, 4], res_1[i, 5]
        rh1, rup1, rdn1 = 10**res_1[i, 1], 10**res_1[i, 0], 10**res_1[i, 3]
        an1, ang1 = 10**res_1[i, 2] / 10**res_1[i, 1], res_1[i, 6]
        m2_dup, m2_ddn = res_2[i, 4], res_2[i, 5]
        rh2, rup2, rdn2 = 10**res_2[i, 1], 10**res_2[i, 0], 10**res_2[i, 3]
        an2, ang2 = 10**res_2[i, 2] / 10**res_2[i, 1], res_2[i, 6]
        print(f'{md:4.1f} | {true_d_up[i]:8.2f} | {true_d_down[i]:8.2f} | '
              f'{m1_dup:7.2f} | {m1_ddn:7.2f} | {m2_dup:7.2f} | {m2_ddn:7.2f} | '
              f'{rh1:7.1f} | {rup1:6.1f} | {rdn1:6.1f} | {an1:4.2f} | {ang1:5.1f} | '
              f'{rh2:7.1f} | {rup2:6.1f} | {rdn2:6.1f} | {an2:4.2f} | {ang2:5.1f} | '
              f'{true_alpha[i]:7.2f}')

    print('\n[OK] Формирую графики...')
    fig, axs = plt.subplots(5, 1, figsize=(14, 20))
    axs[0].plot(md_array, true_d_up, 'k-', alpha=0.2, linewidth=8, label='Ист. Кровля')
    axs[0].plot(md_array, true_d_down, 'k-', alpha=0.2, linewidth=8, label='Ист. Подошва')
    axs[0].plot(md_array, res_1[:, 4], 'g--', linewidth=2, label='Модель 1 (Кровля)')
    axs[0].plot(md_array, res_1[:, 5], 'g:', linewidth=2, label='Модель 1 (Подошва)')
    axs[0].plot(md_array, res_2[:, 4], 'r--', linewidth=2, label='Модель 2 (Кровля)')
    axs[0].plot(md_array, res_2[:, 5], 'r:', linewidth=2, label='Модель 2 (Подошва)')
    axs[0].invert_yaxis(); axs[0].grid(True); axs[0].legend(loc='upper right')
    axs[0].set_title('Геометрия границ')

    axs[1].plot(md_array, true_rh_pl, 'k-', alpha=0.2, linewidth=8, label='Истинное Rh_pl')
    axs[1].plot(md_array, 10**res_0[:, 1], 'b-', linewidth=2, label='Модель 0')
    axs[1].plot(md_array, 10**res_1[:, 1], 'g-', linewidth=2, label='Модель 1')
    axs[1].plot(md_array, 10**res_2[:, 1], 'r-', linewidth=2, label='Модель 2')
    axs[1].set_yscale('log'); axs[1].grid(True); axs[1].legend(loc='upper right')
    axs[1].set_title('Горизонтальное сопротивление пласта (Rh)')

    axs[2].plot(md_array, true_rh_up, 'k-', alpha=0.2, linewidth=8, label='Ист. Rh_up')
    axs[2].plot(md_array, true_rh_dn, 'k:', alpha=0.2, linewidth=8, label='Ист. Rh_dn')
    axs[2].plot(md_array, 10**res_1[:, 0], 'g--', linewidth=2, label='М1 Кровля (Rh_up)')
    axs[2].plot(md_array, 10**res_1[:, 3], 'g:', linewidth=2, label='М1 Подошва (Rh_dn)')
    axs[2].plot(md_array, 10**res_2[:, 0], 'r--', linewidth=2, label='М2 Кровля (Rh_up)')
    axs[2].plot(md_array, 10**res_2[:, 3], 'r:', linewidth=2, label='М2 Подошва (Rh_dn)')
    axs[2].set_yscale('log'); axs[2].grid(True); axs[2].legend(loc='upper right')
    axs[2].set_title('Сопротивление смежных пластов (Rh_up, Rh_dn)')

    true_anis = true_rv_pl / true_rh_pl
    axs[3].plot(md_array, true_anis, 'k-', alpha=0.2, linewidth=8, label='Ист. Анизотропия')
    axs[3].plot(md_array, 10**res_0[:, 2]/10**res_0[:, 1], 'b-', linewidth=2, label='Модель 0')
    axs[3].plot(md_array, 10**res_1[:, 2]/10**res_1[:, 1], 'g-', linewidth=2, label='Модель 1')
    axs[3].plot(md_array, 10**res_2[:, 2]/10**res_2[:, 1], 'r-', linewidth=2, label='Модель 2')
    axs[3].grid(True); axs[3].legend(loc='upper right')
    axs[3].set_title('Коэффициент анизотропии (Rv / Rh)')

    axs[4].plot(md_array, true_alpha, 'k-', alpha=0.2, linewidth=8, label='Ист. Угол')
    axs[4].plot(md_array, res_0[:, 6], 'b-', linewidth=2, label='Модель 0')
    axs[4].plot(md_array, res_1[:, 6], 'g-', linewidth=2, label='Модель 1')
    axs[4].plot(md_array, res_2[:, 6], 'r-', linewidth=2, label='Модель 2')
    axs[4].set_ylabel('Угол (град)'); axs[4].grid(True); axs[4].legend(loc='upper right')
    axs[4].set_title('Зенитный угол пересечения пласта')

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f'\n[OK] Расчет завершен. CSV: {args.out_csv}, PNG: {args.out_png}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('\n!!! ПРОИЗОШЕЛ СБОЙ !!!')
        traceback.print_exc()

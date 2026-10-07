#!/usr/bin/env python3
"""
Анализ сходимости алгоритма Adam PGD для топологической инверсии (M0, M1, M2).
Исследование зависимости точности и функции потерь от количества шагов (15..150)
для подбора оптимального числа шагов под темп ~5-10 секунд на точку на MCU STM32H750.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Константы задачи ---
RO_MIN, RO_MAX = 1.0, 1000.0
D_MIN, D_MAX = 0.05, 4.0
ANGLE_MIN, ANGLE_MAX = 60.0, 120.0
NOISE_THRESHOLD = 0.045
AMP_INDICES = [4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38]
X_LOG_INDICES = [0, 1, 2, 3]

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
        def assign(layer, w_file, b_file):
            W = np.loadtxt(w_file, dtype=np.float32, ndmin=2)
            b = np.loadtxt(b_file, dtype=np.float32)
            layer.weight.data = torch.from_numpy(W.T)
            layer.bias.data = torch.from_numpy(b)

        assign(self.fc1, os.path.join(weights_dir, 'FWD_W1.txt'), os.path.join(weights_dir, 'FWD_b1.txt'))
        assign(self.fc2, os.path.join(weights_dir, 'FWD_W2.txt'), os.path.join(weights_dir, 'FWD_b2.txt'))
        assign(self.fc3, os.path.join(weights_dir, 'FWD_W3.txt'), os.path.join(weights_dir, 'FWD_b3.txt'))
        assign(self.fc4, os.path.join(weights_dir, 'FWD_W4.txt'), os.path.join(weights_dir, 'FWD_b4.txt'))
        assign(self.fc5, os.path.join(weights_dir, 'FWD_W5.txt'), os.path.join(weights_dir, 'FWD_b5.txt'))

class InverseNet(nn.Module):
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

    def load_txt_weights(self, weights_dir):
        def assign(layer, w_file, b_file):
            W = np.loadtxt(w_file, dtype=np.float32, ndmin=2)
            b = np.loadtxt(b_file, dtype=np.float32)
            layer.weight.data = torch.from_numpy(W.T)
            layer.bias.data = torch.from_numpy(b)

        assign(self.fc1, os.path.join(weights_dir, 'INV_W1.txt'), os.path.join(weights_dir, 'INV_b1.txt'))
        assign(self.fc2, os.path.join(weights_dir, 'INV_W2.txt'), os.path.join(weights_dir, 'INV_b2.txt'))
        assign(self.fc3, os.path.join(weights_dir, 'INV_W3.txt'), os.path.join(weights_dir, 'INV_b3.txt'))
        assign(self.fc4, os.path.join(weights_dir, 'INV_W4.txt'), os.path.join(weights_dir, 'INV_b4.txt'))
        assign(self.fc5, os.path.join(weights_dir, 'INV_W5.txt'), os.path.join(weights_dir, 'INV_b5.txt'))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, 'engine_weights')
    csv_file = os.path.join(base_dir, 'test', 'synthetic_well_log_40ch.csv')
    out_png = os.path.join(base_dir, 'test', 'adam_step_convergence.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[1/5] Инициализация моделей (Device: {device})...')

    fwd_net = ForwardSurrogate()
    fwd_net.load_txt_weights(weights_dir)
    fwd_net = fwd_net.to(device)
    fwd_net.eval()
    for p in fwd_net.parameters():
        p.requires_grad = False

    inv_net = InverseNet()
    inv_net.load_txt_weights(weights_dir)
    inv_net = inv_net.to(device)
    inv_net.eval()
    for p in inv_net.parameters():
        p.requires_grad = False

    # Скейлеры
    mean_X_np = np.loadtxt(os.path.join(weights_dir, 'scaler_X_mean.txt'), dtype=np.float32)
    scale_X_np = np.loadtxt(os.path.join(weights_dir, 'scaler_X_scale.txt'), dtype=np.float32)
    mean_Y_np = np.loadtxt(os.path.join(weights_dir, 'scaler_Y_mean.txt'), dtype=np.float32)
    scale_Y_np = np.loadtxt(os.path.join(weights_dir, 'scaler_Y_scale.txt'), dtype=np.float32)

    mean_X = torch.from_numpy(mean_X_np).to(device)
    scale_X = torch.from_numpy(scale_X_np).to(device)
    mean_Y = torch.from_numpy(mean_Y_np).to(device)
    scale_Y = torch.from_numpy(scale_Y_np).to(device)

    print(f'[2/5] Загрузка каротажа {csv_file}...')
    raw_csv = np.loadtxt(csv_file, delimiter=',', skiprows=1, dtype=np.float32)
    md_arr = raw_csv[:, 0]
    true_params = raw_csv[:, 1:8] # [Rh_up, Rh_pl, Rv_pl, Rh_dn, Dup, Ddn, Alpha]
    signals_raw = raw_csv[:, 8:48].copy()

    # Логарифмирование и нормализация сигналов
    for idx in AMP_INDICES:
        signals_raw[:, idx] = np.log10(np.maximum(signals_raw[:, idx], 1e-8))
    y_scaled_np = (signals_raw - mean_Y_np) / scale_Y_np
    y_true = torch.from_numpy(y_scaled_np).to(device)
    num_pts = y_true.shape[0]

    # Границы параметров
    min_b_np = np.array([RO_MIN, RO_MIN, RO_MIN, RO_MIN, D_MIN, D_MIN, ANGLE_MIN], dtype=np.float32)
    max_b_np = np.array([RO_MAX, RO_MAX, RO_MAX, RO_MAX, D_MAX, D_MAX, ANGLE_MAX], dtype=np.float32)
    for idx in X_LOG_INDICES:
        min_b_np[idx] = np.log10(np.maximum(min_b_np[idx], 1e-5))
        max_b_np[idx] = np.log10(np.maximum(max_b_np[idx], 1e-5))
    min_scaled = torch.from_numpy((min_b_np - mean_X_np) / scale_X_np).to(device)
    max_scaled = torch.from_numpy((max_b_np - mean_X_np) / scale_X_np).to(device)

    # Первичное приближение от обратной сети
    with torch.no_grad():
        opt_geo_init = inv_net(y_true)
        opt_geo_init = torch.clamp(opt_geo_init, min_scaled, max_scaled)

    # Маска активности сигнала
    sig_amp = torch.std(y_true[:, AMP_INDICES], dim=1)
    active_mask = (sig_amp > NOISE_THRESHOLD).float().unsqueeze(1)

    print(f'[3/5] Пошаговое отслеживание сходимости Adam (шаги 1..150)...')

    def run_pgd_steptrace(topo_mode, max_steps=150, m1_res=None):
        geo_opt = opt_geo_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([geo_opt], lr=0.015, betas=(0.9, 0.999), eps=1e-8)

        if topo_mode == 2 and m1_res is not None:
            m1_scaled = (m1_res - mean_X) / scale_X
            up_closer = m1_res[:, 4] <= m1_res[:, 5]
            with torch.no_grad():
                geo_opt[up_closer, 4] = m1_scaled[up_closer, 4]
                geo_opt[up_closer, 0] = m1_scaled[up_closer, 0]
                geo_opt[~up_closer, 5] = m1_scaled[~up_closer, 5]
                geo_opt[~up_closer, 3] = m1_scaled[~up_closer, 3]

        losses_trace = []
        snapshots = {}

        for step in range(1, max_steps + 1):
            opt.zero_grad()
            y_pred = fwd_net(geo_opt)
            loss = torch.mean((y_pred - y_true)**2)
            loss.backward()

            if topo_mode >= 0:
                geo_opt.grad.data[:, 4:6] *= active_mask
            if topo_mode == 2 and m1_res is not None:
                geo_opt.grad.data[up_closer, 4] = 0.0
                geo_opt.grad.data[up_closer, 0] = 0.0
                geo_opt.grad.data[~up_closer, 5] = 0.0
                geo_opt.grad.data[~up_closer, 3] = 0.0

            opt.step()

            with torch.no_grad():
                geo_opt.data = torch.clamp(geo_opt.data, min_scaled, max_scaled)
                real_fix = geo_opt.data * scale_X + mean_X

                if topo_mode == 0:
                    real_fix[:, 4] = D_MAX
                    real_fix[:, 5] = D_MAX
                    real_fix[:, 0] = real_fix[:, 1]
                    real_fix[:, 3] = real_fix[:, 1]
                elif topo_mode == 1:
                    m1_up = real_fix[:, 4] <= real_fix[:, 5]
                    real_fix[m1_up, 5] = D_MAX
                    real_fix[m1_up, 3] = real_fix[m1_up, 1]
                    real_fix[~m1_up, 4] = D_MAX
                    real_fix[~m1_up, 0] = real_fix[~m1_up, 1]
                elif topo_mode == 2 and m1_res is not None:
                    real_fix[up_closer, 4] = m1_res[up_closer, 4]
                    real_fix[up_closer, 0] = m1_res[up_closer, 0]
                    real_fix[~up_closer, 5] = m1_res[~up_closer, 5]
                    real_fix[~up_closer, 3] = m1_res[~up_closer, 3]

                geo_opt.data = (real_fix - mean_X) / scale_X

            losses_trace.append(loss.item())
            if step in [15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150]:
                snapshots[step] = (geo_opt.data * scale_X + mean_X).detach().clone()

        final_real = (geo_opt.data * scale_X + mean_X).detach().clone()
        return losses_trace, snapshots, final_real

    # Трассировка сходимости
    print("   Запуск M0 (0 границ)...")
    loss_m0, snap_m0, res_m0_150 = run_pgd_steptrace(0, max_steps=150)
    print("   Запуск M1 (1 граница)...")
    loss_m1, snap_m1, res_m1_150 = run_pgd_steptrace(1, max_steps=150)
    print("   Запуск M2 (2 границы)...")
    loss_m2, snap_m2, res_m2_150 = run_pgd_steptrace(2, max_steps=150, m1_res=res_m1_150)

    print(f'[4/5] Расчет метрик и сравнение конфигураций шагов...')

    configs = [
        ("Базовый 150/150/150", 150, 150, 150),
        ("Сбалансированный 50/50/50", 50, 50, 50),
        ("Оптимальный M0-30 / M1,2-50", 30, 50, 50),
        ("Ускоренный 25/40/40", 25, 40, 40),
        ("Турбо 20/30/30", 20, 30, 30),
        ("Экстремальный 15/25/25", 15, 25, 25),
    ]

    # Сравнение параметров с истинными значениями (True Rh_pl, True Rv_pl)
    true_rh_pl = torch.from_numpy(true_params[:, 1]).to(device)
    true_rv_pl = torch.from_numpy(true_params[:, 2]).to(device)

    print("\n" + "="*105)
    print(f"{'Конфигурация':<28} | {'Шагов':<6} | {'MCU t (c)':<9} | {'Loss M2':<9} | {'Отклон. vs 150':<15} | {'MAPE к True (Rh / Rv)'}")
    print("="*105)

    # 1 шаг FWD+BWD в FP16 на STM32H750 занимает ~82.7 мс (37.236 с / 450 шагов)
    time_per_step_mcu = 37.236 / 450.0

    eval_results = []
    for name, s0, s1, s2 in configs:
        total_steps = s0 + s1 + s2
        est_mcu_time = total_steps * time_per_step_mcu

        l0 = loss_m0[s0 - 1]
        l1 = loss_m1[s1 - 1]
        l2 = loss_m2[s2 - 1]

        # Расчет параметров
        r0 = snap_m0[s0]
        r1 = snap_m1[s1]
        # Для M2 каскад от m1 с шагом s1
        _, _, r2 = run_pgd_steptrace(2, max_steps=s2, m1_res=r1)

        # Отклонение по параметрам M2 относительно эталона 150 шагов
        dev_rh = torch.mean(torch.abs(10**r2[:, 1] - 10**res_m2_150[:, 1]) / 10**res_m2_150[:, 1]).item() * 100.0
        dev_rv = torch.mean(torch.abs(10**r2[:, 2] - 10**res_m2_150[:, 2]) / 10**res_m2_150[:, 2]).item() * 100.0
        dev_mean = 0.5 * (dev_rh + dev_rv)

        # Абсолютная ошибка MAPE относительно истинных параметров True
        mape_rh = torch.mean(torch.abs(10**r2[:, 1] - true_rh_pl) / true_rh_pl).item() * 100.0
        mape_rv = torch.mean(torch.abs(10**r2[:, 2] - true_rv_pl) / true_rv_pl).item() * 100.0

        eval_results.append({
            'name': name,
            's0': s0, 's1': s1, 's2': s2,
            'total_steps': total_steps,
            'time_mcu': est_mcu_time,
            'loss_m0': l0, 'loss_m1': l1, 'loss_m2': l2,
            'dev_rh': dev_rh, 'dev_rv': dev_rv, 'dev_mean': dev_mean,
            'mape_rh': mape_rh, 'mape_rv': mape_rv
        })

        print(f"{name:<28} | {total_steps:<6} | {est_mcu_time:<6.2f} с  | {l2:<9.5f} | {dev_mean:<15.2f}% | Rh: {mape_rh:.2f}%, Rv: {mape_rv:.2f}%")

    print("="*105)

    print(f'\n[5/5] Построение графиков сходимости: {out_png}...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    steps_x = list(range(1, 151))

    # График 1: Функция потерь от шага
    axes[0].plot(steps_x, loss_m0, label='M0 (0 границ)', color='blue', lw=2)
    axes[0].plot(steps_x, loss_m1, label='M1 (1 граница)', color='green', lw=2)
    axes[0].plot(steps_x, loss_m2, label='M2 (2 границы)', color='red', lw=2)
    axes[0].set_yscale('log')
    axes[0].set_title('Сходимость функции потерь Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Шаг Adam PGD')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].axvline(x=30, color='gray', linestyle=':', label='30 шагов')
    axes[0].axvline(x=50, color='black', linestyle=':', label='50 шагов')

    # График 2: Отклонение параметров от эталона 150 шагов
    snap_steps = [15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150]
    devs_m0 = []
    devs_m1 = []
    for s in snap_steps:
        d0 = torch.mean(torch.abs(10**snap_m0[s][:, 1] - 10**res_m0_150[:, 1]) / 10**res_m0_150[:, 1]).item() * 100.0
        d1 = torch.mean(torch.abs(10**snap_m1[s][:, 1] - 10**res_m1_150[:, 1]) / 10**res_m1_150[:, 1]).item() * 100.0
        devs_m0.append(d0)
        devs_m1.append(d1)

    axes[1].plot(snap_steps, devs_m0, 'o-', label='M0: Rh пласта', color='blue', lw=2)
    axes[1].plot(snap_steps, devs_m1, 's-', label='M1: Rh пласта', color='green', lw=2)
    axes[1].set_title('Отклонение от эталона 150 шагов (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Количество шагов')
    axes[1].set_ylabel('Расхождение (%)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Порог 1%')

    # График 3: Компромисс Скорость vs Точность на MCU
    times = [r['time_mcu'] for r in eval_results]
    devs = [r['dev_mean'] for r in eval_results]
    names = [r['name'].split()[0] for r in eval_results]

    axes[2].scatter(times, devs, color='purple', s=80, zorder=5)
    for i, txt in enumerate(names):
        axes[2].annotate(f"{txt}\n({times[i]:.1f}с, {devs[i]:.2f}%)", (times[i]+0.5, devs[i]+0.05), fontsize=9)
    axes[2].plot(times, devs, linestyle='--', color='purple', alpha=0.6)
    axes[2].set_title('Компромисс: Время на MCU vs Погрешность', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Расчетное время точки на MCU STM32H750 (секунды)')
    axes[2].set_ylabel('Среднее расхождение с эталоном (%)')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f'[OK] График сохранен: {out_png}')

if __name__ == '__main__':
    main()

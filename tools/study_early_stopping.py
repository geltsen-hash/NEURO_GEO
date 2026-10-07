#!/usr/bin/env python3
"""
Исследование динамического раннего останова (Early Stopping) с max_steps=150.
Анализ распределения шагов, невязки MSE и нормы градиентов по скважинному логу
для формирования оптимального правила останова на микроконтроллере STM32H750.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    synthlog_bin = os.path.join(base_dir, 'test', 'SYNTHLOG.BIN')
    out_png = os.path.join(base_dir, 'test', 'early_stopping_study.png')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[1/4] Инициализация моделей (Device: {device})...')

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

    mean_X_np = np.loadtxt(os.path.join(weights_dir, 'scaler_X_mean.txt'), dtype=np.float32)
    scale_X_np = np.loadtxt(os.path.join(weights_dir, 'scaler_X_scale.txt'), dtype=np.float32)
    mean_Y_np = np.loadtxt(os.path.join(weights_dir, 'scaler_Y_mean.txt'), dtype=np.float32)
    scale_Y_np = np.loadtxt(os.path.join(weights_dir, 'scaler_Y_scale.txt'), dtype=np.float32)

    mean_X = torch.from_numpy(mean_X_np).to(device)
    scale_X = torch.from_numpy(scale_X_np).to(device)
    mean_Y = torch.from_numpy(mean_Y_np).to(device)
    scale_Y = torch.from_numpy(scale_Y_np).to(device)

    # Загрузка точек из SYNTHLOG.BIN
    sys.path.insert(0, base_dir)
    from tools.run_autonomous_inversion import load_synthetic_log
    points = load_synthetic_log(synthlog_bin)
    num_pts = len(points)
    print(f'[2/4] Загружено точек: {num_pts}')

    md_arr = np.array([p['md'] for p in points], dtype=np.float32)
    raw_signals = np.array([p['signals'] for p in points], dtype=np.float32)
    
    # Нормализация
    signals_log = raw_signals.copy()
    for idx in AMP_INDICES:
        signals_log[:, idx] = np.log10(np.maximum(signals_log[:, idx], 1e-8))
    y_scaled_np = (signals_log - mean_Y_np) / scale_Y_np
    y_true_all = torch.from_numpy(y_scaled_np).to(device)

    min_b_np = np.array([RO_MIN, RO_MIN, RO_MIN, RO_MIN, D_MIN, D_MIN, ANGLE_MIN], dtype=np.float32)
    max_b_np = np.array([RO_MAX, RO_MAX, RO_MAX, RO_MAX, D_MAX, D_MAX, ANGLE_MAX], dtype=np.float32)
    for idx in X_LOG_INDICES:
        min_b_np[idx] = np.log10(np.maximum(min_b_np[idx], 1e-5))
        max_b_np[idx] = np.log10(np.maximum(max_b_np[idx], 1e-5))
    min_scaled = torch.from_numpy((min_b_np - mean_X_np) / scale_X_np).to(device)
    max_scaled = torch.from_numpy((max_b_np - mean_X_np) / scale_X_np).to(device)

    with torch.no_grad():
        opt_geo_init_all = inv_net(y_true_all)
        opt_geo_init_all = torch.clamp(opt_geo_init_all, min_scaled, max_scaled)

    sig_amp = torch.std(y_true_all[:, AMP_INDICES], dim=1)
    active_mask_all = (sig_amp > NOISE_THRESHOLD).float().unsqueeze(1)

    print(f'[3/4] Моделирование оптимизации Adam (потолок 150 шагов) для всех 151 точек...')

    # Функция симуляции для одной точки
    def optimize_point(pt_idx, topo_mode, max_steps=150, m1_real=None):
        y_true = y_true_all[pt_idx:pt_idx+1]
        active = active_mask_all[pt_idx:pt_idx+1]
        geo = opt_geo_init_all[pt_idx:pt_idx+1].clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([geo], lr=0.015, betas=(0.9, 0.999), eps=1e-8)

        up_closer = False
        if topo_mode == 2 and m1_real is not None:
            m1_scaled = (m1_real - mean_X) / scale_X
            up_closer = (m1_real[4] <= m1_real[5]).item()
            with torch.no_grad():
                if up_closer:
                    geo[0, 4] = m1_scaled[4]
                    geo[0, 0] = m1_scaled[0]
                else:
                    geo[0, 5] = m1_scaled[5]
                    geo[0, 3] = m1_scaled[3]

        losses = []
        grad_norms = []
        step_norms = []
        geo_prev = geo.detach().clone()

        for step in range(1, max_steps + 1):
            opt.zero_grad()
            y_pred = fwd_net(geo)
            loss = torch.mean((y_pred - y_true)**2)
            loss.backward()

            if topo_mode >= 0 and not active.item():
                geo.grad.data[0, 4:6] = 0.0
            if topo_mode == 2 and m1_real is not None:
                if up_closer:
                    geo.grad.data[0, 4] = 0.0
                    geo.grad.data[0, 0] = 0.0
                else:
                    geo.grad.data[0, 5] = 0.0
                    geo.grad.data[0, 3] = 0.0

            gnorm = torch.norm(geo.grad.data).item()
            grad_norms.append(gnorm)
            losses.append(loss.item())

            opt.step()

            with torch.no_grad():
                geo.data = torch.clamp(geo.data, min_scaled, max_scaled)
                real_fix = geo.data * scale_X + mean_X

                if topo_mode == 0:
                    real_fix[0, 4] = D_MAX
                    real_fix[0, 5] = D_MAX
                    real_fix[0, 0] = real_fix[0, 1]
                    real_fix[0, 3] = real_fix[0, 1]
                elif topo_mode == 1:
                    if real_fix[0, 4] <= real_fix[0, 5]:
                        real_fix[0, 5] = D_MAX
                        real_fix[0, 3] = real_fix[0, 1]
                    else:
                        real_fix[0, 4] = D_MAX
                        real_fix[0, 0] = real_fix[0, 1]
                elif topo_mode == 2 and m1_real is not None:
                    if up_closer:
                        real_fix[0, 4] = m1_real[4]
                        real_fix[0, 0] = m1_real[0]
                    else:
                        real_fix[0, 5] = m1_real[5]
                        real_fix[0, 3] = m1_real[3]

                geo.data = (real_fix - mean_X) / scale_X
                snorm = torch.norm(geo.data - geo_prev).item()
                step_norms.append(snorm)
                geo_prev = geo.detach().clone()

        real_final = (geo.data * scale_X + mean_X).detach().squeeze(0)
        return losses, grad_norms, step_norms, real_final

    # Выполним трассировку для 3 характерных точек каротажа:
    # 1. Простая точка во вмещающих породах (MD = 1.0 м, pt 10)
    # 2. Сложная точка на границе пласта (MD = 5.0 м, pt 50)
    # 3. Точка внутри тонкого анизотропного пласта (MD = 6.5 м, pt 65)
    test_points = [
        (10, 1.0, "Вмещающие породы (MD=1.0м)"),
        (50, 5.0, "Кровля пласта (MD=5.0м)"),
        (65, 6.5, "Внутри пласта (MD=6.5м)")
    ]

    fig, axs = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle('Анализ сходимости Adam: Loss, Grad Norm и Step Norm (до 150 шагов)', fontsize=14)

    for col_idx, (pt_i, md_val, title_str) in enumerate(test_points):
        # M0
        l0, g0, s0, r0 = optimize_point(pt_i, 0, 150)
        # M1
        l1, g1, s1, r1 = optimize_point(pt_i, 1, 150)
        # M2
        l2, g2, s2, r2 = optimize_point(pt_i, 2, 150, m1_real=r1)

        # Plot Loss
        axs[0, col_idx].plot(l0, label='M0 (0 гр)', color='blue')
        axs[0, col_idx].plot(l1, label='M1 (1 гр)', color='green')
        axs[0, col_idx].plot(l2, label='M2 (2 гр)', color='red')
        axs[0, col_idx].set_yscale('log')
        axs[0, col_idx].set_title(f"{title_str}\nLoss (MSE по сигналам)")
        axs[0, col_idx].grid(True, alpha=0.3)
        if col_idx == 0:
            axs[0, col_idx].set_ylabel('Loss (MSE)')
            axs[0, col_idx].legend()

        # Plot Grad Norm
        axs[1, col_idx].plot(g0, color='blue', alpha=0.8)
        axs[1, col_idx].plot(g1, color='green', alpha=0.8)
        axs[1, col_idx].plot(g2, color='red', alpha=0.8)
        axs[1, col_idx].set_yscale('log')
        axs[1, col_idx].set_title("Норма градиента ||∇x||")
        axs[1, col_idx].grid(True, alpha=0.3)
        if col_idx == 0:
            axs[1, col_idx].set_ylabel('||∇x||')

        # Plot Step Norm
        axs[2, col_idx].plot(s0, color='blue', alpha=0.8)
        axs[2, col_idx].plot(s1, color='green', alpha=0.8)
        axs[2, col_idx].plot(s2, color='red', alpha=0.8)
        axs[2, col_idx].set_yscale('log')
        axs[2, col_idx].set_title("Норма шага ||Δx||")
        axs[2, col_idx].grid(True, alpha=0.3)
        axs[2, col_idx].set_xlabel('Номер шага Adam')
        if col_idx == 0:
            axs[2, col_idx].set_ylabel('||Δx||')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f'[OK] График исследования сохранен: {out_png}')

    # Оценка распределения необходимых шагов по всему логу при различных критериях
    print("[4/4] Статистическая оценка порога сходимости по всем 151 точкам...")
    
    # Критерий:
    # 1. min_steps = 10 для M0, 15 для M1/M2
    # 2. Early stop if |Loss[t] - Loss[t-5]| < eps_loss OR ||∇x|| < eps_grad
    # Проверим разные пороги
    thresholds = [
        {"name": "Консервативный", "min_m0": 15, "min_m": 25, "eps_loss": 1e-4, "eps_grad": 0.005},
        {"name": "Сбалансированный", "min_m0": 10, "min_m": 20, "eps_loss": 5e-4, "eps_grad": 0.010},
        {"name": "Агрессивный", "min_m0": 8,  "min_m": 15, "eps_loss": 1e-3, "eps_grad": 0.020},
    ]

    for th in thresholds:
        total_steps_list = []
        for i in range(num_pts):
            # Симуляция M0
            l0, g0, _, r0 = optimize_point(i, 0, 150)
            step_m0 = 150
            for s in range(th["min_m0"], 150):
                if (abs(l0[s] - l0[s-5]) < th["eps_loss"]) or (g0[s] < th["eps_grad"]):
                    step_m0 = s
                    break

            # Симуляция M1
            l1, g1, _, r1 = optimize_point(i, 1, 150)
            step_m1 = 150
            for s in range(th["min_m"], 150):
                if (abs(l1[s] - l1[s-5]) < th["eps_loss"]) or (g1[s] < th["eps_grad"]):
                    step_m1 = s
                    break

            # Симуляция M2
            l2, g2, _, r2 = optimize_point(i, 2, 150, m1_real=r1)
            step_m2 = 150
            for s in range(th["min_m"], 150):
                if (abs(l2[s] - l2[s-5]) < th["eps_loss"]) or (g2[s] < th["eps_grad"]):
                    step_m2 = s
                    break

            total_steps = step_m0 + step_m1 + step_m2
            total_steps_list.append((step_m0, step_m1, step_m2, total_steps))

        steps_arr = np.array([x[3] for x in total_steps_list])
        avg_steps = steps_arr.mean()
        min_steps_all = steps_arr.min()
        max_steps_all = steps_arr.max()
        time_mcu_avg = avg_steps * 0.0883 # 88.3 мс на шаг на MCU

        print(f"\n--- Профиль: {th['name']} ---")
        print(f"  Среднее число шагов: {avg_steps:.1f} (мин {min_steps_all}, макс {max_steps_all})")
        print(f"  Ожидаемое среднее время на MCU: {time_mcu_avg:.2f} с / точка")
        print(f"  Время всего каротажа (151 точка): {time_mcu_avg * 151 / 60:.1f} мин (сейчас 23.3 мин)")

if __name__ == '__main__':
    main()

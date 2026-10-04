"""Калибровочный скрипт по предыдущей версии ТЗ (без корпусной поправки C/A)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model as m
from scipy.optimize import minimize_scalar
from pathlib import Path


def load_csv(path):
    df = pd.read_csv(path)
    heights = np.array(sorted(df['height'].unique()), dtype=np.float64)
    n_heights = len(heights)
    n_channels = len(m.CHANNELS)
    arr = np.zeros((n_heights, n_channels, 3), dtype=np.float64)
    for i, h in enumerate(heights):
        for j in range(n_channels):
            row = df[(df['height'] == h) & (df['channel'] == j)]
            if len(row) != 1:
                raise ValueError(f"Не найдена ровно одна строка для height={h}, channel={j}")
            arr[i, j, 0] = row['H0'].values[0]
            arr[i, j, 1] = row['H1'].values[0]
            arr[i, j, 2] = row['H2'].values[0]
    return heights, arr


def theoretical_harmonics_at_L(height, L_eff, tx_tilt, rx_tilt):
    samples = np.zeros((16, 2), dtype=np.float64)
    for i, phi_deg in enumerate(m.ROTATION_ANGLES_DEG):
        rot = m.rotation_xz_to_lab(np.deg2rad(phi_deg))
        tx_m = rot @ m.moment_xz_from_tilt(tx_tilt)
        rx_m = rot @ m.moment_xz_from_tilt(rx_tilt)
        direct_r = np.array([0.0, 0.0, L_eff], dtype=np.float64)
        image_r = np.array([0.0, 2.0 * height, L_eff], dtype=np.float64)
        direct_signal = rx_m @ m.dipole_tensor_3d(direct_r) @ tx_m
        image_signal = rx_m @ m.dipole_tensor_3d(image_r) @ (m.IMAGE_MOMENT_REFLECTION @ tx_m)
        samples[i, 0] = direct_signal + image_signal
        samples[i, 1] = 0.0
    return m.instrument_harmonics_from_iq(samples[:, 0], samples[:, 1])


def channel_base(tx_name, rx_name):
    return abs(m.TRANSMITTERS_Z[tx_name] - m.RECEIVERS_Z[rx_name])


MIRROR_PAIRS = [(0, 5), (3, 7), (1, 4), (2, 6)]


def _fit_single_angle(heights, arr, j):
    tx_name, rx_name = m.CHANNELS[j]
    L_eff = channel_base(tx_name, rx_name)

    def objective(angle_deg):
        rx_tilt = np.tan(np.deg2rad(angle_deg))
        errs = []
        for i, h in enumerate(heights):
            H = theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)
            H0_t, H1_t = H[0], H[1]
            ratio_t = H1_t / H0_t
            ratio_m = arr[i, j, 1] / arr[i, j, 0]
            errs.append(ratio_t - ratio_m)
        return np.sum(np.array(errs) ** 2)

    result = minimize_scalar(objective, bounds=(10.0, 89.0), method='bounded')
    return result.x


def stage2_fit_angles(heights, arr):
    angles = np.zeros(len(m.CHANNELS), dtype=np.float64)
    for j1, j2 in MIRROR_PAIRS:
        tx1, rx1 = m.CHANNELS[j1]
        tx2, rx2 = m.CHANNELS[j2]
        L1 = channel_base(tx1, rx1)
        L2 = channel_base(tx2, rx2)
        L = (L1 + L2) / 2.0
        angles[j1] = _fit_single_angle(heights, arr, j1)
        angles[j2] = _fit_single_angle(heights, arr, j2)
        print(f"\n  L={L:.2f} м:")
        print(f"    ch{j1} ({tx1}-{rx1}): beta = {angles[j1]:.2f} град")
        print(f"    ch{j2} ({tx2}-{rx2}): beta = {angles[j2]:.2f} град")
        print(f"    разница: {abs(angles[j1] - angles[j2]):.2f} град")
    return angles


def report_for_channel(heights, arr, angles, j):
    tx_name, rx_name = m.CHANNELS[j]
    L = channel_base(tx_name, rx_name)
    L_eff = L
    rx_tilt = np.tan(np.deg2rad(angles[j]))
    rows = []
    for i, h in enumerate(heights):
        H = theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)
        ratio_t = H[1] / H[0]
        ratio_m = arr[i, j, 1] / arr[i, j, 0]
        abs_err = ratio_t - ratio_m
        rel_err_pct = abs_err / ratio_m * 100.0 if ratio_m != 0 else 0.0
        rows.append((h, h / L, ratio_t, ratio_m, abs_err, rel_err_pct))
    return L, rows


def stage2_report(heights, arr, angles):
    print("\n--- Отчёт по H1/H0 после подбора углов (без корпусной поправки) ---")
    all_abs_errs = []
    all_rel_errs = []
    for j1, j2 in MIRROR_PAIRS:
        tx1, rx1 = m.CHANNELS[j1]
        L1 = channel_base(tx1, rx1)
        tx2, rx2 = m.CHANNELS[j2]
        L2 = channel_base(tx2, rx2)
        L = (L1 + L2) / 2.0
        print(f"\nПара ch{j1}-ch{j2} (L={L:.2f} м):")
        for j in (j1, j2):
            tx_name, rx_name = m.CHANNELS[j]
            L_j, rows = report_for_channel(heights, arr, angles, j)
            abs_errs = [abs(r[4]) for r in rows]
            rel_errs = [abs(r[5]) for r in rows]
            print(f"  ch{j} ({tx_name}-{rx_name}, beta={angles[j]:.2f} град): "
                  f"mean|ошибка| = {np.mean(abs_errs)*100:.2f}%, max = {np.max(abs_errs)*100:.2f}% "
                  f"| относит.: mean = {np.mean(rel_errs):.2f}%, max = {np.max(rel_errs):.2f}%")
            for h, hl, rt, rm, ae, re in rows:
                print(f"    {h:7.3f} {hl:6.2f} {rt:12.5f} {rm:12.5f} {ae*100:10.2f} {re:10.2f}")
            all_abs_errs.extend(abs_errs)
            all_rel_errs.extend(rel_errs)
        diff_beta = abs(angles[j1] - angles[j2])
        print(f"  разница углов: {diff_beta:.2f} град")
    print(f"\nИтого (абсолютная): mean|ошибка| = {np.mean(all_abs_errs)*100:.2f}%, max = {np.max(all_abs_errs)*100:.2f}%")
    print(f"Итого (относительная): mean = {np.mean(all_rel_errs):.2f}%, max = {np.max(all_rel_errs):.2f}%\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Калибровка по базовой модели (без поправки C/A)')
    parser.add_argument('csv', nargs='?', help='Путь к CSV-файлу с измерениями')
    args = parser.parse_args()

    if args.csv:
        path = args.csv
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title='Выберите CSV-файл с измерениями',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            initialdir=str(Path(__file__).parent),
        )
        root.destroy()
        if not path:
            print('Файл не выбран')
            return

    print(f'Загрузка: {path}')
    heights, arr = load_csv(path)
    print(f'Высоты: {heights}')
    print(f'Каналов: {arr.shape[1]}\n')

    print('=== Подбор углов (L_eff = L, без поправки C/A) ===')
    angles = stage2_fit_angles(heights, arr)
    stage2_report(heights, arr, angles)


if __name__ == '__main__':
    main()

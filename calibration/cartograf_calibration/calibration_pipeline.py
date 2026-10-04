import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model as m
from scipy.optimize import minimize_scalar
from pathlib import Path


# ---------------------------------------------------------------------------
# Каналы и геометрия
# ---------------------------------------------------------------------------

# Tx1 = Tzz1 (z=-0.48), Tx2 = Tzz2 (z=+0.84), Tx3 = Tzz4/Tzz3 (z=1.56/-1.20), Tx4 = Tzz5 (z=+0.24)
# Rzx1 (L) находится на z=-1.44, Rzx2 (R) находится на z=+1.80
CHANNEL_INFO_NEW = [
    ('L_Tx1', 'Rzx1', 0.96, 'L', 1),  # ch0: Tzz1-Rzx1 (L = 0.96)
    ('L_Tx2', 'Rzx1', 2.28, 'L', 2),  # ch1: Tzz2-Rzx1 (L = 2.28)
    ('L_Tx3', 'Rzx1', 3.00, 'L', 3),  # ch2: Tzz4-Rzx1 (L = 3.00)
    ('L_Tx4', 'Rzx1', 1.68, 'L', 4),  # ch3: Tzz5-Rzx1 (L = 1.68)
    ('R_Tx1', 'Rzx2', 2.28, 'R', 1),  # ch4: Tzz1-Rzx2 (L = 2.28)
    ('R_Tx2', 'Rzx2', 0.96, 'R', 2),  # ch5: Tzz2-Rzx2 (L = 0.96)
    ('R_Tx3', 'Rzx2', 3.00, 'R', 3),  # ch6: Tzz3-Rzx2 (L = 3.00)
    ('R_Tx4', 'Rzx2', 1.68, 'R', 4),  # ch7: Tzz5-Rzx2 (L = 1.68)
]

# Зеркальные пары равной длины базы:
# L=0.96: ch0 (L_Tx1) - ch5 (R_Tx2)
# L=1.68: ch3 (L_Tx4) - ch7 (R_Tx4)
# L=2.28: ch1 (L_Tx2) - ch4 (R_Tx1)
# L=3.00: ch2 (L_Tx3) - ch6 (R_Tx3)
MIRROR_PAIRS_NEW = [(0, 5), (3, 7), (1, 4), (2, 6)]
MIRROR_PAIRS_LEGACY = [(0, 5), (3, 7), (1, 4), (2, 6)]


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_csv(path):
    """
    Загрузка данных калибровки. Поддерживает как старый, так и новый формат.
    Возвращает словарь по частотам: {freq: (heights, arr, d_ph, chan_info, mirror_pairs)}
    """
    df = pd.read_csv(path)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # Новый формат
    if 'D_m' in df.columns and 'amp_V_zz' in df.columns:
        heights = np.array(sorted(df['D_m'].unique()), dtype=np.float64)
        n_heights = len(heights)
        freqs = sorted(df['freq'].unique()) if 'freq' in df.columns else [None]
        results = {}

        chan_tuples = [(name, rx, L) for (name, rx, L, _, _) in CHANNEL_INFO_NEW]

        for freq in freqs:
            sub_df = df[df['freq'] == freq] if freq is not None else df
            arr = np.zeros((n_heights, 8, 3), dtype=np.float64)
            d_ph = np.zeros((n_heights, 8), dtype=np.float64)

            for i, h in enumerate(heights):
                for j, (name, rx_name, L_val, rx_letter, tx_num) in enumerate(CHANNEL_INFO_NEW):
                    row = sub_df[(sub_df['D_m'] == h) & (sub_df['Rx'] == rx_letter) & (sub_df['Tx'] == tx_num)]
                    if len(row) != 1:
                        raise ValueError(f"Не найдена ровно одна строка для D_m={h}, Rx={rx_letter}, Tx={tx_num}, freq={freq}")

                    arr[i, j, 0] = row['amp_V_zz'].values[0]
                    arr[i, j, 1] = row['amp_V_zx'].values[0]
                    arr[i, j, 2] = row['amp_V_xx'].values[0] if 'amp_V_xx' in row else 0.0
                    d_ph[i, j] = row['d_ph_Vzx_Vzz'].values[0] if 'd_ph_Vzx_Vzz' in row else 0.0

            results[freq] = (heights, arr, d_ph, chan_tuples, MIRROR_PAIRS_NEW)
        return results

    # Старый формат
    elif 'height' in df.columns and 'channel' in df.columns:
        heights = np.array(sorted(df['height'].unique()), dtype=np.float64)
        n_heights = len(heights)
        n_channels = len(m.CHANNELS)
        arr = np.zeros((n_heights, n_channels, 3), dtype=np.float64)
        d_ph = np.zeros((n_heights, n_channels), dtype=np.float64)
        chan_info = [(tx, rx, abs(m.TRANSMITTERS_Z[tx] - m.RECEIVERS_Z[rx])) for tx, rx in m.CHANNELS]

        for i, h in enumerate(heights):
            for j in range(n_channels):
                row = df[(df['height'] == h) & (df['channel'] == j)]
                if len(row) != 1:
                    raise ValueError(f"Не найдена ровно одна строка для height={h}, channel={j}")
                arr[i, j, 0] = row['H0'].values[0]
                arr[i, j, 1] = row['H1'].values[0]
                arr[i, j, 2] = row['H2'].values[0]

        return {None: (heights, arr, d_ph, chan_info, MIRROR_PAIRS_LEGACY)}

    else:
        raise ValueError(f"Неизвестный формат CSV: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Теоретические расчёты
# ---------------------------------------------------------------------------

def theoretical_harmonics_at_L(height, L_eff, tx_tilt, rx_tilt):
    """Расчёт H0, H1, H2 для точечного диполя с базой L_eff и наклонами."""
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


# ---------------------------------------------------------------------------
# Корпусная добавка (C/A) и подбор углов
# ---------------------------------------------------------------------------

def air_h1(h, L):
    """Геометрическая функция отклика отражённого сигнала ZX от границы."""
    return h * L / (L**2 + 4.0 * h**2)**2.5


def fit_casing_offset(heights, arr, chan_info):
    """Определение масштабного коэффициента A_j, корпусной добавки C_j и отношения c_j = C/A."""
    n_channels = len(chan_info)
    A_vec = np.zeros(n_channels)
    C_vec = np.zeros(n_channels)
    c_ratio_vec = np.zeros(n_channels)

    for j in range(n_channels):
        _, _, L = chan_info[j]
        H1_meas = arr[:, j, 1]
        X = air_h1(heights, L)
        w = 1.0 / (H1_meas**2)

        Sxx = np.sum(w * X * X)
        Sx = np.sum(w * X)
        Sxy = np.sum(w * X * H1_meas)
        Sy = np.sum(w * H1_meas)
        S = np.array([[Sxx, Sx], [Sx, np.sum(w)]])
        rhs = np.array([Sxy, Sy])

        try:
            A, C = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            A = np.sum(w * H1_meas * X) / np.sum(w * X * X)
            C = 0.0

        A_vec[j] = A
        C_vec[j] = C
        c_ratio_vec[j] = C / A if A != 0 else 0.0

    return A_vec, C_vec, c_ratio_vec


def _fit_single_angle(heights, arr, j, chan_info, C_offset=0.0):
    """Подбор угла для одного канала j по всем доступным высотам со скорректированным H1."""
    _, _, L_eff = chan_info[j]

    def objective(angle_deg):
        rx_tilt = np.tan(np.deg2rad(angle_deg))
        errs = []
        for i, h in enumerate(heights):
            H = theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)
            ratio_t = H[1] / H[0]
            ratio_m = (arr[i, j, 1] - C_offset) / arr[i, j, 0]
            errs.append(ratio_t - ratio_m)
        return np.sum(np.array(errs) ** 2)

    result = minimize_scalar(objective, bounds=(10.0, 89.0), method='bounded')
    return result.x


def stage2_fit_angles(heights, arr, chan_info, mirror_pairs, C_vec=None):
    angles = np.zeros(len(chan_info), dtype=np.float64)
    for j1, j2 in mirror_pairs:
        tx1, rx1, L1 = chan_info[j1]
        tx2, rx2, L2 = chan_info[j2]
        L = (L1 + L2) / 2.0
        C1 = C_vec[j1] if C_vec is not None else 0.0
        C2 = C_vec[j2] if C_vec is not None else 0.0
        angles[j1] = _fit_single_angle(heights, arr, j1, chan_info, C1)
        angles[j2] = _fit_single_angle(heights, arr, j2, chan_info, C2)
        print(f"\n  L={L:.2f} м:")
        print(f"    ch{j1} ({tx1}-{rx1}): beta = {angles[j1]:.2f} град")
        print(f"    ch{j2} ({tx2}-{rx2}): beta = {angles[j2]:.2f} град")
        print(f"    разница: {abs(angles[j1] - angles[j2]):.2f} град")
    return angles


# ---------------------------------------------------------------------------
# Отчёт и графики
# ---------------------------------------------------------------------------

NOMINAL_BETA_DEG = 45.0
TAN_NOMINAL_BETA = np.tan(np.deg2rad(NOMINAL_BETA_DEG))


def report_for_channel(heights, arr, d_ph, angles, j, chan_info, C_val=0.0, c_val=0.0):
    tx_name, rx_name, L = chan_info[j]
    L_eff = L
    beta_rad = np.deg2rad(angles[j])
    rx_tilt = np.tan(beta_rad)
    mean_dph = np.mean(d_ph[:, j]) if d_ph is not None else 0.0

    rows_h1 = []
    rows_geo = []

    for i, h in enumerate(heights):
        # 1. H1/H0
        H = theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)
        ratio_t = H[1] / H[0]
        ratio_m = (arr[i, j, 1] - C_val) / arr[i, j, 0]
        abs_err_h1 = ratio_t - ratio_m
        rel_err_h1_pct = abs_err_h1 / ratio_m * 100.0 if ratio_m != 0 else 0.0
        rows_h1.append((h, h / L, ratio_t, ratio_m, abs_err_h1, rel_err_h1_pct))

        # 2. Геосигнал (для идеальной 45-градусной антенны)
        H_nom = theoretical_harmonics_at_L(h, L_eff, 0.0, TAN_NOMINAL_BETA)
        R_theor = H_nom[1] / H_nom[0]
        geo_t = (1.0 - R_theor) / (1.0 + R_theor)

        # Экспериментальный R_pure с учетом фазы, корпуса и масштабирования tan(45)/tan(beta)
        R_raw = arr[i, j, 1] / arr[i, j, 0]
        ph_i = d_ph[i, j] if d_ph is not None else 0.0
        R_ph_corr = R_raw * np.exp(-1j * (ph_i - mean_dph)) if d_ph is not None else R_raw
        R_corr = np.real(R_ph_corr) - c_val
        R_pure = R_corr * (TAN_NOMINAL_BETA / rx_tilt)
        geo_m = (1.0 - R_pure) / (1.0 + R_pure)

        abs_err_geo = geo_t - geo_m
        rel_err_geo_pct = abs_err_geo / geo_m * 100.0 if geo_m != 0 else 0.0
        rows_geo.append((h, h / L, geo_t, geo_m, abs_err_geo, rel_err_geo_pct))

    return L, rows_h1, rows_geo


def stage2_report(heights, arr, d_ph, angles, chan_info, mirror_pairs, C_vec=None, c_ratio_vec=None, freq=None, out_txt_path=None):
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    freq_str = f" ({freq} кГц)" if freq is not None else ""
    log(f"\n=======================================================")
    log(f"--- Отчёт по калибровке и геосигналу{freq_str} ---")
    log(f"=======================================================")

    if c_ratio_vec is not None:
        log("\nПараметры каналов (корпусная константа c_j = C/A и разность фаз d_ph):")
        log(f"{'ch':>3s} {'Пара Tx-Rx':>12s} {'L, м':>6s} {'c_j (C/A)':>14s} {'C_j (абс.)':>14s} {'d_ph, рад':>12s} {'d_ph, град':>12s}")
        log("-" * 80)
        for j in range(len(chan_info)):
            tx_name, rx_name, L_j = chan_info[j]
            c_val = c_ratio_vec[j]
            C_val = C_vec[j] if C_vec is not None else 0.0
            mean_dph_rad = np.mean(d_ph[:, j]) if d_ph is not None else 0.0
            mean_dph_deg = np.rad2deg(mean_dph_rad)
            log(f"{j:>3d} {tx_name+'-'+rx_name:>12s} {L_j:>6.2f} {c_val:>14.3e} {C_val:>14.3e} {mean_dph_rad:>12.4f} {mean_dph_deg:>12.2f}")

    all_abs_errs_h1 = []
    all_rel_errs_h1 = []
    all_abs_errs_geo = []
    all_rel_errs_geo = []

    log("\n-------------------------------------------------------")
    log("1. Сходимость отношений гармоник H1/H0 (калибровка углов beta)")
    log("-------------------------------------------------------")

    for j1, j2 in mirror_pairs:
        tx1, rx1, L1 = chan_info[j1]
        tx2, rx2, L2 = chan_info[j2]
        L = (L1 + L2) / 2.0

        log(f"\nПара ch{j1}-ch{j2} (L={L:.2f} м):")
        for j in (j1, j2):
            tx_name, rx_name, _ = chan_info[j]
            C_val = C_vec[j] if C_vec is not None else 0.0
            c_val = c_ratio_vec[j] if c_ratio_vec is not None else 0.0
            L_j, rows_h1, rows_geo = report_for_channel(heights, arr, d_ph, angles, j, chan_info, C_val, c_val)
            abs_errs = [abs(r[4]) for r in rows_h1]
            rel_errs = [abs(r[5]) for r in rows_h1]
            log(f"  ch{j} ({tx_name}-{rx_name}, beta={angles[j]:.2f} град): "
                f"mean|ошибка| = {np.mean(abs_errs)*100:.2f}%, max = {np.max(abs_errs)*100:.2f}% "
                f"| относит.: mean = {np.mean(rel_errs):.2f}%, max = {np.max(rel_errs):.2f}%")
            log(f"    {'h, м':>7s} {'h/L':>6s} {'H1/H0 theor':>12s} {'H1/H0 meas':>12s} {'abs.err,%':>10s} {'rel.err,%':>10s}")
            for h, hl, rt, rm, ae, re in rows_h1:
                log(f"    {h:7.3f} {hl:6.2f} {rt:12.5f} {rm:12.5f} {ae*100:10.2f} {re:10.2f}")
            all_abs_errs_h1.extend(abs_errs)
            all_rel_errs_h1.extend(rel_errs)
        diff_beta = abs(angles[j1] - angles[j2])
        log(f"  разница углов: {diff_beta:.2f} град")

    log("\n-------------------------------------------------------")
    log("2. Сходимость Геосигнала Geo = (1-R)/(1+R) (приведён к 45 град)")
    log("-------------------------------------------------------")

    for j1, j2 in mirror_pairs:
        tx1, rx1, L1 = chan_info[j1]
        tx2, rx2, L2 = chan_info[j2]
        L = (L1 + L2) / 2.0

        log(f"\nПара ch{j1}-ch{j2} (L={L:.2f} м):")
        for j in (j1, j2):
            tx_name, rx_name, _ = chan_info[j]
            C_val = C_vec[j] if C_vec is not None else 0.0
            c_val = c_ratio_vec[j] if c_ratio_vec is not None else 0.0
            L_j, rows_h1, rows_geo = report_for_channel(heights, arr, d_ph, angles, j, chan_info, C_val, c_val)
            abs_errs = [abs(r[4]) for r in rows_geo]
            rel_errs = [abs(r[5]) for r in rows_geo]
            log(f"  ch{j} ({tx_name}-{rx_name}): "
                f"mean|ошибка Geo| = {np.mean(abs_errs):.4f}, max = {np.max(abs_errs):.4f} "
                f"| относит.: mean = {np.mean(rel_errs):.2f}%, max = {np.max(rel_errs):.2f}%")
            log(f"    {'h, м':>7s} {'h/L':>6s} {'Geo theor':>12s} {'Geo meas':>12s} {'abs.err':>10s} {'rel.err,%':>10s}")
            for h, hl, gt, gm, ae, re in rows_geo:
                log(f"    {h:7.3f} {hl:6.2f} {gt:12.5f} {gm:12.5f} {ae:10.4f} {re:10.2f}")
            all_abs_errs_geo.extend(abs_errs)
            all_rel_errs_geo.extend(rel_errs)

    log(f"\nИтого H1/H0{freq_str} (абсолютная): mean|ошибка| = {np.mean(all_abs_errs_h1)*100:.2f}%, max = {np.max(all_abs_errs_h1)*100:.2f}%")
    log(f"Итого H1/H0{freq_str} (относительная): mean = {np.mean(all_rel_errs_h1):.2f}%, max = {np.max(all_rel_errs_h1):.2f}%")
    log(f"Итого Геосигнал{freq_str} (абсолютная): mean|ошибка| = {np.mean(all_abs_errs_geo):.4f}, max = {np.max(all_abs_errs_geo):.4f}")
    log(f"Итого Геосигнал{freq_str} (относительная): mean = {np.mean(all_rel_errs_geo):.2f}%, max = {np.max(all_rel_errs_geo):.2f}%\n")

    log(f"=== Итоговые параметры калибровки{freq_str} ===")
    log("1. Корпусные коэффициенты c_j = C/A (безразмерные):")
    for j in range(len(chan_info)):
        tx_name, rx_name, _ = chan_info[j]
        log(f"  ch{j} ({tx_name}-{rx_name}): c = {c_ratio_vec[j]:+.4e}")

    log("\n2. Углы beta по зеркальным парам (по возрастанию L):")
    for j1, j2 in mirror_pairs:
        tx1, rx1, L1 = chan_info[j1]
        tx2, rx2, L2 = chan_info[j2]
        L = (L1 + L2) / 2.0
        log(f"  L={L:.2f} м:")
        log(f"    ch{j1} ({tx1}-{rx1}): beta = {angles[j1]:.2f} град")
        log(f"    ch{j2} ({tx2}-{rx2}): beta = {angles[j2]:.2f} град")
        log(f"    разница: {abs(angles[j1] - angles[j2]):.2f} град")

    if out_txt_path:
        with open(out_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"\nТекстовый отчет сохранён: {out_txt_path}")

    return '\n'.join(lines)


def fit_beta_single_point(h, L, ratio_meas):
    """Подбор угла beta для одной конкретной высоты по отношению H1/H0."""
    def objective(angle_deg):
        rx_tilt = np.tan(np.deg2rad(angle_deg))
        H = theoretical_harmonics_at_L(h, L, 0.0, rx_tilt)
        ratio_t = H[1] / H[0]
        return (ratio_t - ratio_meas) ** 2

    res = minimize_scalar(objective, bounds=(10.0, 89.0), method='bounded')
    return res.x


def stage2_plot(heights, arr, d_ph, angles, chan_info, C_vec=None, c_ratio_vec=None, freq=None, out_png_path='calibration_stage2_report.png'):
    rx_groups = {
        'Rzx1': [j for j, (_, rx, _) in enumerate(chan_info) if rx == 'Rzx1'],
        'Rzx2': [j for j, (_, rx, _) in enumerate(chan_info) if rx == 'Rzx2'],
    }
    freq_title = f" ({freq} кГц)" if freq is not None else ""
    fig, axes = plt.subplots(4, 2, figsize=(13, 18), constrained_layout=True)

    # Строка 1: H1/H0
    for ax, (rx_name, channels) in zip(axes[0], rx_groups.items()):
        for j in channels:
            tx_name, rx_name_j, L_eff = chan_info[j]
            rx_tilt = np.tan(np.deg2rad(angles[j]))
            C_val = C_vec[j] if C_vec is not None else 0.0
            ratio_meas = (arr[:, j, 1] - C_val) / arr[:, j, 0]
            ratio_theor = np.array([theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)[1] /
                                    theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)[0] for h in heights])
            ax.plot(heights, ratio_meas, 'o-', label=f'ch{j} {tx_name} L={L_eff:.2f} beta={angles[j]:.1f}°')
            ax.plot(heights, ratio_theor, '--', alpha=0.7)
        ax.set_title(f'{rx_name}{freq_title} — H1/H0 (скорректированный)')
        ax.set_xlabel('h, м')
        ax.set_ylabel('H1/H0')
        ax.legend(fontsize=8)
        ax.grid(True)

    # Строка 2: Ошибка H1/H0
    for ax, (rx_name, channels) in zip(axes[1], rx_groups.items()):
        for j in channels:
            tx_name, rx_name_j, L_eff = chan_info[j]
            rx_tilt = np.tan(np.deg2rad(angles[j]))
            C_val = C_vec[j] if C_vec is not None else 0.0
            ratio_meas = (arr[:, j, 1] - C_val) / arr[:, j, 0]
            ratio_theor = np.array([theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)[1] /
                                    theoretical_harmonics_at_L(h, L_eff, 0.0, rx_tilt)[0] for h in heights])
            ax.plot(heights, (ratio_theor - ratio_meas) / ratio_meas * 100, 'o-', label=f'ch{j} {tx_name}')
        ax.set_title(f'{rx_name}{freq_title} — ошибка H1/H0, %')
        ax.set_xlabel('h, м')
        ax.set_ylabel('(theor - meas)/meas * 100')
        ax.legend(fontsize=8)
        ax.grid(True)

    # Строка 3: beta(h)
    for ax, (rx_name, channels) in zip(axes[2], rx_groups.items()):
        for j in channels:
            tx_name, rx_name_j, L_eff = chan_info[j]
            C_val = C_vec[j] if C_vec is not None else 0.0
            betas_h = []
            for i, h in enumerate(heights):
                ratio_m = (arr[i, j, 1] - C_val) / arr[i, j, 0]
                b_val = fit_beta_single_point(h, L_eff, ratio_m)
                betas_h.append(b_val)
            betas_h = np.array(betas_h)
            ax.plot(heights, betas_h, 'o-', label=f'ch{j} {tx_name} L={L_eff:.2f} (std={np.std(betas_h):.2f}°)')
            ax.axhline(angles[j], linestyle='--', alpha=0.5)
        ax.set_title(f'{rx_name}{freq_title} — угол beta(h), град')
        ax.set_xlabel('h, м')
        ax.set_ylabel('beta, град')
        ax.legend(fontsize=8)
        ax.grid(True)

    # Строка 4: Геосигнал Geo = (1-R)/(1+R) (приведенный к 45 град)
    for ax, (rx_name, channels) in zip(axes[3], rx_groups.items()):
        for j in channels:
            tx_name, rx_name_j, L_eff = chan_info[j]
            C_val = C_vec[j] if C_vec is not None else 0.0
            c_val = c_ratio_vec[j] if c_ratio_vec is not None else 0.0
            _, _, rows_geo = report_for_channel(heights, arr, d_ph, angles, j, chan_info, C_val, c_val)
            geo_t = [r[2] for r in rows_geo]
            geo_m = [r[3] for r in rows_geo]
            ax.plot(heights, geo_m, 'o-', label=f'ch{j} {tx_name} (изм)')
            ax.plot(heights, geo_t, '--', alpha=0.7)
        ax.set_title(f'{rx_name}{freq_title} — Геосигнал (приведён к 45°)')
        ax.set_xlabel('h, м')
        ax.set_ylabel('Geo = (1-R)/(1+R)')
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.savefig(out_png_path, dpi=150)
    print(f"График сохранён: {out_png_path}")


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Калибровка с учётом корпусной поправки c=C/A')
    parser.add_argument('csv', nargs='?', help='Путь к CSV-файлу с измерениями')
    parser.add_argument('--freq', type=int, default=None, help='Выбрать конкретную частоту (например, 400 или 2000)')
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
    data_dict = load_csv(path)

    all_reports = []

    for freq, (heights, arr, d_ph, chan_info, mirror_pairs) in data_dict.items():
        if args.freq is not None and freq != args.freq:
            continue

        freq_label = f"{freq}kHz" if freq is not None else "main"
        freq_print = f" {freq} кГц" if freq is not None else ""
        print(f'\n======================================================')
        print(f'=== Обработка данных{freq_print} ===')
        print(f'======================================================')
        print(f'Высоты: {heights}')
        print(f'Каналов: {arr.shape[1]}\n')

        print(f'=== Шаг 1: Расчёт корпусной поправки c_j = C/A{freq_print} ===')
        A_vec, C_vec, c_ratio_vec = fit_casing_offset(heights, arr, chan_info)

        print(f'\n=== Шаг 2: Подбор углов beta (со скорректированным H1){freq_print} ===')
        angles = stage2_fit_angles(heights, arr, chan_info, mirror_pairs, C_vec)

        txt_name = f'calibration_report_{freq_label}.txt' if len(data_dict) > 1 else 'calibration_report.txt'
        png_name = f'calibration_stage2_report_{freq_label}.png' if len(data_dict) > 1 else 'calibration_stage2_report.png'

        rep_text = stage2_report(heights, arr, d_ph, angles, chan_info, mirror_pairs, C_vec, c_ratio_vec, freq, txt_name)
        stage2_plot(heights, arr, d_ph, angles, chan_info, C_vec, c_ratio_vec, freq, png_name)
        all_reports.append(rep_text)

    # Сохранение общего объединенного отчета
    if len(all_reports) > 1:
        with open('calibration_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_reports) + '\n')
        print("\nОбщий сводный отчет сохранён: calibration_report.txt")


if __name__ == '__main__':
    main()

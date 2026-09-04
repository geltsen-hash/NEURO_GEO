import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import numpy as np
import pandas as pd
import empymod
from tqdm import tqdm


# === ГЕОЛОГИЧЕСКИЕ НАСТРОЙКИ СРЕДЫ ===
RH_UP = 2.0
RH_PL = 50.0
RH_DN = 2.0
ANIS_UP = 2.0       # Rv/Rh для верхнего пласта (условно, см. README)
ANIS_PL = 1.0       # Rv/Rh для основного пласта
ANIS_DN = 2.0       # Rv/Rh для нижнего пласта
THICKNESS = 3.0
ALPHA = 85.0
START_DEPTH = 5.0

# --- НАСТРОЙКИ ШУМА (физические единицы прибора) ---
NOISE_AMP_DB = 0.05      # СКО шума амплитудных каналов, дБ
NOISE_PHASE_DEG = 0.2    # СКО шума фазовых и ZZ-каналов, градусы
# =======================================

FILE_NAME = 'synthetic_well_log_40ch.csv'
MD_START = 0.0
MD_END = 15.0
MD_STEP = 0.1
FREQS = np.array([400_000.0, 2_000_000.0])

Z_TX = {'T1': -0.480, 'T2': 0.840, 'T3': -1.200, 'T4': 1.560, 'T5': 0.180}
Z_RX_ZZ = {'Rzz1': -0.120, 'Rzz2': 0.120}
Z_RX_ZX = {'Rzx1': -1.440, 'Rzx2': 1.800}

# Источник правды о раскладке каналов: амплитуды — линейные, остальные — градусы.
AMP_INDICES = [4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38]
PHASE_INDICES = [i for i in range(40) if i not in AMP_INDICES]


def calculate_point(params):
    Rh_up, Rh_pl, Rv_pl, Rh_dn, D_up, D_dn, alpha = params
    res = [Rh_up, Rh_pl, Rh_dn]
    # aniso в empymod: lambda = sqrt(Rv / Rh)
    aniso = [1.0, np.sqrt(Rv_pl / Rh_pl), 1.0]
    alpha_rad = np.deg2rad(alpha)
    sina = np.sin(alpha_rad)
    cosa = np.cos(alpha_rad)
    depth = [-D_up, D_dn]

    def get_fields(z_t, z_r):
        src = [z_t * sina, 0.0, z_t * cosa]
        rec = [z_r * sina, 0.0, z_r * cosa]
        Hxx = empymod.dipole(src=src, rec=rec, depth=depth, res=res, aniso=aniso,
                             freqtime=FREQS, ab=44, verb=0)
        Hxz = empymod.dipole(src=src, rec=rec, depth=depth, res=res, aniso=aniso,
                             freqtime=FREQS, ab=46, verb=0)
        Hzx = empymod.dipole(src=src, rec=rec, depth=depth, res=res, aniso=aniso,
                             freqtime=FREQS, ab=64, verb=0)
        Hzz = empymod.dipole(src=src, rec=rec, depth=depth, res=res, aniso=aniso,
                             freqtime=FREQS, ab=66, verb=0)
        Vzz = (sina**2)*Hxx + sina*cosa*(Hxz + Hzx) + (cosa**2)*Hzz
        Vzx = -sina*cosa*Hxx + (sina**2)*Hxz - (cosa**2)*Hzx + sina*cosa*Hzz
        return Vzz, Vzx

    f1_sig, f2_sig = [], []
    T_raw_f1, T_raw_f2 = [], []
    for t_name, z_t in [('T1', Z_TX['T1']), ('T2', Z_TX['T2']),
                        ('T3', Z_TX['T3']), ('T4', Z_TX['T4'])]:
        Vzz1, _ = get_fields(z_t, Z_RX_ZZ['Rzz1'])
        Vzz2, _ = get_fields(z_t, Z_RX_ZZ['Rzz2'])
        dist1, dist2 = abs(z_t - Z_RX_ZZ['Rzz1']), abs(z_t - Z_RX_ZZ['Rzz2'])
        if dist2 > dist1:
            p_diff = np.angle(Vzz2 * np.conj(Vzz1), deg=True)
        else:
            p_diff = np.angle(Vzz1 * np.conj(Vzz2), deg=True)
        T_raw_f1.append(float(p_diff[0]))
        T_raw_f2.append(float(p_diff[1]))

    def symm(T):
        return [
            0.75*T[0] + 0.50*T[1] - 0.25*T[2] + 0.00*T[3],
            0.25*T[0] + 0.50*T[1] + 0.25*T[2] + 0.00*T[3],
            0.00*T[0] + 0.25*T[1] + 0.50*T[2] + 0.25*T[3],
            0.00*T[0] - 0.25*T[1] + 0.50*T[2] + 0.75*T[3]
        ]

    f1_sig.extend(symm(T_raw_f1))
    f2_sig.extend(symm(T_raw_f2))

    sym_pairs = [
        (('T1', 'Rzx1'), ('T2', 'Rzx2')),
        (('T5', 'Rzx1'), ('T5', 'Rzx2')),
        (('T2', 'Rzx1'), ('T1', 'Rzx2')),
        (('T4', 'Rzx1'), ('T3', 'Rzx2'))
    ]
    for (t_a, r_a), (t_b, r_b) in sym_pairs:
        Vzz_a, Vzx_a = get_fields(Z_TX[t_a], Z_RX_ZX[r_a])
        Vzz_b, Vzx_b = get_fields(Z_TX[t_b], Z_RX_ZX[r_b])
        S_a = np.conj((Vzz_a - Vzx_a) / (Vzz_a + Vzx_a))
        S_b = np.conj((Vzz_b + Vzx_b) / (Vzz_b - Vzx_b))
        f1_sig.extend([
            float(np.abs(S_a[0])), float(np.angle(S_a[0], deg=True)),
            float(np.abs(S_b[0])), float(np.angle(S_b[0], deg=True))
        ])
        f2_sig.extend([
            float(np.abs(S_a[1])), float(np.angle(S_a[1], deg=True)),
            float(np.abs(S_b[1])), float(np.angle(S_b[1], deg=True))
        ])

    return f1_sig + f2_sig


def build_medium_at_md(md):
    """Возвращает (rh_up, rh_pl, rv_pl, rh_dn, d_up, d_dn) для точки MD."""
    if md <= START_DEPTH:
        rh_up, rh_pl, rv_pl, rh_dn = RH_UP, RH_UP, RH_UP * ANIS_UP, RH_PL
        d_up, d_dn = 4.0, min(START_DEPTH - md, 4.0)
    elif md <= START_DEPTH + THICKNESS:
        rh_up, rh_pl, rv_pl, rh_dn = RH_UP, RH_PL, RH_PL * ANIS_PL, RH_DN
        d_up, d_dn = min(md - START_DEPTH, 4.0), min((START_DEPTH + THICKNESS) - md, 4.0)
    else:
        rh_up, rh_pl, rv_pl, rh_dn = RH_PL, RH_DN, RH_DN * ANIS_DN, RH_DN
        d_up, d_dn = min(md - (START_DEPTH + THICKNESS), 4.0), 4.0
    return rh_up, rh_pl, rv_pl, rh_dn, d_up, d_dn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Генератор синтетического каротажа 40 каналов / 7 параметров.')
    parser.add_argument('--out', default=FILE_NAME, help='Путь к выходному CSV')
    parser.add_argument('--noise-amp-db', type=float, default=NOISE_AMP_DB, help='СКО шума амплитуд, дБ')
    parser.add_argument('--noise-phase-deg', type=float, default=NOISE_PHASE_DEG, help='СКО шума фаз, градусы')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f'Генерация теста: пласт {THICKNESS} м (MD {START_DEPTH} - {START_DEPTH + THICKNESS})...')
    md_array = np.arange(MD_START, MD_END + MD_STEP, MD_STEP)
    results = []
    for md in tqdm(md_array):
        rh_up, rh_pl, rv_pl, rh_dn, d_up, d_dn = build_medium_at_md(md)
        params = [rh_up, rh_pl, rv_pl, rh_dn, d_up, d_dn, ALPHA]
        try:
            sig = calculate_point(params)
        except Exception as e:
            print(f'[WARN] Ошибка на MD={md:.2f}: {e}')
            sig = [np.nan] * 40
        results.append([md] + params + sig)

    header_geo = ['MD', 'True_Rh_up', 'True_Rh_pl', 'True_Rv_pl',
                  'True_Rh_dn', 'True_D_up', 'True_D_down', 'True_Alpha']
    headers_sig = []
    for f_label in ['F400K', 'F2M']:
        headers_sig.extend([f'{f_label}_T1_Simm', f'{f_label}_T2_Simm',
                            f'{f_label}_T3_Simm', f'{f_label}_T4_Simm'])
        for spacing in ['L0.96', 'L1.62', 'L2.28', 'L3.00']:
            headers_sig.extend([
                f'{f_label}_{spacing}_A_Amp', f'{f_label}_{spacing}_A_Phase',
                f'{f_label}_{spacing}_B_Amp', f'{f_label}_{spacing}_B_Phase'
            ])

    df = pd.DataFrame(results, columns=header_geo + headers_sig)

    # --- НАЛОЖЕНИЕ ШУМА ---
    print(f'Наложение физического шума: амплитуды {args.noise_amp_db} дБ, '
          f'фазы/ZZ {args.noise_phase_deg} град...')
    sig_columns = df.columns[8:]
    sig_values = df[sig_columns].values.astype(np.float64, copy=True)

    n_points = sig_values.shape[0]
    amp_noise = 10 ** (np.random.normal(
        0.0, args.noise_amp_db / 20.0, size=(n_points, len(AMP_INDICES))))
    phase_noise = np.random.normal(
        0.0, args.noise_phase_deg, size=(n_points, len(PHASE_INDICES)))

    sig_values[:, AMP_INDICES] *= amp_noise
    sig_values[:, PHASE_INDICES] += phase_noise

    df.loc[:, sig_columns] = sig_values

    out_path = os.path.abspath(args.out)
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f'Готово. Сохранено в {out_path}')

"""Генератор датасета 7 параметров -> 40 сигналов (empymod, 3-слойная анизотропная модель).

Базовые точки берутся из LHS, зеркальные (Rh_up<->Rh_dn, D_up<->D_dn, alpha->180-alpha)
получаются аналитически: ZZ-каналы инвариантны, |S| -> 1/|S|, arg S -> -arg S.
Счёт возобновляемый: готовые чанки лежат в temp_chunks/ и при перезапуске пропускаются.
Результат — dataset_7p_splitZX_40f.parquet.
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
import multiprocessing
import shutil
import time

import empymod
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import qmc
from tqdm import tqdm

# ==========================================
# ПАРАМЕТРЫ ДАТАСЕТА
# ==========================================
N_SAMPLES = 5_000_000
CHUNK_SIZE = 2000
ANISO_MIN, ANISO_MAX = 1.0, 5.0
RO_MIN, RO_MAX = 1.0, 1000.0
D_MIN, D_MAX = 0.05, 4.0
ANGLE_MIN, ANGLE_MAX = 60.0, 120.0
SEED = 12345
ANISO_ADAPTIVE = True   # False -> старое поведение с np.clip(Rh*ratio, .., RO_MAX)
RESUME = True

FREQS = np.array([400_000.0, 2_000_000.0])
Z_TX = {'T1': -0.480, 'T2': 0.840, 'T3': -1.200, 'T4': 1.560, 'T5': 0.180}
Z_RX_ZZ = {'Rzz1': -0.120, 'Rzz2': 0.120}
Z_RX_ZX = {'Rzx1': -1.440, 'Rzx2': 1.800}

TEMP_DIR = 'temp_chunks'
FILE_NAME_PQ = 'dataset_7p_splitZX_40f.parquet'


# ==========================================
# ФИЗИЧЕСКОЕ ЯДРО
# ==========================================
def calculate_point(params):
    Rh_up, Rh_pl, Rv_pl, Rh_dn, D_up, D_dn, alpha = params
    res = [Rh_up, Rh_pl, Rh_dn]
    aniso = [1.0, np.sqrt(Rv_pl / Rh_pl), 1.0]
    alpha_rad = np.deg2rad(alpha)
    sina = np.sin(alpha_rad)
    cosa = np.cos(alpha_rad)
    depth = [-D_up, D_dn]

    def get_fields(z_t, z_r):
        src = [z_t * sina, 0.0, z_t * cosa]
        rec = [z_r * sina, 0.0, z_r * cosa]
        kw = dict(src=src, rec=rec, depth=depth, res=res, aniso=aniso, freqtime=FREQS, verb=0)
        Hxx = empymod.dipole(ab=44, **kw)
        Hxz = empymod.dipole(ab=46, **kw)
        Hzx = empymod.dipole(ab=64, **kw)
        Hzz = empymod.dipole(ab=66, **kw)
        Vzz = (sina**2) * Hxx + sina * cosa * (Hxz + Hzx) + (cosa**2) * Hzz
        Vzx = -sina * cosa * Hxx + (sina**2) * Hxz - (cosa**2) * Hzx + sina * cosa * Hzz
        return Vzz, Vzx

    f1_sig, f2_sig = [], []

    # 1. ZZ компенсированное
    T_raw_f1, T_raw_f2 = [], []
    for t_name in ['T1', 'T2', 'T3', 'T4']:
        z_t = Z_TX[t_name]
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
            0.75 * T[0] + 0.50 * T[1] - 0.25 * T[2] + 0.00 * T[3],
            0.25 * T[0] + 0.50 * T[1] + 0.25 * T[2] + 0.00 * T[3],
            0.00 * T[0] + 0.25 * T[1] + 0.50 * T[2] + 0.25 * T[3],
            0.00 * T[0] - 0.25 * T[1] + 0.50 * T[2] + 0.75 * T[3],
        ]

    f1_sig.extend(symm(T_raw_f1))
    f2_sig.extend(symm(T_raw_f2))

    # 2. ZX с разделением
    sym_pairs = [
        (('T1', 'Rzx1'), ('T2', 'Rzx2')),
        (('T5', 'Rzx1'), ('T5', 'Rzx2')),
        (('T2', 'Rzx1'), ('T1', 'Rzx2')),
        (('T4', 'Rzx1'), ('T3', 'Rzx2')),
    ]
    for (t_a, r_a), (t_b, r_b) in sym_pairs:
        Vzz_a, Vzx_a = get_fields(Z_TX[t_a], Z_RX_ZX[r_a])
        Vzz_b, Vzx_b = get_fields(Z_TX[t_b], Z_RX_ZX[r_b])
        S_a = np.conj((Vzz_a - Vzx_a) / (Vzz_a + Vzx_a))
        S_b = np.conj((Vzz_b + Vzx_b) / (Vzz_b - Vzx_b))
        f1_sig.extend([float(np.abs(S_a[0])), float(np.angle(S_a[0], deg=True)),
                       float(np.abs(S_b[0])), float(np.angle(S_b[0], deg=True))])
        f2_sig.extend([float(np.abs(S_a[1])), float(np.angle(S_a[1], deg=True)),
                       float(np.abs(S_b[1])), float(np.angle(S_b[1], deg=True))])
    return f1_sig + f2_sig


AMP_COLS = np.array([blk + 4 + 4 * k + j for blk in (0, 20) for k in range(4) for j in (0, 2)])
PH_COLS = np.array([blk + 4 + 4 * k + j for blk in (0, 20) for k in range(4) for j in (1, 3)])


def mirror_signals(Y):
    """Отклик зеркальной модели (up<->dn, alpha->180-alpha) — точное тождество,
    проверено до 1e-12: ZZ не меняется, S -> 1/S."""
    Ym = Y.copy()
    Ym[:, AMP_COLS] = 1.0 / Y[:, AMP_COLS]
    Ym[:, PH_COLS] = -Y[:, PH_COLS]
    return Ym


def mirror_params(X):
    Rh_up, Rh_pl, Rv_pl, Rh_dn, D_up, D_dn, alpha = X.T
    return np.column_stack((Rh_dn, Rh_pl, Rv_pl, Rh_up, D_dn, D_up, 180.0 - alpha))


# ==========================================
# WORKER
# ==========================================
def worker_generator(task):
    idx, X_chunk, seed = task
    n = X_chunk.shape[0]
    Y = np.empty((n, 40))
    for i in range(n):
        try:
            Y[i] = calculate_point(X_chunk[i])
        except Exception:
            Y[i] = np.nan
    X_all = np.vstack((X_chunk, mirror_params(X_chunk)))
    Y_all = np.vstack((Y, mirror_signals(Y)))
    data = np.hstack((X_all, Y_all))
    np.random.default_rng(seed).shuffle(data, axis=0)
    return idx, data


def build_headers():
    headers = ['Rh_up', 'Rh_pl', 'Rv_pl', 'Rh_dn', 'D_up', 'D_dn', 'Alpha']
    for f_label in ['F400K', 'F2M']:
        headers.extend([f'{f_label}_T{k}_Simm' for k in range(1, 5)])
        for spacing in ['L0.96', 'L1.62', 'L2.28', 'L3.00']:
            headers.extend([
                f'{f_label}_{spacing}_A_Amp', f'{f_label}_{spacing}_A_Phase',
                f'{f_label}_{spacing}_B_Amp', f'{f_label}_{spacing}_B_Phase',
            ])
    return headers


def sample_parameters(n_base, seed):
    s = qmc.LatinHypercube(d=7, seed=seed).random(n=n_base)
    l_lo, l_hi = np.log10(RO_MIN), np.log10(RO_MAX)
    Rh_up = 10 ** (s[:, 0] * (l_hi - l_lo) + l_lo)
    Rh_pl = 10 ** (s[:, 1] * (l_hi - l_lo) + l_lo)
    Rh_dn = 10 ** (s[:, 3] * (l_hi - l_lo) + l_lo)
    if ANISO_ADAPTIVE:
        ratio_max = np.minimum(ANISO_MAX, RO_MAX / Rh_pl)
        ratio_max = np.maximum(ratio_max, ANISO_MIN)
        ratio = ANISO_MIN + s[:, 2] * (ratio_max - ANISO_MIN)
        Rv_pl = Rh_pl * ratio
    else:
        ratio = ANISO_MIN + s[:, 2] * (ANISO_MAX - ANISO_MIN)
        Rv_pl = np.clip(Rh_pl * ratio, RO_MIN, RO_MAX)
    D_up = s[:, 4] * (D_MAX - D_MIN) + D_MIN
    D_dn = s[:, 5] * (D_MAX - D_MIN) + D_MIN
    alpha = s[:, 6] * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
    return np.column_stack((Rh_up, Rh_pl, Rv_pl, Rh_dn, D_up, D_dn, alpha))


def main():
    start_time = time.time()
    cores = multiprocessing.cpu_count()
    n_base = N_SAMPLES // 2
    chunk_base = CHUNK_SIZE // 2
    headers = build_headers()

    print(f'LHS: {n_base} базовых точек (seed={SEED})...')
    X_global = sample_parameters(n_base, SEED)
    os.makedirs(TEMP_DIR, exist_ok=True)

    tasks = []
    for i, start in enumerate(range(0, n_base, chunk_base)):
        path = os.path.join(TEMP_DIR, f'chunk_{i:06d}.parquet')
        if RESUME and os.path.exists(path):
            continue
        tasks.append((i, X_global[start:start + chunk_base].copy(), SEED + i))

    print(f'Расчёт: {len(tasks)} батчей на {cores} ядрах...')
    if tasks:
        with multiprocessing.Pool(cores) as pool:
            it = pool.imap_unordered(worker_generator, tasks)
            for idx, matrix in tqdm(it, total=len(tasks), desc='Расчет', unit='батч'):
                df = pd.DataFrame(matrix.astype(np.float32), columns=headers)
                df.to_parquet(os.path.join(TEMP_DIR, f'chunk_{idx:06d}.parquet'), engine='pyarrow')

    print('\nПотоковая склейка чанков в Parquet...')
    files = sorted(glob.glob(os.path.join(TEMP_DIR, 'chunk_*.parquet')))
    writer = None
    try:
        for path in tqdm(files, desc='Склейка', unit='файл'):
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(FILE_NAME_PQ, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    shutil.rmtree(TEMP_DIR)
    n_rows = pq.ParquetFile(FILE_NAME_PQ).metadata.num_rows
    print(f'ГОТОВО: {FILE_NAME_PQ}, строк {n_rows}, {(time.time() - start_time) / 60:.1f} мин.')


if __name__ == '__main__':
    main()

"""Общая часть прямого и обратного обучения: раскладка каналов, сплит по базовой
точке, скейлеры и перевод масштабированных величин в физические.

Оба train-скрипта импортируют этот модуль, поэтому он должен лежать рядом с ними.
Единый источник правды для констант, которые дублируются в C++ DLL
(`AMP_INDICES`, индексы log10, имена файлов артефактов).
"""
import json
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

# Каталог, из которого DLL читает всё сразу: FWD_*, INV_* и скейлеры.
ENGINE_DIR = 'engine_weights'

# Индексы амплитудных каналов среди 40 (остальные — фазы ZZ и ZX в градусах).
AMP_INDICES = [4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38]
# Индексы входных параметров, которые логарифмируются (УЭС).
X_LOG_INDICES = [0, 1, 2, 3]

PARAM_NAMES = ['Rh_up', 'Rh_pl', 'Rv_pl', 'Rh_dn', 'D_up', 'D_dn', 'Alpha']


def preprocess_X(X_raw):
    """log10 по четырём УЭС, остальное без изменений."""
    X = X_raw.astype(np.float32, copy=True)
    X[:, X_LOG_INDICES] = np.log10(np.maximum(X[:, X_LOG_INDICES], 1e-5))
    return X


def preprocess_Y(Y_raw):
    """log10 по амплитудам, фазы остаются в градусах."""
    Y = Y_raw.astype(np.float32, copy=True)
    Y[:, AMP_INDICES] = np.log10(np.maximum(Y[:, AMP_INDICES], 1e-8))
    return Y


def group_ids(X_raw):
    """Номер базовой точки: одинаков у образца и его зеркала.

    Зеркало: Rh_up<->Rh_dn, D_up<->D_dn, alpha->180-alpha (Rh_pl, Rv_pl не меняются).
    Упорядочиваем верх/низ по (Rh, D) — получаем ключ, инвариантный к зеркалу.
    Alpha в ключ не входит: 180-alpha считается в плавающей точке и у пары
    может отличаться на единицу младшего разряда, а шести остальных параметров
    для идентификации точки достаточно.
    """
    Rh_up, Rh_pl, Rv_pl, Rh_dn, D_up, D_dn, _alpha = X_raw.T
    swap = (Rh_up > Rh_dn) | ((Rh_up == Rh_dn) & (D_up > D_dn))
    canon = np.column_stack((
        np.where(swap, Rh_dn, Rh_up), np.where(swap, D_dn, D_up),
        Rh_pl, Rv_pl,
        np.where(swap, Rh_up, Rh_dn), np.where(swap, D_up, D_dn),
    )).astype(np.float32)
    view = np.ascontiguousarray(canon).view([('', np.float32)] * canon.shape[1]).ravel()
    _uniq, inverse = np.unique(view, return_inverse=True)
    return inverse


def split_indices(X_raw, rng, val_split=0.1, split_mode='group'):
    """Индексы train/val. 'group' — зеркальная пара не разъезжается между частями,
    'canonical' — обучение только на половине alpha <= 90."""
    n = X_raw.shape[0]
    if split_mode == 'canonical':
        keep = np.flatnonzero(X_raw[:, 6] <= 90.0)
        print(f'   режим canonical: оставлено {keep.size} из {n} строк (зеркала отброшены)')
        perm = rng.permutation(keep.size)
        n_val = int(round(keep.size * val_split))
        return keep[perm[n_val:]], keep[perm[:n_val]]

    inverse = group_ids(X_raw)
    n_groups = int(inverse.max()) + 1
    perm = rng.permutation(n_groups)
    n_val_groups = int(round(n_groups * val_split))
    val_groups = np.zeros(n_groups, dtype=bool)
    val_groups[perm[:n_val_groups]] = True
    is_val = val_groups[inverse]
    print(f'   режим group: {n_groups} базовых точек на {n} строк, '
          f'в валидации {n_val_groups} точек ({is_val.sum()} строк)')
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def fit_scalers(X_data, Y_data, idx_train):
    return (StandardScaler().fit(X_data[idx_train]),
            StandardScaler().fit(Y_data[idx_train]))


def save_scalers(save_dir, scaler_X, scaler_Y):
    np.savetxt(os.path.join(save_dir, 'scaler_X_mean.txt'), scaler_X.mean_)
    np.savetxt(os.path.join(save_dir, 'scaler_X_scale.txt'), scaler_X.scale_)
    np.savetxt(os.path.join(save_dir, 'scaler_Y_mean.txt'), scaler_Y.mean_)
    np.savetxt(os.path.join(save_dir, 'scaler_Y_scale.txt'), scaler_Y.scale_)


def load_scalers(save_dir):
    """Скейлеры прямого обучения. Обратная сеть обязана использовать их же:
    иначе выход обратной сети и де-масштабирование в DLL окажутся в разных шкалах."""
    out = []
    for name, size in (('scaler_X_mean', 7), ('scaler_X_scale', 7),
                       ('scaler_Y_mean', 40), ('scaler_Y_scale', 40)):
        path = os.path.join(save_dir, f'{name}.txt')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} не найден — сначала выполните прямое обучение, '
                f'оно создаёт скейлеры в {save_dir}')
        arr = np.loadtxt(path).astype(np.float32).ravel()
        if arr.size != size:
            raise ValueError(f'{path}: ожидалось {size} значений, прочитано {arr.size}')
        out.append(arr)
    return tuple(out)


def scale(data, mean, std):
    return ((data - mean) / std).astype(np.float32)


def noise_sigma_scaled(scale_Y, amp_db, phase_deg):
    """Сигма шума в масштабированном пространстве, заданная в физических единицах.

    Амплитудный канал хранится как log10|S|, поэтому X дБ -> X/20 в единицах канала.
    Фазовые и ZZ-каналы хранятся в градусах.
    """
    sigma_units = np.full(len(scale_Y), float(phase_deg), dtype=np.float64)
    sigma_units[AMP_INDICES] = float(amp_db) / 20.0
    return (sigma_units / scale_Y).astype(np.float32)


def physical_mae(mae_scaled, scale_Y, headers):
    """MAE каналов в единицах прибора: амплитуды в дБ, фазы и ZZ в градусах."""
    mae_units = mae_scaled * scale_Y
    rows = []
    for i, name in enumerate(headers):
        if i in AMP_INDICES:
            rows.append((name, 20.0 * mae_units[i], 'дБ'))
        else:
            rows.append((name, mae_units[i], 'град'))
    return rows


def write_meta(save_dir, extra):
    """preprocess_meta.json — контракт предобработки для C++-стороны.
    Обновляет уже существующий файл, не затирая записи другого скрипта."""
    path = os.path.join(save_dir, 'preprocess_meta.json')
    meta = {
        'x_log10_indices': X_LOG_INDICES,
        'y_log10_indices': AMP_INDICES,
        'param_names': PARAM_NAMES,
        'activation': 'gelu_erf',
    }
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            meta.update(json.load(fh))
    meta.update(extra)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


def export_layers(model, save_dir, prefix):
    """Веса в формате, который читает DLL: W транспонирована, по файлу на слой."""
    for i in range(1, 6):
        layer = getattr(model, f'fc{i}')
        np.savetxt(os.path.join(save_dir, f'{prefix}W{i}.txt'),
                   layer.weight.detach().cpu().numpy().T)
        np.savetxt(os.path.join(save_dir, f'{prefix}b{i}.txt'),
                   layer.bias.detach().cpu().numpy())

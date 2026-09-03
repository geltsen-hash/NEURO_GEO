"""Обучение прямого суррогата 7 параметров -> 40 сигналов.

Читает dataset_7p_splitZX_40f.parquet, делит train/val по базовой точке (зеркальная пара
не разъезжается между частями), учит MLP 7-512-1024-512-256-40 с GELU и ReduceLROnPlateau.
В SAVE_DIR кладёт веса лучшей эпохи (.txt для C++-инференса и .pt), параметры скейлеров,
preprocess_meta.json и val_mae_physical.txt — ошибку по каналам в дБ и градусах.
"""
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

FILE_NAME = 'dataset_7p_splitZX_40f.parquet'
SAVE_DIR = 'forward_7p_40_sig_weights'
BATCH_SIZE = 8192
EPOCHS = 250
VAL_SPLIT = 0.1
PATIENCE = 30          # early stopping
LR_PATIENCE = 8        # ReduceLROnPlateau
LR_FACTOR = 0.3
MIN_LR = 1e-6
SEED = 42

# 'group'     - обе строки зеркальной пары попадают в одну часть (формат инференса не меняется);
# 'canonical' - обучение только на половине alpha <= 90 (вдвое быстрее эпоха,
#               но тогда C++-сторона обязана сама применять симметрию при alpha > 90:
#               up<->dn, D_up<->D_dn, alpha->180-alpha, Amp->1/Amp, Phase->-Phase).
SPLIT_MODE = 'group'

AMP_INDICES = [4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38]


class EmSurrogateNet7D_40CH(nn.Module):
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


def group_ids(X):
    """Номер базовой точки: одинаков у образца и его зеркала.

    Зеркало: Rh_up<->Rh_dn, D_up<->D_dn, alpha->180-alpha (Rh_pl, Rv_pl не меняются).
    Упорядочиваем верх/низ по (Rh, D) — получаем ключ, инвариантный к зеркалу.
    Alpha в ключ не входит: 180-alpha считается в плавающей точке и у пары
    может отличаться на единицу младшего разряда, а шести остальных параметров
    для идентификации точки достаточно.
    """
    Rh_up, Rh_pl, Rv_pl, Rh_dn, D_up, D_dn, _alpha = X.T
    swap = (Rh_up > Rh_dn) | ((Rh_up == Rh_dn) & (D_up > D_dn))
    canon = np.column_stack((
        np.where(swap, Rh_dn, Rh_up), np.where(swap, D_dn, D_up),
        Rh_pl, Rv_pl,
        np.where(swap, Rh_up, Rh_dn), np.where(swap, D_up, D_dn),
    )).astype(np.float32)
    view = np.ascontiguousarray(canon).view([('', np.float32)] * canon.shape[1]).ravel()
    _uniq, inverse = np.unique(view, return_inverse=True)
    return inverse


def split_indices(X_raw, rng):
    n = X_raw.shape[0]
    if SPLIT_MODE == 'canonical':
        keep = np.flatnonzero(X_raw[:, 6] <= 90.0)
        print(f'   режим canonical: оставлено {keep.size} из {n} строк (зеркала отброшены)')
        perm = rng.permutation(keep.size)
        n_val = int(round(keep.size * VAL_SPLIT))
        return keep[perm[n_val:]], keep[perm[:n_val]]

    inverse = group_ids(X_raw)
    n_groups = int(inverse.max()) + 1
    perm = rng.permutation(n_groups)
    n_val_groups = int(round(n_groups * VAL_SPLIT))
    val_groups = np.zeros(n_groups, dtype=bool)
    val_groups[perm[:n_val_groups]] = True
    is_val = val_groups[inverse]
    print(f'   режим group: {n_groups} базовых точек на {n} строк, '
          f'в валидации {n_val_groups} точек ({is_val.sum()} строк)')
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def eval_val(model, X_v, Y_v, criterion, batch_size):
    """Возвращает MSE в масштабированном пространстве и сумму |ошибок| по каналам."""
    model.eval()
    total_loss = 0.0
    abs_err = torch.zeros(Y_v.shape[1], device=Y_v.device, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(X_v), batch_size):
            b_x, b_y = X_v[i:i + batch_size], Y_v[i:i + batch_size]
            out = model(b_x)
            total_loss += criterion(out, b_y).item() * len(b_x)
            abs_err += (out - b_y).abs().sum(dim=0).double()
    return total_loss / len(X_v), (abs_err / len(X_v)).cpu().numpy()


def physical_mae(mae_scaled, scaler_Y, headers):
    """MAE каналов в единицах прибора: амплитуды в дБ, фазы и ZZ в градусах."""
    mae_units = mae_scaled * scaler_Y.scale_          # обратно в единицы датасета
    rows = []
    for i, name in enumerate(headers):
        if i in AMP_INDICES:
            rows.append((name, 20.0 * mae_units[i], 'дБ'))   # канал хранится как log10|S|
        else:
            rows.append((name, mae_units[i], 'град'))
    return rows


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print('1. Чтение Parquet датасета...')
    df = pd.read_parquet(FILE_NAME)
    headers = list(df.columns[7:])
    X_raw = df.iloc[:, :7].values.astype(np.float32)
    Y_raw = df.iloc[:, 7:].values.astype(np.float32)
    del df

    print('2. Отсев нефинитных строк...')
    finite = np.isfinite(X_raw).all(axis=1) & np.isfinite(Y_raw).all(axis=1)
    if not finite.all():
        print(f'   отброшено {int((~finite).sum())} строк')
        X_raw, Y_raw = X_raw[finite], Y_raw[finite]

    print('3. Разделение по базовой точке...')
    idx_train, idx_val = split_indices(X_raw, rng)

    print('4. Логарифмирование УЭС и амплитуд...')
    X_data = X_raw.copy()
    X_data[:, 0:4] = np.log10(np.maximum(X_data[:, 0:4], 1e-5))
    Y_data = Y_raw.copy()
    Y_data[:, AMP_INDICES] = np.log10(np.maximum(Y_data[:, AMP_INDICES], 1e-8))

    print('5. Масштабирование (StandardScaler, fit только на train)...')
    scaler_X = StandardScaler().fit(X_data[idx_train])
    scaler_Y = StandardScaler().fit(Y_data[idx_train])
    X_scaled = scaler_X.transform(X_data).astype(np.float32)
    Y_scaled = scaler_Y.transform(Y_data).astype(np.float32)

    np.savetxt(os.path.join(SAVE_DIR, 'scaler_X_mean.txt'), scaler_X.mean_)
    np.savetxt(os.path.join(SAVE_DIR, 'scaler_X_scale.txt'), scaler_X.scale_)
    np.savetxt(os.path.join(SAVE_DIR, 'scaler_Y_mean.txt'), scaler_Y.mean_)
    np.savetxt(os.path.join(SAVE_DIR, 'scaler_Y_scale.txt'), scaler_Y.scale_)
    with open(os.path.join(SAVE_DIR, 'preprocess_meta.json'), 'w', encoding='utf-8') as fh:
        json.dump({
            'x_log10_indices': [0, 1, 2, 3],
            'y_log10_indices': AMP_INDICES,
            'y_headers': headers,
            'activation': 'gelu_erf',
            'split_mode': SPLIT_MODE,
            'seed': SEED,
        }, fh, ensure_ascii=False, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n6. Трансфер в {device}...')
    X_train_gpu = torch.from_numpy(X_scaled[idx_train]).to(device)
    Y_train_gpu = torch.from_numpy(Y_scaled[idx_train]).to(device)
    X_val_gpu = torch.from_numpy(X_scaled[idx_val]).to(device)
    Y_val_gpu = torch.from_numpy(Y_scaled[idx_val]).to(device)
    train_size = X_train_gpu.shape[0]

    model = EmSurrogateNet7D_40CH().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)

    best_val_loss = float('inf')
    best_state = None
    best_mae_scaled = None
    epochs_no_improve = 0

    print('\nЗАПУСК ОБУЧЕНИЯ (ReduceLROnPlateau + Early Stopping)')
    print(f'{"Эпоха":<8} | {"Train MSE":<12} | {"Val MSE":<12} | {"LR":<10}')
    print('-' * 52)
    train_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss, n_seen = 0.0, 0
        permutation = torch.randperm(train_size, device=device)
        for i in range(0, train_size, BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_X, batch_Y = X_train_gpu[indices], Y_train_gpu[indices]
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(indices)
            n_seen += len(indices)
        avg_train_loss = running_loss / n_seen

        val_loss, mae_scaled = eval_val(model, X_val_gpu, Y_val_gpu, criterion, BATCH_SIZE)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae_scaled = mae_scaled
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'{epoch+1:<8d} | {avg_train_loss:<12.6f} | {val_loss:<12.6f} | {lr_now:<10.2e}')

        if epochs_no_improve >= PATIENCE:
            print(f'\n!!! Ранняя остановка на {epoch+1} эпохе !!!')
            break

    print(f'\nАбсолютный минимум Val MSE: {best_val_loss:.6f}')
    print(f'Обучение заняло {(time.time() - train_start)/60:.1f} минут.')

    print('\n7. Точность в физических единицах (val, лучшая эпоха):')
    rows = physical_mae(best_mae_scaled, scaler_Y, headers)
    with open(os.path.join(SAVE_DIR, 'val_mae_physical.txt'), 'w', encoding='utf-8') as fh:
        fh.write('channel\tMAE\tunit\n')
        for name, val, unit in rows:
            fh.write(f'{name}\t{val:.6g}\t{unit}\n')
    amps = [r for r in rows if r[2] == 'дБ']
    phases = [r for r in rows if r[2] == 'град']
    print(f'   амплитуды: средн. {np.mean([r[1] for r in amps]):.4f} дБ, '
          f'худший {max(amps, key=lambda r: r[1])[0]} {max(r[1] for r in amps):.4f} дБ')
    print(f'   фазы/ZZ:   средн. {np.mean([r[1] for r in phases]):.4f} град, '
          f'худший {max(phases, key=lambda r: r[1])[0]} {max(r[1] for r in phases):.4f} град')
    print(f'   полный список: {os.path.join(SAVE_DIR, "val_mae_physical.txt")}')

    print('\n8. Экспорт ЛУЧШИХ весов...')
    model.load_state_dict(best_state)
    for i in range(1, 6):
        layer = getattr(model, f'fc{i}')
        np.savetxt(os.path.join(SAVE_DIR, f'W{i}.txt'), layer.weight.detach().cpu().numpy().T)
        np.savetxt(os.path.join(SAVE_DIR, f'b{i}.txt'), layer.bias.detach().cpu().numpy())
    torch.save(best_state, os.path.join(SAVE_DIR, 'best_forward_model.pt'))
    print('Готово!')


if __name__ == '__main__':
    main()

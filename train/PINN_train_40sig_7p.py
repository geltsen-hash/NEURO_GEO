"""Обучение обратной сети 40 сигналов -> 7 параметров (PINN: geo-loss + физический loss).

Отличия от исходной версии:
* работает с общим каталогом движка (pipeline_common.ENGINE_DIR): читает оттуда FWD_*.txt
  и скейлеры прямого обучения, туда же кладёт INV_*.txt — это ровно то, что читает InitEngine;
* скейлеры НЕ фитятся заново: обратная сеть обязана говорить в той же шкале, в которой
  работают замороженный суррогат и де-масштабирование в DLL;
* сплит по базовой точке (зеркальная пара не разъезжается между train и val);
* валидация детерминированная: шум фиксирован и не меняется от эпохи к эпохе;
* шум задаётся в физических единицах (дБ по амплитудам, градусы по фазам), а не в сигмах;
* веса экспортируются из состояния лучшей эпохи.

Требует pipeline_common.py рядом со скриптом и выполненного прямого обучения.
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import pipeline_common as pc

FILE_NAME = 'dataset_7p_splitZX_40f.parquet'
ENGINE_DIR = pc.ENGINE_DIR
BEST_MODEL_NAME = os.path.join(ENGINE_DIR, 'PINN_40sig_7p.pt')
BATCH_SIZE = 16384
EPOCHS = 500
VAL_SPLIT = 0.1
PATIENCE = 50
LR_PATIENCE = 10
LR_FACTOR = 0.5
SEED = 42
PHYS_WEIGHT = 2.0

# Сплит должен совпадать с прямым обучением, иначе валидация обратной сети содержит
# точки, на которых учился суррогат.
SPLIT_MODE = 'group'

# --- НАСТРОЙКИ ШУМА (в физических единицах прибора) ---
NOISE_AMP_DB = 0.05        # СКО шума амплитудных каналов, дБ
NOISE_PHASE_DEG = 0.2      # СКО шума фазовых и ZZ-каналов, градусы

# --- ЗОНА ИССЛЕДОВАНИЯ ---
# Границу глубже DOI_THRESHOLD сигнал не различает. BLIND_SENTINEL — значение, которым
# такая глубина заменяется в обучающей выборке. Держим его внутри диапазона поиска DLL
# (max_bounds по D обычно 4.0 м): значение вне границ всё равно будет обрезано клипом,
# и «границы не видно» станет неотличимо от «граница ровно на границе диапазона».
# Договорённость с вызывающим приложением: D >= DOI_THRESHOLD означает «граница за
# пределами зоны исследования».
DOI_THRESHOLD = 3.8
BLIND_SENTINEL = 4.0
APPLY_BLINDING = True


class ForwardSurrogate(nn.Module):
    """Замороженный суррогат 7 -> 40, веса берутся из прямого обучения."""

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
        """Строгая загрузка: любое несоответствие — исключение, а не предупреждение.
        Молча обученная на битом суррогате обратная сеть выглядит рабочей и даёт мусор."""
        for i in range(1, 6):
            layer = getattr(self, f'fc{i}')
            w_file = os.path.join(weights_dir, f'FWD_W{i}.txt')
            b_file = os.path.join(weights_dir, f'FWD_b{i}.txt')
            for path in (w_file, b_file):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f'{path} не найден — сначала выполните прямое обучение '
                        f'(оно кладёт FWD_*.txt в {weights_dir})')
            W = np.loadtxt(w_file, ndmin=2)
            b = np.loadtxt(b_file, ndmin=1)
            expected_w = (layer.in_features, layer.out_features)   # файл хранит W.T
            if W.shape != expected_w:
                raise ValueError(f'{w_file}: ожидалась матрица {expected_w}, прочитана {W.shape}')
            if b.shape != (layer.out_features,):
                raise ValueError(f'{b_file}: ожидался вектор {(layer.out_features,)}, '
                                 f'прочитан {b.shape}')
            layer.weight.data = torch.FloatTensor(W.T)
            layer.bias.data = torch.FloatTensor(b)


class InverseNet40CH(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 7)
        self.gelu = nn.GELU()

    def forward(self, y_logs):
        x = self.gelu(self.fc1(y_logs))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.gelu(self.fc4(x))
        return self.fc5(x)


def apply_blinding(X_raw):
    """Глубины за зоной исследования заменяются сентинелом, а УЭС за такой границей —
    на УЭС пласта: за пределами DOI сигнал об этих величинах ничего не знает."""
    X = X_raw.copy()
    blind_up = X[:, 4] >= DOI_THRESHOLD
    blind_dn = X[:, 5] >= DOI_THRESHOLD
    X[blind_up, 4] = BLIND_SENTINEL
    X[blind_dn, 5] = BLIND_SENTINEL
    X[blind_up, 0] = X[blind_up, 1]
    X[blind_dn, 3] = X[blind_dn, 1]
    print(f'   слепая зона: кровля {blind_up.mean()*100:.1f}% строк, '
          f'подошва {blind_dn.mean()*100:.1f}% (сентинел D={BLIND_SENTINEL} м)')
    return X


def eval_val(inv_model, fwd_model, X_v, Y_v, noise_v, criterion, batch_size):
    """Детерминированная валидация: шум задан заранее и одинаков на всех эпохах."""
    inv_model.eval()
    total_loss = 0.0
    abs_err = torch.zeros(X_v.shape[1], device=X_v.device, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(X_v), batch_size):
            b_x = X_v[i:i + batch_size]
            b_y_clean = Y_v[i:i + batch_size]
            b_y_noisy = b_y_clean + noise_v[i:i + batch_size]
            pred_x = inv_model(b_y_noisy)
            recon_y = fwd_model(pred_x)
            loss = criterion(pred_x, b_x) + PHYS_WEIGHT * criterion(recon_y, b_y_clean)
            total_loss += loss.item() * len(b_x)
            abs_err += (pred_x - b_x).abs().sum(dim=0).double()
    return total_loss / len(X_v), (abs_err / len(X_v)).cpu().numpy()


def report_param_mae(mae_scaled, scale_X):
    """MAE параметров в физических единицах: УЭС в декадах (лог-шкала), D в метрах,
    Alpha в градусах."""
    mae_units = mae_scaled * scale_X
    units = ['декад log10', 'декад log10', 'декад log10', 'декад log10', 'м', 'м', 'град']
    return [(pc.PARAM_NAMES[i], mae_units[i], units[i]) for i in range(7)]


def main():
    os.makedirs(ENGINE_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print('1. Чтение Parquet датасета...')
    df = pd.read_parquet(FILE_NAME)
    X_raw = df.iloc[:, :7].values.astype(np.float32)
    Y_raw = df.iloc[:, 7:].values.astype(np.float32)
    del df

    finite = np.isfinite(X_raw).all(axis=1) & np.isfinite(Y_raw).all(axis=1)
    if not finite.all():
        print(f'   отброшено {int((~finite).sum())} нефинитных строк')
        X_raw, Y_raw = X_raw[finite], Y_raw[finite]

    print('2. Разделение по базовой точке (тот же режим, что в прямом обучении)...')
    idx_train, idx_val = pc.split_indices(X_raw, rng, VAL_SPLIT, SPLIT_MODE)

    print('3. Слепая зона и логарифмирование...')
    X_blind = apply_blinding(X_raw) if APPLY_BLINDING else X_raw
    X_data = pc.preprocess_X(X_blind)
    Y_data = pc.preprocess_Y(Y_raw)

    print('4. Скейлеры прямого обучения (без повторного fit)...')
    mean_X, scale_X, mean_Y, scale_Y = pc.load_scalers(ENGINE_DIR)
    X_scaled = pc.scale(X_data, mean_X, scale_X)
    Y_scaled = pc.scale(Y_data, mean_Y, scale_Y)

    sigma = pc.noise_sigma_scaled(scale_Y, NOISE_AMP_DB, NOISE_PHASE_DEG)
    print(f'   шум: {NOISE_AMP_DB} дБ по амплитудам, {NOISE_PHASE_DEG} град по фазам '
          f'(в сигмах каналов {sigma.min():.4f}..{sigma.max():.4f})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'5. Устройство: {device}')
    X_train_gpu = torch.from_numpy(X_scaled[idx_train]).to(device)
    Y_train_gpu = torch.from_numpy(Y_scaled[idx_train]).to(device)
    X_val_gpu = torch.from_numpy(X_scaled[idx_val]).to(device)
    Y_val_gpu = torch.from_numpy(Y_scaled[idx_val]).to(device)
    sigma_gpu = torch.from_numpy(sigma).to(device)
    train_size = X_train_gpu.shape[0]

    # Фиксированная реализация валидационного шума: иначе val_loss шумит сам по себе,
    # а от него зависят выбор лучшей эпохи, шедулер и early stopping.
    val_gen = torch.Generator(device=device).manual_seed(SEED)
    noise_val = torch.randn(Y_val_gpu.shape, generator=val_gen, device=device) * sigma_gpu

    forward_model = ForwardSurrogate()
    forward_model.load_txt_weights(ENGINE_DIR)
    forward_model = forward_model.to(device).eval()
    for param in forward_model.parameters():
        param.requires_grad = False

    inverse_model = InverseNet40CH().to(device)
    optimizer = optim.Adam(inverse_model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    best_mae_scaled = None
    epochs_no_improve = 0
    train_start = time.time()

    print('\nЗАПУСК PINN-ОБУЧЕНИЯ ОБРАТНОЙ СЕТИ')
    print(f'{"Эпоха":<8} | {"Total":<12} | {"Geo":<12} | {"Phys":<12} | {"Val":<12} | {"LR":<10}')
    print('-' * 78)

    for epoch in range(EPOCHS):
        inverse_model.train()
        run_total, run_geo, run_phys, n_batches = 0.0, 0.0, 0.0, 0
        permutation = torch.randperm(train_size, device=device)

        for i in range(0, train_size, BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_X = X_train_gpu[indices]
            batch_Y_clean = Y_train_gpu[indices]

            optimizer.zero_grad()
            batch_Y_noisy = batch_Y_clean + torch.randn_like(batch_Y_clean) * sigma_gpu
            batch_X_pred = inverse_model(batch_Y_noisy)
            batch_Y_recon = forward_model(batch_X_pred)

            loss_geo = criterion(batch_X_pred, batch_X)
            loss_phys = criterion(batch_Y_recon, batch_Y_clean)
            loss_total = loss_geo + PHYS_WEIGHT * loss_phys
            loss_total.backward()
            optimizer.step()

            run_total += loss_total.item()
            run_geo += loss_geo.item()
            run_phys += loss_phys.item()
            n_batches += 1

        val_loss, mae_scaled = eval_val(
            inverse_model, forward_model, X_val_gpu, Y_val_gpu, noise_val, criterion, BATCH_SIZE)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae_scaled = mae_scaled
            best_state = copy.deepcopy(inverse_model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'{epoch+1:<8d} | {run_total/n_batches:<12.5f} | {run_geo/n_batches:<12.5f} | '
                  f'{run_phys/n_batches:<12.5f} | {val_loss:<12.5f} | {lr_now:<10.6f}')

        if epochs_no_improve >= PATIENCE:
            print(f'\n!!! Ранняя остановка на {epoch+1} эпохе !!!')
            break

    print(f'\nЛучший Val Loss: {best_val_loss:.6f}')
    print(f'Обучение заняло {(time.time() - train_start)/60:.1f} мин.')

    print('\n6. Точность по параметрам (val, лучшая эпоха):')
    rows = report_param_mae(best_mae_scaled, scale_X)
    with open(os.path.join(ENGINE_DIR, 'val_mae_params.txt'), 'w', encoding='utf-8') as fh:
        fh.write('param\tMAE\tunit\n')
        for name, val, unit in rows:
            fh.write(f'{name}\t{val:.6g}\t{unit}\n')
            print(f'   {name:<8} {val:.4f} {unit}')

    print('\n7. Экспорт ЛУЧШИХ весов обратной сети...')
    inverse_model.load_state_dict(best_state)
    pc.export_layers(inverse_model, ENGINE_DIR, 'INV_')
    torch.save(best_state, BEST_MODEL_NAME)
    pc.write_meta(ENGINE_DIR, {
        'inverse_split_mode': SPLIT_MODE,
        'inverse_seed': SEED,
        'doi_threshold_m': DOI_THRESHOLD,
        'blind_sentinel_m': BLIND_SENTINEL if APPLY_BLINDING else None,
        'noise_amp_db': NOISE_AMP_DB,
        'noise_phase_deg': NOISE_PHASE_DEG,
        'phys_weight': PHYS_WEIGHT,
    })
    print(f'Готово! В {ENGINE_DIR} лежат FWD_*, INV_*, скейлеры и preprocess_meta.json — '
          f'этот каталог передаётся в InitEngine.')


if __name__ == '__main__':
    main()

# Список правок для `syntetic_log_40sig_gen.py` и `test_PINN_40sig_7p.py`

## `syntetic_log_40sig_gen.py`

### 1. Исправить модель шума
**Что сейчас:**
```python
noise = np.random.normal(0, NOISE_LEVEL, sig_values.shape)
df[sig_columns] = sig_values * (1.0 + noise)
```
Шум умножается на *все* 40 каналов, включая фазы/ZZ-разности в градусах. Для амплитуд это допустимо, для фаз — физически неверно (умножение градусов даёт масштабирование, а не аддитивный фазовый шум).

**Что сделать:**
- амплитудные каналы (`AMP_INDICES = [4,6,8,10,12,14,16,18,24,26,28,30,32,34,36,38]`) — мультипликативный шум в долях или в дБ;
- фазовые и ZZ-каналы — аддитивный шум в градусах.

Пример:
```python
amp_indices = [4,6,8,10,12,14,16,18,24,26,28,30,32,34,36,38]
noise_amp = np.random.normal(0, NOISE_LEVEL_AMP, sig_values.shape)
noise_phase_deg = np.random.normal(0, NOISE_LEVEL_PHASE_DEG, sig_values.shape)
sig_values[:, amp_indices] *= (1.0 + noise_amp[:, amp_indices])
sig_values[:, [i for i in range(40) if i not in amp_indices]] += noise_phase_deg[:, [i for i in range(40) if i not in amp_indices]]
```

### 2. Исправить комментарий к `NOISE_LEVEL`
**Что сейчас:** `#NOISE_LEVEL = 0.005 # 2% гауссовского шума от величины сигнала`  
**Что сделать:** `0.005` = 0.5 %, не 2 %. Либо поменять значение на `0.02`, либо комментарий на «0.5%».

### 3. (Опционально) Добавить обработку ошибок `empymod`
Если какая-то точка не сойдётся (редко, но возможно при экстремальных параметрах), скрипт упадёт. Достаточно `try/except` в `calculate_point` или обёртка в цикле `for md`.

### 4. (Опционально) Добавить seed
Сейчас `np.random.seed(42)` только для шума. Для воспроизводимости параметров среды тоже стоит зафиксировать seed в начале.

---

## `test_PINN_40sig_7p.py`

### 1. Обновить путь к прямым весам и скейлерам
**Что сейчас:**
```python
FWD_DIR = 'forward_7p_40_sig_weights'
assign_layer(self.fc1, os.path.join(FWD_DIR, 'W1.txt'), os.path.join(FWD_DIR, 'b1.txt'))
```
**Что сделать:**
```python
FWD_DIR = 'engine_weights'
assign_layer(self.fc1, os.path.join(FWD_DIR, 'FWD_W1.txt'), os.path.join(FWD_DIR, 'FWD_b1.txt'))
```
и так для всех 5 слоёв.

### 2. Обновить путь к файлу обратной сети
**Что сейчас:**
```python
INV_FILE = 'PINN_40sig_7p.pt'
```
**Что сделать:**
```python
INV_FILE = os.path.join('engine_weights', 'PINN_40sig_7p.pt')
```

### 3. Загружать метаданные из `preprocess_meta.json`
**Что сейчас:** `amp_indices` и имена каналов захардкожены.  
**Что сделать:** читать `AMP_INDICES`, `X_LOG_INDICES`, `y_headers` из `engine_weights/preprocess_meta.json`, чтобы при изменении pipeline не рассинхронизироваться.

### 4. Обновить `min_bounds` / `max_bounds` под формат генератора
**Что сейчас:**
```python
min_bounds[i] = [0.5,   0.5,   0.5,   0.5, 0.05, 0.05, 60.0]
max_bounds[i] = [200.0, 200.0, 200.0, 200.0, 4.0,  4.0, 120.0]
```
Генератор даёт `Rh` в диапазоне `1..1000`. Если модель обучена на `1..1000`, `min=0.5` допустим, но не соответствует обучающей выборке. Рекомендуется:
```python
min_bounds[i] = [1.0, 1.0, 1.0, 1.0, 0.05, 0.05, 60.0]
max_bounds[i] = [1000.0, 1000.0, 1000.0, 1000.0, 4.0, 4.0, 120.0]
```

### 5. Обновить логику шумовой маски
Сейчас `active_mask = (signal_amplitude > NOISE_THRESHOLD).float()` использует стандартное отклонение в масштабированном/логарифмированном пространстве. После перехода на физический шум (дБ/градусы) этот порог потерял смысл. Рекомендуется либо убрать маску, либо заменить на известный физический Signal-to-Noise criterion.

### 6. (Опционально) Синхронизировать PGD с C++ DLL
Сейчас `test_PINN_40sig_7p.py` делает собственную пакетную Adam-оптимизацию, которая логически похожа, но не идентична C++ `RunPGDInversion` (разные细节 обработки градиентов, topology-режимы, ручной клиппинг). Для end-to-end интеграции лучше использовать ту же математику, что и в `dll/NEURO_40_7.cpp`.

### 7. (Опционально) Отвязать от `matplotlib`
Если скрипт запускается в headless-окружении, `plt.show()` может падать. `plt.savefig` работает, но стоит добавить `plt.switch_backend('Agg')` или оборачивать `plt.show()` в `try`.

### 8. (Опционально) Добавить argparse для `CSV`/`engine_weights`
Сейчас пути захардкожены. Для гибкости:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='synthetic_well_log_40ch.csv')
parser.add_argument('--engine', default='engine_weights')
args = parser.parse_args()
```

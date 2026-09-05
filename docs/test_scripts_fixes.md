# Список правок для `syntetic_log_40sig_7p.py` и `test_PINN_40sig_7p.py`

## `syntetic_log_40sig_gen.py`

### 1. Модель шума (согласно статье)
**Состояние:**
```python
NOISE_STD = 0.005  # 0.5% относительного шума
noise = np.random.normal(0.0, NOISE_STD, size=sig_values.shape)
sig_values *= (1.0 + noise)
```
Шум умножается на *все* 40 каналов. Это соответствует методике в статье (`Y_noisy = Y_clean * (1 + ε)`, `ε ~ N(0, 0.005^2)`) и отличается от модели шума в обучении (дБ для амплитуд, градусы для фаз/ZZ). Статья явно отмечает это расхождение как особенность тестового каротажа.

### 2. Комментарий к `NOISE_STD`
`0.005` = 0.5 %, не 2 %.

### 3. (Опционально) Добавить обработку ошибок `empymod`
Если какая-то точка не сойдётся (редко, но возможно при экстремальных параметрах), скрипт упадёт. Достаточно `try/except` в `calculate_point` или обёртка в цикле `for md`.

### 4. (Опционально) Добавить seed
Сейчас `np.random.seed(args.seed)` задаётся в `__main__`. Для воспроизводимости параметров среды seed лучше фиксировать в начале.

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

import numpy as np
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time
# ==========================================
# ПАРАМЕТРЫ ЗАДАЧИ
# ==========================================
FILENAME = r'D:\EXP\GEO_PALLETE\AMK_geo_nodes_50_50_50_48_48.geo'
HEADER_SIZE_BYTES = 4096
SAMPLE_SIZE = 5000000          
BATCH_SIZE = 8192              # Возвращаем проверенный батч 8192
EPOCHS = 150                   
class EmSurrogateNet(nn.Module):
    def __init__(self):
        super(EmSurrogateNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
# ==========================================
# ЧТЕНИЕ ФАЙЛА И СЭМПЛИРОВАНИЕ (БЕЗ ОШИБОК ПАМЯТИ)
# ==========================================
print("Чтение заголовка...")
with open(FILENAME, 'rb') as f:
    header_data = f.read(HEADER_SIZE_BYTES)
points_offset = 3172
points = struct.unpack('<5I', header_data[points_offset : points_offset + 20])
def extract_array(byte_data, start_offset, count):
    return np.array(struct.unpack(f'<{count}f', byte_data[start_offset : start_offset + count * 4]), dtype=np.float32)
offset = 1172
Ro_sonde_tab = extract_array(header_data, offset, points[0]); offset += 400
Ro_up_tab    = extract_array(header_data, offset, points[1]); offset += 400
Ro_down_tab  = extract_array(header_data, offset, points[2]); offset += 400
D_up_tab     = extract_array(header_data, offset, points[3]); offset += 400
D_down_tab   = extract_array(header_data, offset, points[4])
grid_shape = (points[0], points[1], points[2], points[3], points[4])
dt_out = np.dtype([
    ('signals', np.complex64, (10,)), 
    ('pallete_N', np.uint32),
    ('mirror_N', np.uint32)
])
print("Подключение к бинарным данным...")
mmap_data = np.memmap(FILENAME, dtype=dt_out, mode='r', offset=HEADER_SIZE_BYTES)
actual_nodes = len(mmap_data)
sample_size = min(SAMPLE_SIZE, actual_nodes)
print(f"Генерация {sample_size} случайных индексов...")
sample_indices = np.sort(np.random.choice(actual_nodes, size=sample_size, replace=False))
print("Извлечение данных (с безопасным нунлевым массивом)...")
# КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: np.zeros гарантирует чистую память без мусора
sampled_data = np.zeros(sample_size, dtype=dt_out) 
CHUNK_SIZE = 20_000_000  
filled = 0
start_time = time.time()
for start in range(0, actual_nodes, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, actual_nodes)
    mask = (sample_indices >= start) & (sample_indices < end)
    chunk_indices = sample_indices[mask]
    
    if len(chunk_indices) > 0:
        chunk_data = mmap_data[start:end]
        local_indices = chunk_indices - start
        sampled_data[filled : filled + len(chunk_indices)] = chunk_data[local_indices]
        filled += len(chunk_indices)
        
    print(f"  Сканирование диска: {end/1000000:.0f} млн точек...")
print("Перемешивание данных...")
np.random.shuffle(sampled_data)
# ==========================================
# РАСКОДИРОВАНИЕ
# ==========================================
print("Раскодирование N -> параметры...")
pallete_ids = sampled_data['pallete_N']
i_sonde, i_up, i_down, i_dup, i_ddown = np.unravel_index(pallete_ids, grid_shape)
X_data = np.column_stack((
    Ro_sonde_tab[i_sonde],
    Ro_up_tab[i_up],
    Ro_down_tab[i_down],
    D_up_tab[i_dup],
    D_down_tab[i_ddown]
)).astype(np.float32)
print("Подготовка X (Логарифмирование)...")
# Дополнительная защита от абсолютных нулей
X_data[:, 0:3] = np.maximum(X_data[:, 0:3], 1e-5) 
X_data[:, 0:3] = np.log10(X_data[:, 0:3])
print("Подготовка Y (Амплитуда/Фаза)...")
SGN = sampled_data['signals'][:, 0:8]
Y_data = np.hstack((np.abs(SGN), np.angle(SGN, deg=True))).astype(np.float32)
# ==========================================
# ОЧИСТКА ДАННЫХ ОТ СЛОМАННЫХ ТОЧЕК (NaN и Inf)
# ==========================================
print("Проверка на битые расчетные точки из C++...")
# Ищем строки, где все значения являются конечными нормальными числами
valid_mask = np.isfinite(X_data).all(axis=1) & np.isfinite(Y_data).all(axis=1)
bad_count = len(X_data) - np.sum(valid_mask)
if bad_count > 0:
    print(f"ВНИМАНИЕ: Найдено {bad_count} сломанных точек (NaN или Inf). Удаляем их...")
    X_data = X_data[valid_mask]
    Y_data = Y_data[valid_mask]
    print(f"Осталось чистых данных для обучения: {len(X_data)} узлов.")
else:
    print("Все данные корректны, битых точек нет.")

print("Масштабирование StandardScaler...")
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_data)
Y_scaled = scaler_Y.fit_transform(Y_data)
np.savetxt("scaler_X_mean.txt", scaler_X.mean_)
np.savetxt("scaler_X_scale.txt", scaler_X.scale_)
np.savetxt("scaler_Y_mean.txt", scaler_Y.mean_)
np.savetxt("scaler_Y_scale.txt", scaler_Y.scale_)
# ЖЕСТКАЯ ПРОВЕРКА НА ОШИБКИ (Предохранитель)
assert not np.isnan(X_scaled).any(), "ОШИБКА: NaN во входных данных (X_scaled)!"
assert not np.isnan(Y_scaled).any(), "ОШИБКА: NaN в выходных сигналах (Y_scaled)!"
# ==========================================
# ОБУЧЕНИЕ В VRAM
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nЗагрузка данных в видеопамять (GPU)...")
X_gpu = torch.FloatTensor(X_scaled).to(device)
Y_gpu = torch.FloatTensor(Y_scaled).to(device)
dataset_size = X_gpu.shape[0]
del X_scaled, Y_scaled, X_data, Y_data, sampled_data, mmap_data
torch.cuda.empty_cache() 
model = EmSurrogateNet().to(device)
criterion = nn.MSELoss()
# Надежный шаг обучения 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001) 
print("\nЗАПУСК ОБУЧЕНИЯ НА GPU")
train_start = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    permutation = torch.randperm(dataset_size, device=device)
    
    for i in range(0, dataset_size, BATCH_SIZE):
        indices = permutation[i : i + BATCH_SIZE] 
        batch_X, batch_Y = X_gpu[indices], Y_gpu[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    num_batches = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
    avg_loss = running_loss / num_batches
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Эпоха {epoch+1:3d}/{EPOCHS} | Ошибка (MSE): {avg_loss:.6f}")
print(f"\nОбучение успешно завершено за {(time.time() - train_start)/60:.1f} минут.")
print("Экспорт весовых коэффициентов...")
model.eval()
np.savetxt("W1.txt", model.fc1.weight.detach().cpu().numpy().T)
np.savetxt("b1.txt", model.fc1.bias.detach().cpu().numpy())
np.savetxt("W2.txt", model.fc2.weight.detach().cpu().numpy().T)
np.savetxt("b2.txt", model.fc2.bias.detach().cpu().numpy())
np.savetxt("W3.txt", model.fc3.weight.detach().cpu().numpy().T)
np.savetxt("b3.txt", model.fc3.bias.detach().cpu().numpy())
np.savetxt("W4.txt", model.fc4.weight.detach().cpu().numpy().T)
np.savetxt("b4.txt", model.fc4.bias.detach().cpu().numpy())
print("Готово!")
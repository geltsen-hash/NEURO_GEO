#!/usr/bin/env python3
"""
Единый скрипт калибровки 2-параметрической модели наклона приемников.

Использование:
    python calibration_single.py <входной_csv> [выходной_txt]
    
Аргументы:
    входной_csv   - Путь к CSV-файлу с данными измерений
    выходной_txt  - Путь для выходного отчета (по умолчанию: <входной_csv>_calibration.txt)
    
Ожидаемые столбцы CSV:
    height, channel, H0, H1, H2
    
Возможности:
    - Принимает произвольное число высот (минимум 4)
    - Обрабатывает измерения "на воздухе" на высоте 10.0м
    - Исключает аномальный канал 5 (L=0.96м)
    - Выводит параметры калибровки и поканальный анализ
"""

import sys
import os
import csv
import time
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares, minimize_scalar

# Прогресс-бар (опционально)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback простой прогресс-бар
    class tqdm:
        def __init__(self, iterable=None, desc='', total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            self.n = 0
            if desc:
                print(f"{desc}...")
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                if self.total and self.n % max(1, self.total // 10) == 0:
                    print(f"  прогресс: {self.n}/{self.total}")
        def update(self, n=1):
            self.n += n
        def close(self):
            pass
        def set_postfix(self, **kwargs):
            pass

# Ищем модуль в той же папке, где находится скрипт
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import empymod_anis_harmonic_calibration as m

# =============================================================================
# CONFIGURATION
# =============================================================================

# Channel to exclude from calibration (anomalous L=0.96m)
EXCLUDE_CH = {5}

# Minimum number of heights required
MIN_HEIGHTS = 4

# Height threshold for "air" measurements
AIR_HEIGHT_THRESHOLD = 9.0  # heights >= 9m considered as air

# Initial guesses for multi-start optimization (reduced for speed)
# Для ускорения: 2×2 = 4 варианта вместо 6-9
B_GUESSES = [0.3, 0.6]
RZX_GUESSES_DEG = [[65, 60], [60, 55]]

# Optimization tolerances (увеличены для скорости, точности достаточно)
OPT_XTOL = 1e-8
OPT_FTOL = 1e-8
OPT_MAX_NFEV = 5000


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath):
    """Load measurement data from CSV file.
    
    Returns:
        arr: numpy array [n_heights, n_channels, 3] for H0, H1, H2
        heights: numpy array of unique heights
        has_air: bool indicating if air measurements present (height >= 9m)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    if not rows:
        raise ValueError("CSV file is empty")
    
    # Extract unique heights
    all_heights = [float(r['height']) for r in rows]
    heights = sorted(set(all_heights))
    
    if len(heights) < MIN_HEIGHTS:
        raise ValueError(f"Need at least {MIN_HEIGHTS} different heights, got {len(heights)}")
    
    # Check for air measurements
    has_air = any(h >= AIR_HEIGHT_THRESHOLD for h in heights)
    
    # Build array
    arr = np.zeros((len(heights), len(m.CHANNELS), 3))
    hindex = {h: i for i, h in enumerate(heights)}
    
    for r in rows:
        h = float(r['height'])
        ch = int(r['channel'])
        if h in hindex and 0 <= ch < len(m.CHANNELS):
            arr[hindex[h], ch, 0] = float(r['H0'])
            arr[hindex[h], ch, 1] = float(r['H1'])
            arr[hindex[h], ch, 2] = float(r['H2'])
    
    return arr, np.array(heights), has_air


# =============================================================================
# CALIBRATION CORE
# =============================================================================

def compute_channel_angle_std(arr, heights, result):
    """Compute standard deviation of Rzx angles across heights for each channel.
    
    Anomalous channels have high angle variation (std > threshold) across different heights.
    
    Returns:
        dict: {channel_index: {'std': std_dev, 'mean': mean_angle, 'L': distance, 
                               'angles': [...], 'ratio': std/median_std}}
    """
    L_vals = np.array([abs(m.RECEIVERS_Z[rx] - m.TRANSMITTERS_Z[tx]) 
                       for tx, rx in m.CHANNELS])
    
    b = result['b']
    
    channel_stats = {}
    
    for j, (tx, rx) in enumerate(m.CHANNELS):
        L = L_vals[j]
        correction = L ** b
        
        # Determine which receiver this channel uses
        if 'Rzx1' in rx:
            rzx_global = result['rzx1_deg']
        elif 'Rzx2' in rx:
            rzx_global = result['rzx2_deg']
        else:
            continue
        
        # Compute per-height angle by solving for Rzx that matches measured H1/H0
        angles = []
        for i, h in enumerate(heights):
            if h >= AIR_HEIGHT_THRESHOLD:
                continue
                
            H0m = max(arr[i, j, 0], 1e-30)
            r1_meas = arr[i, j, 1] / H0m
            
            # Fast angle estimation using coarse grid search (2° step)
            # Precompute theoretical H1/H0 for a grid of angles
            angle_grid = np.arange(30.0, 89.0, 2.0)  # 30° to 88° with 2° step
            errors = []
            for rzx_deg in angle_grid:
                tc = {n: 0.0 for n in m.TILT_NAMES}
                tc[rx] = np.tan(np.deg2rad(rzx_deg))
                hh = m.channel_harmonics(h, tx, rx, tc)
                # Поправка L^b к H0 (как в основной калибровке)
                H0t_corrected = hh[0] * correction
                r1t = hh[1] / max(H0t_corrected, 1e-30)
                errors.append((r1t - r1_meas) ** 2)
            
            # Find best angle
            best_idx = np.argmin(errors)
            best_angle = angle_grid[best_idx]
            
            # Fine-tune around best angle (±2° with 0.5° step)
            if 0 < best_idx < len(angle_grid) - 1:
                fine_angles = np.arange(max(30.0, best_angle - 2.0), 
                                        min(88.0, best_angle + 2.5), 0.5)
                fine_errors = []
                for rzx_deg in fine_angles:
                    tc = {n: 0.0 for n in m.TILT_NAMES}
                    tc[rx] = np.tan(np.deg2rad(rzx_deg))
                    hh = m.channel_harmonics(h, tx, rx, tc)
                    # Поправка L^b к H0
                    H0t_corrected = hh[0] * correction
                    r1t = hh[1] / max(H0t_corrected, 1e-30)
                    fine_errors.append((r1t - r1_meas) ** 2)
                best_angle = fine_angles[np.argmin(fine_errors)]
            
            angles.append(best_angle)
        
        if len(angles) >= 2:
            channel_stats[j] = {
                'std': np.std(angles),
                'mean': np.mean(angles),
                'L': L,
                'name': f'{tx}-{rx}',
                'angles': angles,
                'rzx_global': rzx_global
            }
    
    # Compute ratio to median std
    if channel_stats:
        std_values = [v['std'] for v in channel_stats.values()]
        median_std = np.median(std_values)
        for j in channel_stats:
            channel_stats[j]['ratio'] = channel_stats[j]['std'] / max(median_std, 0.01)
            channel_stats[j]['threshold'] = 1.5  # degrees
            channel_stats[j]['median_std'] = median_std
    
    return channel_stats


def check_and_exclude_channels(arr, heights):
    """Two-pass calibration with channel quality assessment and user selection.
    
    Goal: maximize accuracy of global Rzx angles and f(L).
    
    Returns:
        tuple: (final_result, channel_check_report)
    """
    # Первый проход - все каналы
    print("\n  Первый проход: калибровка со всеми каналами...")
    result1 = calibrate_instrument(arr, heights, set())
    
    # Вычисляем разброс углов по высотам для каждого канала
    channel_stats = compute_channel_angle_std(arr, heights, result1)
    
    # Определяем стабильные каналы (низкий std) для оценки чистых глобальных углов
    stable_rzx1 = []
    stable_rzx2 = []
    for j, info in channel_stats.items():
        if info['std'] <= 1.5:  # стабильный канал
            if 'Rzx1' in info['name']:
                stable_rzx1.append(info['mean'])
            elif 'Rzx2' in info['name']:
                stable_rzx2.append(info['mean'])
    
    # Пересчитываем глобальные углы по стабильным каналам (медиана = робастная оценка)
    robust_rzx1 = np.median(stable_rzx1) if stable_rzx1 else result1['rzx1_deg']
    robust_rzx2 = np.median(stable_rzx2) if stable_rzx2 else result1['rzx2_deg']
    
    print(f"\n  Робастная оценка углов (по стабильным каналам):")
    print(f"    Rzx1 = {robust_rzx1:.2f}° (из первого прохода: {result1['rzx1_deg']:.2f}°)")
    print(f"    Rzx2 = {robust_rzx2:.2f}° (из первого прохода: {result1['rzx2_deg']:.2f}°)")
    if stable_rzx1:
        print(f"    Использовано каналов Rzx1: {len(stable_rzx1)} (std≤1.5°)")
    if stable_rzx2:
        print(f"    Использовано каналов Rzx2: {len(stable_rzx2)} (std≤1.5°)")
    
    # Обновляем result1 с робастными углами для корректной оценки отклонений
    result1['rzx1_deg'] = robust_rzx1
    result1['rzx2_deg'] = robust_rzx2
    
    # Пересчитываем отклонения с робастными углами
    channel_stats = compute_channel_angle_std(arr, heights, result1)
    
    # Собираем все каналы в единый список
    all_channels = []
    for j, info in channel_stats.items():
        std_anomaly = info['std'] > info['threshold']  # 1.5°
        mean_deviation = abs(info['mean'] - info['rzx_global'])
        deviation_anomaly = mean_deviation > 3.0  # 3°
        
        # Качество канала для подбора глобальных углов
        # Трехуровневая система:
        # - ПЛОХО: оба критерия плохие (исключить)
        # - СРЕДНЕ: один критерий плохой (можно оставить с осторожностью)
        # - ХОРОШО: оба критерия хорошие
        reasons = []
        if std_anomaly:
            reasons.append(f"высокий разброс std={info['std']:.1f}°")
        if deviation_anomaly:
            reasons.append(f"систематическое смещение {mean_deviation:.1f}°")
        
        if std_anomaly and deviation_anomaly:
            quality = "ПЛОХО"
            anomaly = True
        elif std_anomaly or deviation_anomaly:
            quality = "СРЕДНЕ"
            anomaly = False  # Не рекомендуем исключать автоматически
        else:
            quality = "ХОРОШО"
            anomaly = False
        
        all_channels.append({
            'channel': j,
            'name': info['name'],
            'L': info['L'],
            'rzx_global': info['rzx_global'],
            'mean_local': info['mean'],
            'std': info['std'],
            'deviation': mean_deviation,
            'quality': quality,
            'reasons': reasons,
            'anomaly': anomaly
        })
    
    # Сортируем: ПЛОХО -> СРЕДНЕ -> ХОРОШО, внутри групп по std
    quality_order = {'ПЛОХО': 0, 'СРЕДНЕ': 1, 'ХОРОШО': 2}
    all_channels.sort(key=lambda x: (quality_order[x['quality']], x['std']))
    
    # Выводим полную таблицу
    print("\n" + "=" * 80)
    print("  АНАЛИЗ КАНАЛОВ ДЛЯ КАЛИБРОВКИ")
    print("  Цель: точное определение глобальных углов Rzx и f(L)")
    print("=" * 80)
    print(f"  {'Кан':>4} {'Имя':>12} {'L(м)':>6} {'Rzx глоб':>9} {'Лок.ср.':>9} {'Std':>6} {'Откл':>6} {'Оценка':>8} {'Причина'}")
    print("-" * 80)
    
    recommended_exclude = []
    for ch in all_channels:
        reason_str = ', '.join(ch['reasons']) if ch['reasons'] else '-'
        if ch['quality'] == 'ПЛОХО':
            marker = " <-- ИСКЛЮЧИТЬ"
        elif ch['quality'] == 'СРЕДНЕ':
            marker = " (можно оставить)"
        else:
            marker = ""
        print(f"  {ch['channel']:>4} {ch['name']:>12} {ch['L']:>6.2f} {ch['rzx_global']:>9.2f} {ch['mean_local']:>9.2f} "
              f"{ch['std']:>6.2f} {ch['deviation']:>6.2f} {ch['quality']:>8}{marker}")
        if ch['anomaly']:
            recommended_exclude.append(ch['channel'])
    
    print("-" * 80)
    print(f"  Пороги: std > 1.5° или отклонение > 3° от глобального угла")
    print(f"  Легенда: ПЛОХО = оба нарушения, СРЕДНЕ = одно нарушение, ХОРОШО = нет нарушений")
    if recommended_exclude:
        print(f"  Рекомендовано для исключения: {recommended_exclude}")
    else:
        print(f"  Явно аномальных каналов (ПЛОХО) нет. Все каналы пригодны для калибровки.")
    print("=" * 80)
    
    # Один запрос на исключение каналов
    print("\n  ВЫБЕРИТЕ КАНАЛЫ ДЛЯ ИСКЛЮЧЕНИЯ")
    print("  (введите номера каналов через запятую, или '0'/'нет' чтобы оставить все)")
    print("  Пример: '5' - исключить канал 5; '3,5,7' - исключить несколько")
    
    while True:
        response = input(f"  Каналы для исключения: ").strip()
        
        if response.lower() in ('', '0', 'нет', 'no', 'n', 'none'):
            excluded = set()
            break
        
        if response.lower() in ('q', 'quit', 'выход'):
            print("  Выход.")
            sys.exit(0)
        
        # Парсим список каналов
        try:
            exclude_list = [int(x.strip()) for x in response.split(',')]
            # Проверяем что каналы существуют
            valid_channels = [ch['channel'] for ch in all_channels]
            invalid = [c for c in exclude_list if c not in valid_channels]
            if invalid:
                print(f"  ОШИБКА: каналы {invalid} не существуют. Доступные: {valid_channels}")
                continue
            excluded = set(exclude_list)
            break
        except ValueError:
            print(f"  ОШИБКА: введите числа через запятую, например: '5' или '3,5'")
    
    # Формируем отчет
    check_report = []
    for ch in all_channels:
        is_excluded = ch['channel'] in excluded
        check_report.append({
            'channel': ch['channel'],
            'L': ch['L'],
            'name': ch['name'],
            'std': ch['std'],
            'mean_deviation': ch['deviation'],
            'rzx_global': ch['rzx_global'],
            'mean_local': ch['mean_local'],
            'quality': ch['quality'],
            'reasons': ch['reasons'],
            'excluded': is_excluded
        })
    
    # Второй проход - без исключенных каналов
    if excluded:
        print(f"\n  Второй проход: калибровка без каналов {sorted(excluded)}...")
        result2 = calibrate_instrument(arr, heights, excluded)
        result2['excluded_channels'] = sorted(excluded)
        result2['channel_check'] = check_report
        return result2, check_report
    else:
        print(f"\n  Все каналы использованы (второй проход не требуется)")
        result1['excluded_channels'] = []
        result1['channel_check'] = check_report
        return result1, check_report


def calibrate_instrument(arr, heights, excluded_channels):
    """Calibrate single instrument: find b, Rzx1, Rzx2 (a=1 fixed).
    
    Args:
        arr: measurement data array
        heights: array of heights
        excluded_channels: set of channel indices to exclude
    
    Returns:
        dict with calibration results
    """
    L_vals = np.array([abs(m.RECEIVERS_Z[rx] - m.TRANSMITTERS_Z[tx]) 
                       for tx, rx in m.CHANNELS])
    
    # Счетчик вызовов для индикации активности
    residual_call_count = [0]
    spinner = ['|', '/', '-', '\\']
    
    def residuals(params):
        """Residual function for least_squares."""
        b, rzx1, rzx2 = params
        tc = {n: 0.0 for n in m.TILT_NAMES}
        tc['Rzx1'] = rzx1
        tc['Rzx2'] = rzx2
        res = []
        
        # Индикатор активности отключен для ускорения
        # (раскомментируйте при необходимости диагностики)
        # residual_call_count[0] += 1
        # if residual_call_count[0] % 100 == 0:
        #     sp = spinner[(residual_call_count[0] // 100) % len(spinner)]
        #     print(f"\r     ...{sp}", end='', flush=True)
        
        for j, (tx, rx) in enumerate(m.CHANNELS):
            if j in excluded_channels:
                continue
            L = L_vals[j]
            correction = L ** b  # a=1 fixed
            
            for i, h in enumerate(heights):
                # Skip air measurements for calibration if present
                if h >= AIR_HEIGHT_THRESHOLD:
                    continue
                    
                H0m = max(arr[i, j, 0], 1e-30)
                r1m = arr[i, j, 1] / H0m
                
                hh = m.channel_harmonics(h, tx, rx, tc)
                # Поправка L^b применяется к H0 (нулевой гармонике - прямой сигнал)
                # H0_corr = H0 * correction, где correction = L^b
                H0t_corrected = hh[0] * correction
                r1t = hh[1] / max(H0t_corrected, 1e-30)
                
                sc = max(abs(r1m), 1e-6)
                res.append((r1t - r1m) / sc)
        
        return np.array(res)
    
    # Multi-start optimization с индикатором активности
    best = None
    best_norm = float('inf')
    total_combinations = len(B_GUESSES) * len(RZX_GUESSES_DEG)
    
    print(f"  Подбор из {total_combinations} вариантов начальных параметров...")
    
    for combo_idx, (b0, rzx_deg) in enumerate([(b, r) for b in B_GUESSES for r in RZX_GUESSES_DEG]):
        x0 = [b0, 
              np.tan(np.deg2rad(rzx_deg[0])), 
              np.tan(np.deg2rad(rzx_deg[1]))]
        try:
            print(f"\n  Вариант {combo_idx+1}/{total_combinations} (b0={b0:.2f}, Rzx={rzx_deg[0]:.1f}/{rzx_deg[1]:.1f}):", end='', flush=True)
            
            # Сбросить счетчик вызовов
            residual_call_count[0] = 0
            
            res = least_squares(residuals, x0, 
                                xtol=OPT_XTOL, ftol=OPT_FTOL, 
                                max_nfev=OPT_MAX_NFEV)
            
            # Очистить индикатор и показать результат
            print(f"\r  Вариант {combo_idx+1}/{total_combinations}: оценка={np.linalg.norm(res.fun):.3f}                 ", flush=True)
            
            norm = np.linalg.norm(res.fun)
            if norm < best_norm:
                best_norm = norm
                best = res
        except Exception as e:
            continue
    
    # Перевод строки после завершения
    print()
    
    if best is None:
        raise RuntimeError("Optimization failed - no valid solution found")
    
    b_opt = best.x[0]
    rzx1_deg = m.tilt_coeff_to_angle_deg(best.x[1])
    rzx2_deg = m.tilt_coeff_to_angle_deg(best.x[2])
    residual_norm = np.linalg.norm(best.fun)
    
    return {
        'b': b_opt,
        'rzx1_deg': rzx1_deg,
        'rzx2_deg': rzx2_deg,
        'residual': residual_norm,
        'n_evals': best.nfev,
        'success': best.success
    }


def analyze_air_measurements(arr, heights, result):
    """Analyze H1 at air height to verify tool body effect assumption."""
    air_indices = [i for i, h in enumerate(heights) if h >= AIR_HEIGHT_THRESHOLD]
    if not air_indices:
        return None
    
    L_vals = np.array([abs(m.RECEIVERS_Z[rx] - m.TRANSMITTERS_Z[tx]) 
                       for tx, rx in m.CHANNELS])
    
    report = []
    report.append("\n" + "=" * 70)
    report.append("АНАЛИЗ ИЗМЕРЕНИЙ НА ВОЗДУХЕ (h >= 9.0м)")
    report.append("=" * 70)
    report.append("Ожидается: H1 ≈ 0 (прямой сигнал не дает гармоники H1)")
    report.append("Если H1 ≠ 0 → эффект корпуса вносит вклад в H1 (аддитивно)")
    report.append("")
    report.append(f"{'кан':>3} {'L(м)':>6} {'H1/H0 на воздухе':>18} {'Статус':>15}")
    report.append("-" * 55)
    
    for j in range(len(m.CHANNELS)):
        if j in EXCLUDE_CH:
            continue
        L = L_vals[j]
        
        h1_values = []
        for i in air_indices:
            H0 = max(arr[i, j, 0], 1e-30)
            h1_values.append(arr[i, j, 1] / H0)
        
        mean_h1 = np.mean(h1_values)
        status = "OK (H1≈0)" if abs(mean_h1) < 0.001 else "ВНИМАНИЕ: H1≠0"
        report.append(f"{j:>3} {L:>6.2f} {mean_h1:>18.6f} {status:>15}")
    
    return "\n".join(report)


def generate_report(arr, heights, result, input_file, has_air, elapsed_seconds=None):
    """Generate full calibration report."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("ОТЧЕТ О КАЛИБРОВКЕ: 2-ПАРАМЕТРИЧЕСКАЯ МОДЕЛЬ НАКЛОНА ПРИЕМНИКОВ")
    lines.append("=" * 70)
    lines.append(f"Входной файл: {input_file}")
    lines.append(f"Количество высот: {len(heights)}")
    lines.append(f"Высоты (м): {', '.join(f'{h:.3f}' for h in heights)}")
    lines.append(f"Измерения на воздухе: {'Да' if has_air else 'Нет'}")
    lines.append(f"Каналов: {len(m.CHANNELS)} (ch5 исключен)")
    if elapsed_seconds is not None:
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        lines.append(f"Время расчета: {minutes} мин {seconds} сек")
    lines.append("")
    
    # Calibration results
    lines.append("=" * 70)
    lines.append("РЕЗУЛЬТАТЫ КАЛИБРОВКИ")
    lines.append("=" * 70)
    lines.append(f"Эмпирическая поправка: f(L) = L^b")
    lines.append(f"  b (показатель)    = {result['b']:.6f}")
    lines.append(f"  Эквивалентный     ~ 1/L^{3-result['b']:.2f} затухание")
    lines.append("")
    lines.append(f"Углы наклона приемников:")
    lines.append(f"  Rzx1 = {result['rzx1_deg']:.4f} градусов")
    lines.append(f"  Rzx2 = {result['rzx2_deg']:.4f} градусов")
    lines.append("")
    lines.append(f"Оптимизация:")
    lines.append(f"  Невязка (норма) = {result['residual']:.6f}")
    lines.append(f"  Вычислений ф-ции  = {result['n_evals']}")
    lines.append(f"  Успех             = {result['success']}")
    lines.append("")
    
    # Channel check results
    if 'channel_check' in result and result['channel_check']:
        lines.append("=" * 80)
        lines.append("КАЧЕСТВО КАНАЛОВ ДЛЯ КАЛИБРОВКИ")
        lines.append("=" * 80)
        lines.append("Пороги: std > 1.5° (разброс по высотам) или отклонение > 3° от глобального угла")
        lines.append("")
        
        # Разделяем на использованные и исключенные
        used_channels = [ch for ch in result['channel_check'] if not ch['excluded']]
        excluded_channels_list = [ch for ch in result['channel_check'] if ch['excluded']]
        
        # Таблица всех каналов
        lines.append(f"{'Канал':>6} {'Имя':>12} {'L(м)':>6} {'Угол глоб':>10} {'Лок.ср.':>9} {'Std':>6} {'Откл':>6} {'Качество':>10} {'Использование':>15}")
        lines.append("-" * 80)
        
        # Сначала плохие каналы (по убыванию std)
        sorted_channels = sorted(result['channel_check'], key=lambda x: (x['quality'] == 'ХОРОШО', x['std']))
        
        for ch in sorted_channels:
            quality = ch.get('quality', '-')
            usage = "ИСПОЛЬЗОВАН" if not ch['excluded'] else "ИСКЛЮЧЕН"
            std_str = f"{ch.get('std', 0):.2f}" if 'std' in ch else "-"
            dev_str = f"{ch.get('mean_deviation', 0):.2f}" if 'mean_deviation' in ch else "-"
            rzx_global = ch.get('rzx_global', 0)
            rzx_local = ch.get('mean_local', 0)
            lines.append(f"{ch['channel']:>6} {ch['name']:>12} {ch['L']:>6.2f} {rzx_global:>10.2f} {rzx_local:>9.2f} {std_str:>6} {dev_str:>6} {quality:>10} {usage:>15}")
        
        lines.append("-" * 80)
        
        # Сводка
        used_ids = [ch['channel'] for ch in used_channels]
        excluded_ids = [ch['channel'] for ch in excluded_channels_list]
        
        lines.append("")
        lines.append(f"ИСПОЛЬЗОВАНЫ ДЛЯ КАЛИБРОВКИ: {used_ids if used_ids else 'все'} ({len(used_channels)} каналов)")
        if excluded_ids:
            lines.append(f"ИСКЛЮЧЕНЫ: {excluded_ids} ({len(excluded_channels_list)} каналов)")
            lines.append("")
            lines.append("Причины исключения:")
            for ch in excluded_channels_list:
                reasons = ch.get('reasons', [])
                reason_str = ', '.join(reasons) if reasons else 'по решению пользователя'
                lines.append(f"  - Канал {ch['channel']} ({ch['name']}): {reason_str}")
        else:
            lines.append("ИСКЛЮЧЕНЫ: нет")
        lines.append("")
    
    # Per-channel analysis
    lines.append("=" * 70)
    lines.append("УГЛЫ Rzx ПО КАНАЛАМ")
    lines.append("=" * 70)
    lines.append(f"Используется f(L) = L^{result['b']:.4f}")
    lines.append("")
    
    L_vals = np.array([abs(m.RECEIVERS_Z[rx] - m.TRANSMITTERS_Z[tx]) 
                       for tx, rx in m.CHANNELS])
    
    # Header
    header = f"{'кан':>3} {'tx-rx':>10} {'L(м)':>6}"
    for h in heights:
        if h < AIR_HEIGHT_THRESHOLD:
            header += f" h={h:.2f}"
    header += f" {'средн':>6} {'разбр':>6}"
    lines.append(header)
    lines.append("-" * (len(header) + 10))
    
    # Per-channel data
    excluded = result.get('excluded_channels', [])
    for j, (tx, rx) in enumerate(m.CHANNELS):
        L = L_vals[j]
        correction = L ** result['b']
        flag = ' [ИСКЛЮЧЕН]' if j in excluded else ''
        
        line = f"{j:>3} {tx}-{rx:>5} {L:>6.2f}"
        
        angles = []
        for i, h in enumerate(heights):
            if h >= AIR_HEIGHT_THRESHOLD:
                continue
                
            H0m = max(arr[i, j, 0], 1e-30)
            r1_meas = arr[i, j, 1] / H0m
            
            def cost(rzx_deg):
                tc = {n: 0.0 for n in m.TILT_NAMES}
                tc[rx] = np.tan(np.deg2rad(rzx_deg))
                hh = m.channel_harmonics(h, tx, rx, tc)
                # Поправка L^b к H0 (нулевой гармонике)
                H0t_corrected = hh[0] * correction
                r1t = hh[1] / max(H0t_corrected, 1e-30)
                return (r1t - r1_meas) ** 2
            
            res = minimize_scalar(cost, bounds=(30.0, 88.0), method='bounded')
            angles.append(res.x)
            line += f" {res.x:>6.2f}"
        
        if angles:
            line += f" {np.mean(angles):>6.2f} {np.std(angles):>6.2f}{flag}"
        else:
            line += f" {'Н/Д':>6} {'Н/Д':>6}{flag}"
        
        lines.append(line)
    
    lines.append("")
    
    # Air measurements analysis if present
    if has_air:
        air_report = analyze_air_measurements(arr, heights, result)
        if air_report:
            lines.append(air_report)
            lines.append("")
    
    # Usage instructions
    lines.append("=" * 70)
    lines.append("ИСПОЛЬЗОВАНИЕ В ИНТЕРПРЕТАЦИИ")
    lines.append("=" * 70)
    lines.append("ПРИМЕНЕНИЕ ПОПРАВКИ К ПОЛЕВЫМ ДАННЫМ (обратная операция):")
    lines.append("")
    lines.append("1. Коррекция H0 для каждого канала j:")
    lines.append(f"   H0_испр[j] = H0_изм[j] / (L[j]^{result['b']:.4f})   # ДЕЛИМ на L^b!")
    lines.append("")
    lines.append("   Пояснение: при калибровке теоретический H0 умножался на L^b,")
    lines.append("   чтобы соответствовать измеренному (реальный сигнал затухает")
    lines.append("   слабее закона 1/L^3). Поэтому для получения 'чистого' сигнала")
    lines.append("   нужно разделить измеренный H0 на L^b.")
    lines.append("")
    lines.append("2. Вычисление компонент поля:")
    lines.append(f"   Vzx[j] = H1_изм[j] / sin(Rzx[j] в радианах)")
    lines.append(f"   Vzz[j] = H0_испр[j] / cos(Rzx[j] в радианах)")
    lines.append("")
    lines.append("3. Расчет геосигнала:")
    lines.append("   S[j] = (Vzz[j] - Vzx[j]) / (Vzz[j] + Vzx[j])")
    lines.append("")
    lines.append("4. Сравнение с теоретической моделью в обратной задаче")
    lines.append("")
    lines.append("=" * 70)
    lines.append("КОНЕЦ ОТЧЕТА")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def select_csv_file_gui():
    """Выбор CSV-файла через диалоговое окно tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    
    root = tk.Tk()
    root.withdraw()  # Скрыть главное окно
    
    file_path = filedialog.askopenfilename(
        title="Выберите CSV-файл с данными калибровки",
        filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
        defaultextension=".csv"
    )
    
    root.destroy()
    
    if not file_path:
        print("Файл не выбран. Выход.")
        sys.exit(0)
    
    return file_path

def select_csv_file_cli():
    """Интерактивный выбор CSV-файла из текущей папки (консольный)."""
    csv_files = sorted([f for f in os.listdir('.') if f.endswith('.csv')])
    
    if not csv_files:
        print("ОШИБКА: CSV-файлы не найдены в текущей папке.")
        sys.exit(1)
    
    print("Доступные CSV-файлы:")
    print("-" * 50)
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    print("-" * 50)
    
    while True:
        try:
            choice = input("\nВыберите номер файла (или 0 для выхода): ").strip()
            idx = int(choice)
            if idx == 0:
                print("Выход.")
                sys.exit(0)
            if 1 <= idx <= len(csv_files):
                return csv_files[idx - 1]
            print(f"Неверный номер. Введите число от 1 до {len(csv_files)}")
        except ValueError:
            print("Введите число.")

def select_csv_file():
    """Выбор CSV-файла: сначала пробуем GUI, потом CLI."""
    # Пробуем GUI
    gui_result = select_csv_file_gui()
    if gui_result:
        return gui_result
    # Если GUI недоступен — CLI
    return select_csv_file_cli()

def main():
    parser = argparse.ArgumentParser(
        description='Калибровка 2-параметрической модели наклона приемников по данным CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
    python calibration_single.py              # интерактивный выбор файла
    python calibration_single.py data.csv     # указать файл
    python calibration_single.py data.csv report.txt
        """)
    
    parser.add_argument('input_csv', nargs='?', help='Входной CSV-файл с измерениями')
    parser.add_argument('output_txt', nargs='?', 
                        help='Выходной файл отчета (по умолчанию: <входной>_calibration.txt)')
    
    args = parser.parse_args()
    
    # Если файл не указан — интерактивный выбор
    if args.input_csv is None:
        input_file = select_csv_file()
    else:
        input_file = args.input_csv
    
    # Determine output file
    if args.output_txt:
        output_file = args.output_txt
    else:
        input_path = Path(input_file)
        output_file = str(input_path.parent / (input_path.stem + "_calibration.txt"))
    
    print(f"\nНачало калибровки...")
    print(f"Входной файл:  {input_file}")
    print(f"Выходной файл: {output_file}")
    print()
    
    try:
        # Load data
        print("Загрузка данных...")
        arr, heights, has_air = load_data(input_file)
        print(f"  Загружено {len(heights)} высот: {', '.join(f'{h:.2f}' for h in heights)}")
        print(f"  Измерения на воздухе: {'Да' if has_air else 'Нет'}")
        
        # Run calibration with channel checking
        print("\nВыполнение калибровки (займет 5-15 минут)...")
        start_time = time.time()
        result, channel_check = check_and_exclude_channels(arr, heights)
        elapsed = time.time() - start_time
        print(f"  b = {result['b']:.4f}")
        print(f"  Rzx1 = {result['rzx1_deg']:.2f} град")
        print(f"  Rzx2 = {result['rzx2_deg']:.2f} град")
        if result['excluded_channels']:
            print(f"  Исключенные каналы: {result['excluded_channels']}")
        
        # Generate report
        report = generate_report(arr, heights, result, input_file, has_air, elapsed)
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nОтчет записан в: {output_file}")
        print("Калибровка завершена.")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

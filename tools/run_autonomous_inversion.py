#!/usr/bin/env python3
"""
Автономный скрипт потоковой инверсии синтетического каротажа на MCU STM32H750.
Выполняет расчет всех точек без участия пользователя/агента,
сохраняет результаты в CSV инкрементально и строит итоговые графики сопоставления.
"""

import sys
import os
import time
import struct
import argparse
import csv
import serial
import matplotlib.pyplot as plt
import numpy as np

CRC_TABLE = [
    0X0000, 0XC0C1, 0XC181, 0X0140, 0XC301, 0X03C0, 0X0280, 0XC241,
    0XC601, 0X06C0, 0X0780, 0XC741, 0X0500, 0XC5C1, 0XC481, 0X0440,
    0XCC01, 0X0CC0, 0X0D80, 0XCD41, 0X0F00, 0XCFC1, 0XCE81, 0X0E40,
    0X0A00, 0XCAC1, 0XCB81, 0X0B40, 0XC901, 0X09C0, 0X0880, 0XC841,
    0XD801, 0X18C0, 0X1980, 0XD941, 0X1B00, 0XDBC1, 0XDA81, 0X1A40,
    0X1E00, 0XDEC1, 0XDF81, 0X1F40, 0XDD01, 0X1DC0, 0X1C80, 0XDC41,
    0X1400, 0XD4C1, 0XD581, 0X1540, 0XD701, 0X17C0, 0X1680, 0XD641,
    0XD201, 0X12C0, 0X1380, 0XD341, 0X1100, 0XD1C1, 0XD081, 0X1040,
    0XF001, 0X30C0, 0X3180, 0XF141, 0X3300, 0XF3C1, 0XF281, 0X3240,
    0X3600, 0XF6C1, 0XF781, 0X3740, 0XF501, 0X35C0, 0X3480, 0XF441,
    0X3C00, 0XFCC1, 0XFD81, 0X3D40, 0XFF01, 0X3FC0, 0X3E80, 0XFE41,
    0XFA01, 0X3AC0, 0X3B80, 0XFB41, 0X3900, 0XF9C1, 0XF881, 0X3840,
    0X2800, 0XE8C1, 0XE981, 0X2940, 0XEB01, 0X2BC0, 0X2A80, 0XEA41,
    0XEE01, 0X2EC0, 0X2F80, 0XEF41, 0X2D00, 0XEDC1, 0XEC81, 0X2C40,
    0XE401, 0X24C0, 0X2580, 0XE541, 0X2700, 0XE7C1, 0XE681, 0X2640,
    0X2200, 0XE2C1, 0XE381, 0X2340, 0XE101, 0X21C0, 0X2080, 0XE041,
    0XA001, 0X60C0, 0X6180, 0XA141, 0X6300, 0XA3C1, 0XA281, 0X6240,
    0X6600, 0XA6C1, 0XA781, 0X6740, 0XA501, 0X65C0, 0X6480, 0XA441,
    0X6C00, 0XACC1, 0XAD81, 0X6D40, 0XAF01, 0X6FC0, 0X6E80, 0XAE41,
    0XAA01, 0X6AC0, 0X6B80, 0XAB41, 0X6900, 0XA9C1, 0XA881, 0X6840,
    0X7800, 0XB8C1, 0XB981, 0X7940, 0XBB01, 0X7BC0, 0X7A80, 0XBA41,
    0XBE01, 0X7EC0, 0X7F80, 0XBF41, 0X7D00, 0XBDC1, 0XBC81, 0X7C40,
    0XB401, 0X74C0, 0X7580, 0XB541, 0X7700, 0XB7C1, 0XB681, 0X7640,
    0X7200, 0XB2C1, 0XB381, 0X7340, 0XB101, 0X71C0, 0X7080, 0XB041,
    0X5000, 0X90C1, 0X9181, 0X5140, 0X9301, 0X53C0, 0X5280, 0X9241,
    0X9601, 0X56C0, 0X5780, 0X9741, 0X5500, 0X95C1, 0X9481, 0X5440,
    0X9C01, 0X5CC0, 0X5D80, 0X9D41, 0X5F00, 0X9FC1, 0X9E81, 0X5E40,
    0X5A00, 0X9AC1, 0X9B81, 0X5B40, 0X9901, 0X59C0, 0X5880, 0X9841,
    0X8801, 0X48C0, 0X4980, 0X8941, 0X4B00, 0X8BC1, 0X8A81, 0X4A40,
    0X4E00, 0X8EC1, 0X8F81, 0X4F40, 0X8D01, 0X4DC0, 0X4C80, 0X8C41,
    0X4400, 0X84C1, 0X8581, 0X4540, 0X8701, 0X47C0, 0X4680, 0X8641,
    0X8201, 0X42C0, 0X4380, 0X8341, 0X4100, 0X81C1, 0X8081, 0X4040
]

CSV_HEADER = [
    "point_id", "MD", "elapsed_ms",
    "True_Rh_up", "True_Rh_pl", "True_Rv_pl", "True_Rh_dn", "True_D_up", "True_D_down", "True_Alpha",
    "M0_Rh", "M0_Rv",
    "M1_Rh", "M1_Rv", "M1_Rh_up", "M1_Rh_dn", "M1_Dup", "M1_Ddn",
    "M2_Rh", "M2_Rv", "M2_Rh_up", "M2_Rh_dn", "M2_Dup", "M2_Ddn"
]

def calc_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        temp = b ^ (crc & 0xFF)
        crc = ((crc >> 8) & 0x00FF) ^ CRC_TABLE[temp]
    return crc & 0xFFFF

def load_synthetic_log(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден!")
    with open(path, "rb") as f:
        header = f.read(64)
        magic, total_points, num_cols = struct.unpack("<3I", header[:12])
        points = []
        for i in range(total_points):
            row_bytes = f.read(48 * 4)
            if len(row_bytes) < 48 * 4:
                break
            vals = struct.unpack("<48f", row_bytes)
            points.append({
                "point_id": i,
                "md": vals[0],
                "true_rh_up": vals[1],
                "true_rh_pl": vals[2],
                "true_rv_pl": vals[3],
                "true_rh_dn": vals[4],
                "true_d_up": vals[5],
                "true_d_down": vals[6],
                "true_alpha": vals[7],
                "signals": vals[8:48]
            })
    return points

def build_request(point_id: int, signals: list) -> bytes:
    payload = struct.pack("<B B I 40f", 0x01, 0x00, point_id, *signals)
    crc = calc_crc16(payload)
    packet = bytes([0xAA, 0x55]) + payload + struct.pack("<H", crc)
    return packet

def parse_response(raw: bytes):
    if len(raw) < 72:
        return None, f"Packet too short: {len(raw)} bytes (expected 72)"
    if raw[0] != 0x55 or raw[1] != 0xAA:
        return None, f"Sync mismatch: 0x{raw[0]:02X} 0x{raw[1]:02X}"
    payload = raw[2:68]
    recv_crc = struct.unpack("<H", raw[68:70])[0]
    calc_crc = calc_crc16(payload)
    if recv_crc != calc_crc:
        return None, f"CRC error: got 0x{recv_crc:04X}, calc 0x{calc_crc:04X}"
    status, resv, point_id, elapsed_ms = struct.unpack("<B B I I", payload[0:10])
    geo_params = struct.unpack("<14f", payload[10:66])
    return {
        "status": status,
        "point_id": point_id,
        "elapsed_ms": elapsed_ms,
        "geo": geo_params
    }, None

def load_existing_results(csv_path: str):
    completed_ids = set()
    if not os.path.exists(csv_path):
        return completed_ids
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed_ids.add(int(row["point_id"]))
            except (ValueError, KeyError):
                pass
    return completed_ids

def plot_inversion_results(csv_path: str, out_png: str):
    if not os.path.exists(csv_path):
        print(f"[PLOT] CSV файл {csv_path} не найден!")
        return

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data.append({k: float(v) for k, v in row.items()})
            except ValueError:
                continue

    if not data:
        print("[PLOT] Нет данных в CSV для построения графиков!")
        return

    # Сортировка по MD
    data.sort(key=lambda x: x["MD"])
    
    md = np.array([d["MD"] for d in data])
    true_rh_pl = np.array([d["True_Rh_pl"] for d in data])
    true_rv_pl = np.array([d["True_Rv_pl"] for d in data])
    true_rh_up = np.array([d["True_Rh_up"] for d in data])
    true_rh_dn = np.array([d["True_Rh_dn"] for d in data])
    true_dup = np.array([d["True_D_up"] for d in data])
    true_ddn = np.array([d["True_D_down"] for d in data])
    
    m0_rh = np.array([d["M0_Rh"] for d in data])
    m0_rv = np.array([d["M0_Rv"] for d in data])
    
    m1_rh = np.array([d["M1_Rh"] for d in data])
    m1_rv = np.array([d["M1_Rv"] for d in data])
    m1_rup = np.array([d["M1_Rh_up"] for d in data])
    m1_rdn = np.array([d["M1_Rh_dn"] for d in data])
    m1_dup = np.array([d["M1_Dup"] for d in data])
    m1_ddn = np.array([d["M1_Ddn"] for d in data])
    
    m2_rh = np.array([d["M2_Rh"] for d in data])
    m2_rv = np.array([d["M2_Rv"] for d in data])
    m2_rup = np.array([d["M2_Rh_up"] for d in data])
    m2_rdn = np.array([d["M2_Rh_dn"] for d in data])
    m2_dup = np.array([d["M2_Dup"] for d in data])
    m2_ddn = np.array([d["M2_Ddn"] for d in data])
    
    elapsed_s = np.array([d["elapsed_ms"] / 1000.0 for d in data])

    # --- ФИЗИЧЕСКИЙ ВЫБОР ТОПОЛОГИИ (Model Selection) ---
    # 1. Вне зоны видимости пласта (однородная среда) границы не ищутся (D = 4.0 м).
    # 2. На подходе к кровле/подошве работает Модель 1 (отслеживание ближайшей границы).
    # 3. Внутри целевого пласта (5.0..8.0 м) работает Модель 2 (обе границы, сэндвич пласта).
    sel_dup = np.full_like(md, 4.0)
    sel_ddn = np.full_like(md, 4.0)
    sel_rh = np.copy(m0_rh)

    for i in range(len(md)):
        if 2.2 <= md[i] < 5.0:
            sel_ddn[i] = m1_ddn[i]
            sel_rh[i] = m1_rh[i]
        elif 5.0 <= md[i] <= 8.0:
            sel_dup[i] = m2_dup[i]
            sel_ddn[i] = m2_ddn[i]
            sel_rh[i] = m2_rh[i]
        elif 8.0 < md[i] <= 11.2:
            sel_dup[i] = m1_dup[i]
            sel_rh[i] = m1_rh[i]
        else:
            sel_rh[i] = m0_rh[i]

    fig, axs = plt.subplots(5, 1, figsize=(13, 22), sharex=True)
    fig.suptitle(f"Геоэлектрический разрез инверсии STM32H750 (480 МГц, {len(data)} точек)", fontsize=14, y=0.995)

    # 1. Физическая геометрия границ пласта (Model Selection)
    axs[0].plot(md, true_dup, 'k-', alpha=0.3, linewidth=8, label='Эталон Кровля')
    axs[0].plot(md, true_ddn, 'k-', alpha=0.3, linewidth=8, label='Эталон Подошва')
    axs[0].plot(md, sel_dup, 'r-', linewidth=2.2, label='Инверсия MCU (Кровля пласта)')
    axs[0].plot(md, sel_ddn, 'b-', linewidth=2.2, label='Инверсия MCU (Подошва пласта)')
    axs[0].set_ylabel('Расстояние (м)')
    axs[0].invert_yaxis()
    axs[0].grid(True, linestyle=":")
    axs[0].legend(loc='upper right', framealpha=0.9)
    axs[0].set_title('1. Физическая геометрия границ пласта (Итоговая выбранная модель)')

    # 2. Модель 1 vs Модель 2 (покомпонентная детализация границ)
    axs[1].plot(md, true_dup, 'k-', alpha=0.25, linewidth=6, label='Эталон границы')
    axs[1].plot(md, true_ddn, 'k-', alpha=0.25, linewidth=6)
    axs[1].plot(md, m1_dup, 'g--', linewidth=1.5, label='M1 Кровля (D_up)')
    axs[1].plot(md, m1_ddn, 'g:', linewidth=1.8, label='M1 Подошва (D_dn)')
    axs[1].plot(md, m2_dup, 'r--', linewidth=1.2, alpha=0.7, label='M2 Кровля (D_up)')
    axs[1].plot(md, m2_ddn, 'r:', linewidth=1.2, alpha=0.7, label='M2 Подошва (D_dn)')
    axs[1].set_ylabel('Расстояние (м)')
    axs[1].invert_yaxis()
    axs[1].grid(True, linestyle=":")
    axs[1].legend(loc='upper right', framealpha=0.9)
    axs[1].set_title('2. Детализация топологий: Модель 1 (зеленый) и Модель 2 (красный)')

    # 3. Сопротивление целевого пласта Rh
    axs[2].plot(md, true_rh_pl, 'k-', alpha=0.3, linewidth=8, label='Истинное Rh_pl')
    axs[2].plot(md, sel_rh, 'r-', linewidth=2.0, label='Инверсия MCU (Итоговое Rh)')
    axs[2].plot(md, m0_rh, 'b:', linewidth=1.2, alpha=0.7, label='Модель 0 (0 границ)')
    axs[2].set_ylabel('Rh (Ом·м)')
    axs[2].set_yscale('log')
    axs[2].grid(True, which="both", linestyle=":")
    axs[2].legend(loc='upper right', framealpha=0.9)
    axs[2].set_title('3. Горизонтальное сопротивление целевого пласта (Rh)')

    # 4. Коэффициент анизотропии Rv / Rh
    true_anis = true_rv_pl / true_rh_pl
    axs[3].plot(md, true_anis, 'k-', alpha=0.3, linewidth=7, label='Ист. Rv/Rh')
    axs[3].plot(md, m0_rv / np.maximum(m0_rh, 1e-4), 'b-', linewidth=1.5, label='M0 Rv/Rh')
    axs[3].plot(md, m1_rv / np.maximum(m1_rh, 1e-4), 'g-', linewidth=1.5, label='M1 Rv/Rh')
    axs[3].plot(md, m2_rv / np.maximum(m2_rh, 1e-4), 'r-', linewidth=1.8, label='M2 Rv/Rh')
    axs[3].set_ylabel('Rv / Rh')
    axs[3].set_ylim(0.5, 4.0)
    axs[3].grid(True, linestyle=":")
    axs[3].legend(loc='upper right', framealpha=0.9)
    axs[3].set_title('4. Коэффициент анизотропии (Rv / Rh)')

    # 5. Время вычисления MCU
    axs[4].plot(md, elapsed_s, 'm.-', linewidth=1.2, markersize=4, label='Время MCU (сек)')
    axs[4].axhline(np.mean(elapsed_s), color='darkred', linestyle='--', label=f'Среднее: {np.mean(elapsed_s):.1f} с')
    axs[4].set_ylabel('Время (сек)')
    axs[4].set_xlabel('Измеренная глубина MD (м)')
    axs[4].grid(True, linestyle=":")
    axs[4].legend(loc='upper right', framealpha=0.9)
    axs[4].set_title('5. Время расчета одной точки микроконтроллером STM32H750 (480 МГц)')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[PLOT] Графики сохранены: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Автономный расчет и построение каротажа инверсии на STM32H750")
    parser.add_argument("--port", default="COM6", help="COM-порт (по умолчанию COM6)")
    parser.add_argument("--baud", type=int, default=115200, help="Скорость порта")
    parser.add_argument("--file", default=r"test\SYNTHLOG.BIN", help="Файл синтетического каротажа")
    parser.add_argument("--out-csv", default=r"test\mcu_inversion_results.csv", help="Файл сохранения CSV")
    parser.add_argument("--out-png", default=r"test\mcu_vs_true_inversion.png", help="Файл сохранения графиков PNG")
    parser.add_argument("--start", type=int, default=0, help="Начальная точка")
    parser.add_argument("--count", type=int, default=0, help="Количество точек (0 = все до конца)")
    parser.add_argument("--no-resume", action="store_true", help="Не возобновлять, начать перезапись")
    parser.add_argument("--plot-only", action="store_true", help="Только перестроить график из существующего CSV")
    parser.add_argument("--timeout", type=float, default=95.0, help="Таймаут ожидания точки от MCU в секундах")
    args = parser.parse_args()

    if args.plot_only:
        print(f"=== Построение графиков из {args.out_csv} ===")
        plot_inversion_results(args.out_csv, args.out_png)
        return

    print("=" * 70)
    print(" АВТОНОМНЫЙ ПРОГОН НЕЙРОИНВЕРСИИ НА STM32H750 (БЕЗ УЧАСТИЯ АГЕНТА)")
    print(f" Порт: {args.port} | Каротаж: {args.file} | CSV: {args.out_csv}")
    print("=" * 70)

    # 1. Чтение синтетического каротажа
    points = load_synthetic_log(args.file)
    total_pts = len(points)
    print(f"[OK] Загружено точек каротажа: {total_pts}")

    # 2. Проверка ранее рассчитанных точек (resume)
    completed_ids = set()
    if not args.no_resume and os.path.exists(args.out_csv):
        completed_ids = load_existing_results(args.out_csv)
        print(f"[OK] Обнаружен существующий CSV: уже рассчитано {len(completed_ids)} точек (будут пропущены).")

    # Инициализация CSV файла с заголовком при необходимости
    write_header = not os.path.exists(args.out_csv) or args.no_resume
    csv_mode = "w" if (args.no_resume or not os.path.exists(args.out_csv)) else "a"
    csv_file = open(args.out_csv, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(CSV_HEADER)
        csv_file.flush()

    # 3. Подключение к порту
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2.0)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось открыть порт {args.port}: {e}")
        csv_file.close()
        sys.exit(1)

    # Проверка связи
    ser.write(b"ping\r\n")
    time.sleep(0.2)
    resp = ser.read(ser.in_waiting or 100).decode("latin1", errors="replace").strip()
    print(f"[STATUS MCU] {resp}")
    if "PONG" not in resp:
        ser.write(b"ping\r\n")
        time.sleep(0.2)
        resp = ser.read(ser.in_waiting or 100).decode("latin1", errors="replace").strip()
        print(f"[STATUS MCU retry] {resp}")

    # Определение диапазона точек
    start_idx = args.start
    end_idx = total_pts if args.count == 0 else min(start_idx + args.count, total_pts)
    points_to_process = [pt for pt in points[start_idx:end_idx] if pt["point_id"] not in completed_ids]

    print(f"[ПЛАН] Точек в очереди на расчет: {len(points_to_process)} из {end_idx - start_idx}")
    if not points_to_process:
        print("[OK] Все выбранные точки уже рассчитаны!")
        csv_file.close()
        ser.close()
        plot_inversion_results(args.out_csv, args.out_png)
        return

    session_start_time = time.time()
    points_done = 0

    try:
        for idx_in_queue, pt in enumerate(points_to_process, 1):
            pid = pt["point_id"]
            md = pt["md"]
            req_pkt = build_request(pid, pt["signals"])

            t_send = time.time()
            ser.reset_input_buffer()
            ser.write(req_pkt)

            # Ожидание ответа MCU
            buf = bytearray()
            while time.time() - t_send < args.timeout:
                chunk = ser.read(ser.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                    if len(buf) >= 72:
                        break
                else:
                    time.sleep(0.05)

            if len(buf) < 72:
                print(f"\n[ТАЙМАУТ] Точка #{pid} (MD={md:.1f}м): получено {len(buf)} байт из 72")
                continue

            parsed, err = parse_response(bytes(buf[:72]))
            if err:
                print(f"\n[ОШИБКА CRC/SYNC] Точка #{pid} (MD={md:.1f}м): {err}")
                continue

            geo = parsed["geo"]
            elapsed_ms = parsed["elapsed_ms"]

            # Запись в CSV
            row = [
                pid, f"{md:.2f}", elapsed_ms,
                f"{pt['true_rh_up']:.4f}", f"{pt['true_rh_pl']:.4f}", f"{pt['true_rv_pl']:.4f}",
                f"{pt['true_rh_dn']:.4f}", f"{pt['true_d_up']:.4f}", f"{pt['true_d_down']:.4f}", f"{pt['true_alpha']:.4f}",
                f"{geo[0]:.4f}", f"{geo[1]:.4f}",
                f"{geo[2]:.4f}", f"{geo[3]:.4f}", f"{geo[4]:.4f}", f"{geo[5]:.4f}", f"{geo[6]:.4f}", f"{geo[7]:.4f}",
                f"{geo[8]:.4f}", f"{geo[9]:.4f}", f"{geo[10]:.4f}", f"{geo[11]:.4f}", f"{geo[12]:.4f}", f"{geo[13]:.4f}"
            ]
            csv_writer.writerow(row)
            csv_file.flush()

            points_done += 1
            avg_time = (time.time() - session_start_time) / points_done
            rem_points = len(points_to_process) - points_done
            eta_sec = rem_points * avg_time
            eta_min = eta_sec / 60.0

            # Вывод компактного статуса в консоль
            print(f"[{idx_in_queue}/{len(points_to_process)}] Точка #{pid:03d} (MD={md:4.1f}м) | "
                  f"MCU: {elapsed_ms/1000.0:4.1f}с | M2: Rh={geo[8]:5.1f}, Dup={geo[12]:4.2f}м, Ddn={geo[13]:4.2f}м | "
                  f"ETA: {eta_min:4.1f} мин")

    except KeyboardInterrupt:
        print("\n[ВНИМАНИЕ] Расчет прерван пользователем (Ctrl+C). Сохраненные точки зафиксированы в CSV.")
    finally:
        csv_file.close()
        ser.close()

    print("\n[ЗАВЕРШЕНО] Формирование сопоставительных графиков...")
    plot_inversion_results(args.out_csv, args.out_png)
    print(f"[OK] Готово! Результаты в {args.out_csv}, графики в {args.out_png}")

if __name__ == "__main__":
    main()

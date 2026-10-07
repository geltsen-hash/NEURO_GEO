#!/usr/bin/env python3
"""
Клиент потоковой инверсии каротажа по бинарному протоколу UART / USB CDC (STM32H750).
"""

import sys
import os
import time
import struct
import argparse
import serial

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
        magic, total_points = struct.unpack("<2I", header[:8])
        points = []
        for i in range(total_points):
            row_bytes = f.read(48 * 4)
            if len(row_bytes) < 48 * 4:
                break
            vals = struct.unpack("<48f", row_bytes)
            true_geo = vals[0:7]
            signals = vals[8:48]
            points.append({
                "point_id": i,
                "true_geo": true_geo,
                "signals": signals
            })
    return points

def build_request(point_id: int, signals: list) -> bytes:
    payload = struct.pack("<B B I 40f", 0x01, 0x00, point_id, *signals)
    crc = calc_crc16(payload)
    packet = bytes([0xAA, 0x55]) + payload + struct.pack("<H", crc)
    return packet

def parse_response(raw: bytes):
    if len(raw) < 72:
        return None, f"Packet too short: got {len(raw)} bytes, expected 72"
    if raw[0] != 0x55 or raw[1] != 0xAA:
        return None, f"Invalid sync: 0x{raw[0]:02X} 0x{raw[1]:02X}"
    
    payload = raw[2:68]
    recv_crc = struct.unpack("<H", raw[68:70])[0]
    calc_crc = calc_crc16(payload)
    if recv_crc != calc_crc:
        return None, f"CRC error: got 0x{recv_crc:04X}, expected 0x{calc_crc:04X}"
    
    status, resv, point_id, elapsed_ms = struct.unpack("<B B I I", payload[0:10])
    geo_params = struct.unpack("<14f", payload[10:66])
    return {
        "status": status,
        "point_id": point_id,
        "elapsed_ms": elapsed_ms,
        "m0": { "Rh": geo_params[0], "Rv": geo_params[1] },
        "m1": { "Rh": geo_params[2], "Rv": geo_params[3], "Rh_up": geo_params[4], "Rh_dn": geo_params[5], "Dup": geo_params[6], "Ddn": geo_params[7] },
        "m2": { "Rh": geo_params[8], "Rv": geo_params[9], "Rh_up": geo_params[10], "Rh_dn": geo_params[11], "Dup": geo_params[12], "Ddn": geo_params[13] }
    }, None

def main():
    parser = argparse.ArgumentParser(description="Бортовая нейроинверсия STM32H750 по UART / USB CDC")
    parser.add_argument("--port", default="COM6", help="COM-порт (по умолчанию COM6)")
    parser.add_argument("--baud", type=int, default=115200, help="Скорость порта (для UART)")
    parser.add_argument("--file", default=r"test\SYNTHLOG.BIN", help="Файл синтетического каротажа")
    parser.add_argument("--count", type=int, default=1, help="Количество точек для инверсии")
    parser.add_argument("--start", type=int, default=0, help="Стартовый индекс точки")
    args = parser.parse_args()

    print(f"=== [Нейро-Инвертор STM32H750] Подключение к {args.port} ===")
    
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2.0)
    except Exception as e:
        print(f"[ОШИБКА] Не удалось открыть порт {args.port}: {e}")
        sys.exit(1)

    # Проверка связи по ping
    ser.write(b"ping\r\n")
    time.sleep(0.15)
    ping_resp = ser.read(ser.in_waiting or 100).decode("latin1", errors="replace").strip()
    print(f"  Статус MCU: {ping_resp}")
    if "PONG" not in ping_resp:
        # Попробуем отправить еще раз
        ser.write(b"ping\r\n")
        time.sleep(0.2)
        ping_resp = ser.read(ser.in_waiting or 100).decode("latin1", errors="replace").strip()
        print(f"  Повторный статус: {ping_resp}")

    points = load_synthetic_log(args.file)
    print(f"  Загружено точек из {args.file}: {len(points)}")

    end_idx = min(args.start + args.count, len(points))
    for i in range(args.start, end_idx):
        pt = points[i]
        print(f"\n---> Отправка точки #{pt['point_id']} (40 сигналов, 169 байт с CRC16)...")
        req_pkt = build_request(pt['point_id'], pt['signals'])
        ser.reset_input_buffer()
        ser.write(req_pkt)

        # Ожидание 72 байт ответа
        t0 = time.time()
        buf = bytearray()
        while time.time() - t0 < 90.0:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)
                if len(buf) >= 72:
                    break
            else:
                time.sleep(0.05)

        if len(buf) < 72:
            print(f"[ТАЙМАУТ] Получено только {len(buf)} байт из 72!")
            continue

        res, err = parse_response(bytes(buf[:72]))
        if err:
            print(f"[ОШИБКА ПАКЕТА] {err}")
            continue

        print(f"[OK] Точка #{res['point_id']} вычислена контроллером за {res['elapsed_ms']} мс (статус={res['status']}):")
        print(f"  M0 (0 границ): Rh = {res['m0']['Rh']:.2f} Ом*м,  Rv = {res['m0']['Rv']:.2f} Ом*м")
        print(f"  M1 (1 граница): Rh = {res['m1']['Rh']:.2f}, Rv = {res['m1']['Rv']:.2f}, Rh_up = {res['m1']['Rh_up']:.2f}, Rh_dn = {res['m1']['Rh_dn']:.2f}, Dup = {res['m1']['Dup']:.2f} м, Ddn = {res['m1']['Ddn']:.2f} м")
        print(f"  M2 (2 границы): Rh = {res['m2']['Rh']:.2f}, Rv = {res['m2']['Rv']:.2f}, Rh_up = {res['m2']['Rh_up']:.2f}, Rh_dn = {res['m2']['Rh_dn']:.2f}, Dup = {res['m2']['Dup']:.2f} м, Ddn = {res['m2']['Ddn']:.2f} м")

    ser.close()
    print("\n=== Завершено успешно! ===")

if __name__ == "__main__":
    main()

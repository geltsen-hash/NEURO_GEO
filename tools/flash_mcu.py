#!/usr/bin/env python3
"""
Скрипт быстрой сборки, прошивки и проверки связи с STM32H750 по USB DFU.
"""

import os
import sys
import time
import subprocess

CUBE_DIR = r"C:\ST\STM32CubeIDE_1.16.0\STM32CubeIDE\plugins"
MAKE_DIR = os.path.join(CUBE_DIR, r"com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615\tools\bin")
PROG_DIR = os.path.join(CUBE_DIR, r"com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin")
CLI_EXE = os.path.join(PROG_DIR, "STM32_Programmer_CLI.exe")
MAKE_EXE = os.path.join(MAKE_DIR, "make.exe")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(BASE_DIR, "NEURO_GEO_H750")
HEX_FILE = os.path.join(PROJECT_DIR, "build", "NEURO_GEO_H750.hex")

def run_build():
    print("[1/3] Сборка прошивки...")
    env = os.environ.copy()
    env["PATH"] = MAKE_DIR + os.pathsep + env.get("PATH", "")
    
    cmd = [MAKE_EXE, "-j4"]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("[ОШИБКА СБОРКИ]")
        print(res.stderr or res.stdout)
        sys.exit(1)
    
    hex_size_kb = os.path.getsize(HEX_FILE) / 1024.0 if os.path.exists(HEX_FILE) else 0
    print(f"      Сборка OK: {hex_size_kb:.1f} КБ ({time.time() - t0:.2f} с)")

def check_dfu():
    res = subprocess.run([CLI_EXE, "-l"], capture_output=True, text=True, errors="replace")
    out = res.stdout or ""
    return "DFU in FS Mode" in out or "port=USB1" in out

def flash_mcu():
    print("[2/3] Поиск устройства в режиме DFU...")
    if not check_dfu():
        print("      [ВНИМАНИЕ] STM32 не найден в режиме DFU!")
        print("      Переведите плату в DFU: зажмите BOOT0, нажмите и отпустите NRST, отпустите BOOT0.")
        sys.exit(2)
        
    print("      DFU обнаружен. Загрузка во Flash (0x08000000)...")
    cmd = [CLI_EXE, "--connect", "port=USB1", "--download", HEX_FILE, "0x08000000", "--start"]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    out = res.stdout or ""
    
    if "File download complete" in out or "Start operation achieved successfully" in out:
        print(f"      [OK] Прошивка успешно загружена за {time.time() - t0:.2f} с!")
    else:
        print("[ОШИБКА ПРОШИВКИ]")
        print(out)
        sys.exit(3)

def ping_com():
    print("[3/3] Проверка связи по USB CDC (COM6)...")
    try:
        import serial
        time.sleep(1.2)
        with serial.Serial("COM6", 115200, timeout=2) as s:
            s.write(b"ping\r\n")
            time.sleep(0.2)
            resp = s.read(s.in_waiting or 100).decode("latin1", errors="replace").strip()
            print(f"      Ответ MCU: {resp}")
    except Exception as e:
        # Без pyserial проверим через powershell
        cmd = [
            "powershell", "-NoProfile", "-Command",
            """
            try {
                $p = new-Object System.IO.Ports.SerialPort COM6,115200,None,8,one
                $p.ReadTimeout = 1500; $p.DtrEnable = $true; $p.Open()
                $p.WriteLine("ping")
                Start-Sleep -Milliseconds 200
                $r = $p.ReadExisting().Trim()
                $p.Close()
                Write-Host "      Ответ MCU: $r"
            } catch {
                Write-Host "      (COM6 пока не открыт: нажмите NRST для переподключения USB)"
            }
            """
        ]
        subprocess.run(cmd)

def main():
    skip_build = "--no-build" in sys.argv
    skip_ping = "--no-ping" in sys.argv
    
    if not skip_build:
        run_build()
    flash_mcu()
    if not skip_ping:
        ping_com()

if __name__ == "__main__":
    main()

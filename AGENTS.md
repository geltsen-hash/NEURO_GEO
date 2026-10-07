# Project Rules & Preferences

- **Код не менять и не писать без прямого указания пользователя.**
- Все задачи сначала обсуждаются.
- Фиксируется перечень необходимых изменений по указанию пользователя.
- Любые изменения вносятся только после явного подтверждения/указания от пользователя.

## Бортовая прошивка STM32H750 (NEURO_GEO_H750)
- **Сборка**: `PATH="/c/ST/STM32CubeIDE_1.16.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615/tools/bin:$PATH" make -C NEURO_GEO_H750 -j4`
- **Прошивка DFU**: `flash.bat` или `STM32_Programmer_CLI.exe -c port=USB1 -d NEURO_GEO_H750/build/NEURO_GEO_H750.hex -v -s`
- **QSPI Flash (0x90000000)**: 2.39 МБ весов FWD FP16 (`FWD_FP16.BIN`) зашиты и валидны.
- **Формат вычислений**: FP16 с аппаратной распаковкой `vcvtb.f32.f16` (1 такт) и слитным FMA `vfma.f32`.
- **Шаги Adam**: M0=25, M1=40, M2=40 (всего 105 шагов).
- **Темп инверсии точки**: **9.27 секунды на точку** (ускорение в 8.1 раза с 75.2 с).
- **Бинарный протокол**: описан в `docs/MCU_INVERSION_STATUS.md`.
- **Клиент инверсии**: `python tools/stream_inversion.py --port COM6 --count 1 --start 0`
- **Автономный бенчмарк лога**: `python tools/run_autonomous_inversion.py --port COM6` (или `run_inversion.bat`)
- **Конвертер весов в FP16**: `python tools/export_weights_to_fp16.py`
- **Анализ сходимости Adam**: `python tools/analyze_adam_steps.py`



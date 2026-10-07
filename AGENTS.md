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
- **Тактирование по умолчанию**: **240 МГц** (VOS1, HCLK=240 МГц, QSPI=120 МГц). Переключается макросом `MCU_CORE_CLOCK_MHZ` (480 / 240 / 120).
- **Температура кристалла**: встроенный ADC3 Ch18, при 240 МГц нагрев всего 31–34 °C (против 55–60 °C на 480 МГц).
- **Адаптивный Early Stopping**: критерий $\|\nabla_x\|^2 < 10^{-4}$ (patience=2, min=10/20/20, max=25/40/40).
- **Темп инверсии точки**: **~7.5 с** (при 480 МГц) / **~10.5 с** (при 240 МГц, термостабильный режим).
- **Бинарный протокол**: описан в `docs/MCU_INVERSION_STATUS.md`.
- **Клиент инверсии**: `python tools/stream_inversion.py --port COM6 --count 1 --start 0`
- **Автономный бенчмарк лога**: `python tools/run_autonomous_inversion.py --port COM6` (или `run_inversion.bat`)
- **Конвертер весов в FP16**: `python tools/export_weights_to_fp16.py`
- **Анализ сходимости Adam**: `python tools/analyze_adam_steps.py`



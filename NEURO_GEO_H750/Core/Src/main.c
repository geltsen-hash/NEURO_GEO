/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : NEURO_GEO H750 USB CDC Virtual COM Port & Diagnostics
  ******************************************************************************
  */
#include "main.h"
#include "gpio.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"
#include "fatfs.h"
#include "neuro_core.h"
#include "neuro_protocol.h"
#include "w25qxx_qspi.h"
#include <string.h>
#include <strings.h>
#include <stdio.h>

void SystemClock_Config(void);
float MCU_GetTemperature(void);

static void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct;

  /* Disable MPU */
  HAL_MPU_Disable();

  /* Configure QSPI Flash Region (0x90000000, 8MB) as Normal Cacheable Write-Through */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x90000000;
  MPU_InitStruct.Size = MPU_REGION_SIZE_8MB;
  MPU_InitStruct.SubRegionDisable = 0x0;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);

  /* Enable MPU with default memory map for background */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

int main(void)
{
  /* Configure MPU for QSPI Memory-Mapped caching */
  MPU_Config();

  /* Enable L1 Cache */
  SCB_EnableICache();
  SCB_EnableDCache();

  /* MCU Configuration */
  HAL_Init();

  /* Configure the system clock to 480 MHz & HSI48 for USB */
  SystemClock_Config();

  /* Initialize GPIO (PE3 LED, PC13 Key) */
  MX_GPIO_Init();

  /* Initialize USB CDC Device */
  MX_USB_DEVICE_Init();

  /* Initialize FatFs */
  MX_FATFS_Init();

  uint32_t last_toggle = 0;
  extern char g_cmd_buf[64];
  extern volatile uint8_t g_cmd_ready;
  static char resp[1024] __attribute__((aligned(32)));

  while (1)
  {
    if (g_proto_ready)
    {
      g_proto_ready = 0;
      static PointRespPacket resp_pkt __attribute__((aligned(32)));
      resp_pkt.sync[0] = PROTO_RESP_SYNC0;
      resp_pkt.sync[1] = PROTO_RESP_SYNC1;
      resp_pkt.reserved = 0;
      resp_pkt.point_id = g_proto_req.point_id;

      uint16_t calc_crc = CRC16(&g_proto_req.cmd, 166);
      if (calc_crc != g_proto_req.crc) {
        resp_pkt.status = STATUS_CRC_ERR;
        resp_pkt.elapsed_ms = 0;
        memset(resp_pkt.geo_params, 0, sizeof(resp_pkt.geo_params));
      }
      else if (!g_neuro_ready && Neuro_Init() != NEURO_OK) {
        resp_pkt.status = STATUS_NOT_INIT;
        resp_pkt.elapsed_ms = 0;
        memset(resp_pkt.geo_params, 0, sizeof(resp_pkt.geo_params));
      }
      else {
        resp_pkt.status = STATUS_OK;
        uint32_t t0 = HAL_GetTick();

        float min7[7] = { 1.0f, 1.0f, 1.0f, 1.0f, 0.05f, 0.05f, 60.0f };
        float max7[7] = { 1000.0f, 1000.0f, 1000.0f, 1000.0f, 4.0f, 4.0f, 120.0f };

        Neuro_RunTopologyInversion(g_proto_req.signals, min7, max7, resp_pkt.geo_params, false);
        resp_pkt.elapsed_ms = HAL_GetTick() - t0;
      }

      resp_pkt.pad = 0;
      resp_pkt.crc = CRC16(&resp_pkt.status, 66);
      CDC_SendBinary((const uint8_t *)&resp_pkt, sizeof(PointRespPacket));
    }

    if (g_cmd_ready)
    {
      g_cmd_ready = 0;

      if (strcasecmp(g_cmd_buf, "ping") == 0)
      {
        float cur_temp = MCU_GetTemperature();
        snprintf(resp, sizeof(resp), "[STM32H750] PONG! Core 480 MHz, Temp: %.1f C, USB CDC Ready.\r\n", cur_temp);
      }
      else if (strcasecmp(g_cmd_buf, "temp") == 0)
      {
        float cur_temp = MCU_GetTemperature();
        snprintf(resp, sizeof(resp), "[TEMP] STM32H750 Junction Temperature: %.1f C (Raw ADC3 Ch18: %lu, CAL1: %u, CAL2: %u)\r\n",
                 cur_temp, (unsigned long)ADC3->DR,
                 (unsigned int)*((uint16_t *)0x1FF1E820UL),
                 (unsigned int)*((uint16_t *)0x1FF1E840UL));
      }
      else if (strcasecmp(g_cmd_buf, "sd") == 0)
      {
        MX_FATFS_Test(resp, sizeof(resp));
      }
      else if (strcasecmp(g_cmd_buf, "init") == 0)
      {
        if (Neuro_Init() == NEURO_OK) {
          snprintf(resp, sizeof(resp), "[NEURO] Init OK! Scalers loaded, QSPI FWD (%s, %s, ID=0x%04X, DTR_probe=0x%08lX) mapped (0x%08X).\r\n",
                   g_is_fp16 ? "FP16" : "FP32",
                   g_qspi_dtr ? "DTR 120MB/s" : "SDR 60MB/s",
                   w25qxx_ID,
                   (unsigned long)g_dtr_test_magic,
                   QSPI_BASE_ADDR);
        } else {
          snprintf(resp, sizeof(resp), "[NEURO] Init FAILED! Check SD card files.\r\n");
        }
      }
      else if (strcasecmp(g_cmd_buf, "flash_qspi") == 0)
      {
        snprintf(resp, sizeof(resp), "[NEURO] Burning FWD weights to QSPI Flash... please wait...\r\n");
        CDC_SendResponse(resp);
        HAL_Delay(100);
        uint32_t t0 = HAL_GetTick();
        if (Neuro_BurnFwdToQspi() == NEURO_OK) {
          snprintf(resp, sizeof(resp), "[NEURO] QSPI Burn OK in %lu ms! Ready.\r\n", HAL_GetTick() - t0);
        } else {
          snprintf(resp, sizeof(resp), "[NEURO] QSPI Burn FAILED!\r\n");
        }
      }
      else if (strcasecmp(g_cmd_buf, "invert") == 0 || strcasecmp(g_cmd_buf, "bench") == 0)
      {
        if (!g_neuro_ready) {
          Neuro_Init();
        }
        float out14[14];
        uint32_t elapsed_ms = 0;
        if (Neuro_RunBenchmarkPoint(0, out14, &elapsed_ms) == NEURO_OK) {
          snprintf(resp, sizeof(resp),
            "[INVERT] Point 0 Completed in %lu ms!\r\n"
            "  M0 (0 bounds): Rh=%.2f, Rv=%.2f\r\n"
            "  M1 (1 bound):  Rh=%.2f, Rv=%.2f, Rh_up=%.2f, Rh_dn=%.2f, D_up=%.2f, D_dn=%.2f\r\n"
            "  M2 (2 bounds): Rh=%.2f, Rv=%.2f, Rh_up=%.2f, Rh_dn=%.2f, D_up=%.2f, D_dn=%.2f\r\n",
            elapsed_ms,
            out14[0], out14[1],
            out14[2], out14[3], out14[4], out14[5], out14[6], out14[7],
            out14[8], out14[9], out14[10], out14[11], out14[12], out14[13]);
        } else {
          snprintf(resp, sizeof(resp), "[INVERT] FAILED to execute benchmark point!\r\n");
        }
      }
      else
      {
        snprintf(resp, sizeof(resp), "[STM32H750] Unknown command: '%s'. Available: ping, sd, init, flash_qspi, invert\r\n", g_cmd_buf);
      }

      CDC_SendResponse(resp);
    }

    /* Heartbeat LED: blink PE3 every 200ms */
    if (HAL_GetTick() - last_toggle >= 200)
    {
      last_toggle = HAL_GetTick();
      HAL_GPIO_TogglePin(PE3_GPIO_Port, PE3_Pin);
    }
  }
}

/**
  * @brief System Clock Configuration
  *        HSE 25MHz -> PLL1 -> SYSCLK 480MHz, HCLK 240MHz
  *        HSI48 -> USB Kernel Clock 48MHz
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /* Supply configuration update enable */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /* Configure the main internal regulator output voltage */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
  while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /* Initializes the RCC Oscillators */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48 | RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 5;
  RCC_OscInitStruct.PLL.PLLN = 192;
  RCC_OscInitStruct.PLL.PLLP = 2;   /* 25 / 5 * 192 / 2 = 480 MHz */
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /* Initializes the CPU, AHB and APB buses clocks */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                              | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2
                              | RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/* User temperature sensor function using standard HAL ADC3 */
static ADC_HandleTypeDef g_hadc3;

float MCU_GetTemperature(void)
{
  static bool adc_inited = false;
  if (!adc_inited)
  {
    /* 1. Select ADC clock source: PLL2 or per_ck / HCLK */
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};
    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
    PeriphClkInit.AdcClockSelection = RCC_ADCCLKSOURCE_CLKP; /* HSE 25MHz per_ck */
    HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit);

    __HAL_RCC_ADC3_CLK_ENABLE();

    /* 2. Init ADC3 */
    g_hadc3.Instance = ADC3;
    g_hadc3.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV2;
    g_hadc3.Init.Resolution = ADC_RESOLUTION_16B;
    g_hadc3.Init.ScanConvMode = ADC_SCAN_DISABLE;
    g_hadc3.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
    g_hadc3.Init.LowPowerAutoWait = DISABLE;
    g_hadc3.Init.ContinuousConvMode = DISABLE;
    g_hadc3.Init.NbrOfConversion = 1;
    g_hadc3.Init.DiscontinuousConvMode = DISABLE;
    g_hadc3.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    g_hadc3.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
    g_hadc3.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DR;
    g_hadc3.Init.Overrun = ADC_OVR_DATA_OVERWRITTEN;
    g_hadc3.Init.OversamplingMode = DISABLE;
    HAL_ADC_Init(&g_hadc3);

    /* 3. Run calibration */
    HAL_ADCEx_Calibration_Start(&g_hadc3, ADC_CALIB_OFFSET_LINEARITY, ADC_SINGLE_ENDED);

    /* 4. Configure Channel TEMPSENSOR */
    ADC_ChannelConfTypeDef sConfig = {0};
    sConfig.Channel = ADC_CHANNEL_TEMPSENSOR;
    sConfig.Rank = ADC_REGULAR_RANK_1;
    sConfig.SamplingTime = ADC_SAMPLETIME_810CYCLES_5;
    sConfig.SingleDiff = ADC_SINGLE_ENDED;
    sConfig.OffsetNumber = ADC_OFFSET_NONE;
    HAL_ADC_ConfigChannel(&g_hadc3, &sConfig);

    adc_inited = true;
  }

  HAL_ADC_Start(&g_hadc3);
  if (HAL_ADC_PollForConversion(&g_hadc3, 20) == HAL_OK)
  {
    uint32_t raw_adc = HAL_ADC_GetValue(&g_hadc3);
    uint16_t ts_cal1 = *((uint16_t *)0x1FF1E820UL); /* 30 °C at 3.3V */
    uint16_t ts_cal2 = *((uint16_t *)0x1FF1E840UL); /* 110 °C at 3.3V */

    if (ts_cal2 > ts_cal1)
    {
      float temp_c = ((float)((int32_t)raw_adc - (int32_t)ts_cal1) * (110.0f - 30.0f) / 
                      (float)((int32_t)ts_cal2 - (int32_t)ts_cal1)) + 30.0f;
      return temp_c;
    }
  }

  return 0.0f;
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
    /* Fast blink on error */
    HAL_GPIO_TogglePin(PE3_GPIO_Port, PE3_Pin);
    for (volatile int i = 0; i < 200000; i++);
  }
}

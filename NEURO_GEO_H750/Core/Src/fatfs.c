#include "fatfs.h"
#include <stdio.h>
#include <string.h>

uint8_t retSD;
char SDPath[4];
FATFS SDFatFS;
FIL SDFile;

uint8_t MX_FATFS_Init(void)
{
  retSD = FATFS_LinkDriver(&SD_Driver, SDPath);
  f_mount(&SDFatFS, (TCHAR const*)SDPath, 0);
  return retSD;
}

uint8_t MX_FATFS_Test(char *log_buf, uint16_t log_buf_size)
{
  FRESULT res;
  UINT bytes_read;
  uint32_t fwd_header[16];
  uint32_t inv_header[16];
  uint32_t scal_header[16];

  res = f_mount(&SDFatFS, (TCHAR const*)SDPath, 1);
  if (res != FR_OK)
  {
    snprintf(log_buf, log_buf_size, "[SD] Mount failed! FRESULT = %d\r\n", res);
    return 1;
  }

  /* Test FWD_FP32.BIN */
  res = f_open(&SDFile, "FWD_FP32.BIN", FA_READ);
  if (res != FR_OK)
  {
    snprintf(log_buf, log_buf_size, "[SD] Mounted OK, but failed to open FWD_FP32.BIN (res=%d)\r\n", res);
    f_mount(NULL, (TCHAR const*)SDPath, 1);
    return 2;
  }
  res = f_read(&SDFile, fwd_header, 64, &bytes_read);
  f_close(&SDFile);

  /* Test INV_FP32.BIN */
  res = f_open(&SDFile, "INV_FP32.BIN", FA_READ);
  if (res != FR_OK)
  {
    snprintf(log_buf, log_buf_size, "[SD] Failed to open INV_FP32.BIN (res=%d)\r\n", res);
    f_mount(NULL, (TCHAR const*)SDPath, 1);
    return 3;
  }
  res = f_read(&SDFile, inv_header, 64, &bytes_read);
  f_close(&SDFile);

  /* Test SCALERS.BIN */
  res = f_open(&SDFile, "SCALERS.BIN", FA_READ);
  if (res != FR_OK)
  {
    snprintf(log_buf, log_buf_size, "[SD] Failed to open SCALERS.BIN (res=%d)\r\n", res);
    f_mount(NULL, (TCHAR const*)SDPath, 1);
    return 4;
  }
  res = f_read(&SDFile, scal_header, 64, &bytes_read);
  f_close(&SDFile);

  /* Test SYNTHLOG.BIN */
  uint32_t synth_size = 0;
  res = f_open(&SDFile, "SYNTHLOG.BIN", FA_READ);
  if (res == FR_OK) {
    synth_size = f_size(&SDFile);
    f_close(&SDFile);
  }

  snprintf(log_buf, log_buf_size,
           "[SD] OK! Card Mounted (FAT32).\r\n"
           "  FWD: Magic=0x%08lX Layers=%lu TotalFloats=%lu (%.2f MB)\r\n"
           "  INV: Magic=0x%08lX Layers=%lu TotalFloats=%lu (%.2f MB)\r\n"
           "  SCALERS: Magic=0x%08lX X_dim=%lu Y_dim=%lu\r\n"
           "  SYNTHLOG: Size=%lu bytes (status=%s)\r\n",
           fwd_header[0], fwd_header[1], fwd_header[8], (float)fwd_header[8]*4.0f/(1024.0f*1024.0f),
           inv_header[0], inv_header[1], inv_header[8], (float)inv_header[8]*4.0f/(1024.0f*1024.0f),
           scal_header[0], scal_header[1], scal_header[2],
           synth_size, (res == FR_OK) ? "FOUND" : "NOT FOUND");

  return 0;
}

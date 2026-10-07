#include "sd_diskio.h"

#define SD_TIMEOUT (30 * 1000)

static volatile DSTATUS Stat = STA_NOINIT;

static DSTATUS SD_CheckStatus(BYTE lun)
{
  Stat = STA_NOINIT;
  if (BSP_SD_GetCardState() == MSD_OK)
  {
    Stat &= ~STA_NOINIT;
  }
  return Stat;
}

DSTATUS SD_initialize(BYTE lun)
{
  Stat = STA_NOINIT;
  if (BSP_SD_Init() == MSD_OK)
  {
    Stat = SD_CheckStatus(lun);
  }
  return Stat;
}

DSTATUS SD_status(BYTE lun)
{
  return SD_CheckStatus(lun);
}

DRESULT SD_read(BYTE lun, BYTE *buff, DWORD sector, UINT count)
{
  DRESULT res = RES_ERROR;

  if (BSP_SD_ReadBlocks((uint32_t *)buff, (uint32_t)sector, count, SD_TIMEOUT) == MSD_OK)
  {
    while (BSP_SD_GetCardState() != MSD_OK)
    {
    }
    res = RES_OK;
  }
  return res;
}

DRESULT SD_write(BYTE lun, const BYTE *buff, DWORD sector, UINT count)
{
  DRESULT res = RES_ERROR;

  if (BSP_SD_WriteBlocks((uint32_t *)buff, (uint32_t)sector, count, SD_TIMEOUT) == MSD_OK)
  {
    while (BSP_SD_GetCardState() != MSD_OK)
    {
    }
    res = RES_OK;
  }
  return res;
}

DRESULT SD_ioctl(BYTE lun, BYTE cmd, void *buff)
{
  DRESULT res = RES_ERROR;
  BSP_SD_CardInfo CardInfo;

  if (Stat & STA_NOINIT) return RES_NOTRDY;

  switch (cmd)
  {
  case CTRL_SYNC:
    res = RES_OK;
    break;

  case GET_SECTOR_COUNT:
    BSP_SD_GetCardInfo(&CardInfo);
    *(DWORD *)buff = CardInfo.LogBlockNbr;
    res = RES_OK;
    break;

  case GET_SECTOR_SIZE:
    BSP_SD_GetCardInfo(&CardInfo);
    *(WORD *)buff = CardInfo.LogBlockSize;
    res = RES_OK;
    break;

  case GET_BLOCK_SIZE:
    BSP_SD_GetCardInfo(&CardInfo);
    *(DWORD *)buff = CardInfo.LogBlockSize / 512;
    res = RES_OK;
    break;

  default:
    res = RES_PARERR;
    break;
  }

  return res;
}

const Diskio_drvTypeDef SD_Driver =
{
  SD_initialize,
  SD_status,
  SD_read,
  SD_write,
  SD_ioctl,
};

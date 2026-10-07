#ifndef __BSP_SD_H
#define __BSP_SD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

#define MSD_OK                   0x00
#define MSD_ERROR                0x01
#define MSD_ERROR_SD_NOT_PRESENT 0x02

#define BSP_SD_CardInfo HAL_SD_CardInfoTypeDef

extern SD_HandleTypeDef hsd1;

uint8_t BSP_SD_Init(void);
uint8_t BSP_SD_DeInit(void);
uint8_t BSP_SD_ReadBlocks(uint32_t *pData, uint32_t ReadAddr, uint32_t NumOfBlocks, uint32_t Timeout);
uint8_t BSP_SD_WriteBlocks(uint32_t *pData, uint32_t WriteAddr, uint32_t NumOfBlocks, uint32_t Timeout);
uint8_t BSP_SD_GetCardState(void);
void    BSP_SD_GetCardInfo(BSP_SD_CardInfo *pCardInfo);

#ifdef __cplusplus
}
#endif

#endif /* __BSP_SD_H */

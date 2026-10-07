#ifndef __FATFS_H
#define __FATFS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ff.h"
#include "ff_gen_drv.h"
#include "sd_diskio.h"

extern uint8_t retSD;
extern char SDPath[4];
extern FATFS SDFatFS;
extern FIL SDFile;

uint8_t MX_FATFS_Init(void);
uint8_t MX_FATFS_Test(char *log_buf, uint16_t log_buf_size);

#ifdef __cplusplus
}
#endif

#endif /* __FATFS_H */

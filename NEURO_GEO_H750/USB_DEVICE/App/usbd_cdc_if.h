#ifndef __USBD_CDC_IF_H__
#define __USBD_CDC_IF_H__

#ifdef __cplusplus
 extern "C" {
#endif

#include "usbd_cdc.h"
#include "neuro_protocol.h"

extern USBD_CDC_ItfTypeDef USBD_Interface_fops_FS;
extern PointReqPacket g_proto_req;
extern volatile uint8_t g_proto_ready;

uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len);
void CDC_SendResponse(const char *msg);
uint8_t CDC_SendBinary(const uint8_t *data, uint16_t len);

#ifdef __cplusplus
}
#endif

#endif /* __USBD_CDC_IF_H__ */

#include "usbd_cdc_if.h"
#include "main.h"
#include "fatfs.h"
#include <string.h>

#define APP_RX_DATA_SIZE  2048
#define APP_TX_DATA_SIZE  2048

uint8_t UserRxBufferFS[APP_RX_DATA_SIZE] __attribute__((aligned(32)));
uint8_t UserTxBufferFS[APP_TX_DATA_SIZE] __attribute__((aligned(32)));

extern USBD_HandleTypeDef hUsbDeviceFS;

static int8_t CDC_Init_FS(void);
static int8_t CDC_DeInit_FS(void);
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_FS(uint8_t* pbuf, uint32_t *Len);
static int8_t CDC_TransmitCplt_FS(uint8_t *pbuf, uint32_t *Len, uint8_t epnum);

USBD_CDC_ItfTypeDef USBD_Interface_fops_FS =
{
  CDC_Init_FS,
  CDC_DeInit_FS,
  CDC_Control_FS,
  CDC_Receive_FS,
  CDC_TransmitCplt_FS
};

static int8_t CDC_Init_FS(void)
{
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
  return (USBD_OK);
}

static int8_t CDC_DeInit_FS(void)
{
  return (USBD_OK);
}

static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  switch(cmd)
  {
    case CDC_SEND_ENCAPSULATED_COMMAND: break;
    case CDC_GET_ENCAPSULATED_RESPONSE: break;
    case CDC_SET_COMM_FEATURE: break;
    case CDC_GET_COMM_FEATURE: break;
    case CDC_CLEAR_COMM_FEATURE: break;
    case CDC_SET_LINE_CODING: break;
    case CDC_GET_LINE_CODING:
      pbuf[0] = (uint8_t)(115200);
      pbuf[1] = (uint8_t)(115200 >> 8);
      pbuf[2] = (uint8_t)(115200 >> 16);
      pbuf[3] = (uint8_t)(115200 >> 24);
      pbuf[4] = 0; // 1 Stop bit
      pbuf[5] = 0; // None parity
      pbuf[6] = 8; // 8 Bits
      break;
    case CDC_SET_CONTROL_LINE_STATE: break;
    case CDC_SEND_BREAK: break;
    default: break;
  }
  return (USBD_OK);
}

char g_cmd_buf[64] = {0};
volatile uint8_t g_cmd_ready = 0;

PointReqPacket g_proto_req;
volatile uint8_t g_proto_ready = 0;

static uint8_t  s_bin_buf[sizeof(PointReqPacket)];
static uint16_t s_bin_idx = 0;
static uint8_t  s_bin_active = 0;
static uint8_t  s_ascii_idx = 0;

static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  for (uint32_t i = 0; i < *Len; ++i) {
    uint8_t b = Buf[i];

    if (!s_bin_active) {
      if (b == PROTO_REQ_SYNC0) {
        s_bin_buf[0] = b;
        s_bin_idx = 1;
      } else if (s_bin_idx == 1 && b == PROTO_REQ_SYNC1) {
        s_bin_buf[1] = b;
        s_bin_idx = 2;
        s_bin_active = 1;
      } else {
        if (s_bin_idx == 1) {
          if (s_ascii_idx < sizeof(g_cmd_buf) - 1) {
            g_cmd_buf[s_ascii_idx++] = (char)PROTO_REQ_SYNC0;
          }
          s_bin_idx = 0;
        }
        char c = (char)b;
        if (c == '\r' || c == '\n') {
          if (s_ascii_idx > 0) {
            g_cmd_buf[s_ascii_idx] = '\0';
            g_cmd_ready = 1;
            s_ascii_idx = 0;
          }
        } else if (c == '\b' || c == 0x7F) {
          if (s_ascii_idx > 0) {
            s_ascii_idx--;
          }
        } else if (c >= 32 && s_ascii_idx < sizeof(g_cmd_buf) - 1) {
          g_cmd_buf[s_ascii_idx++] = c;
        }
      }
    } else {
      s_bin_buf[s_bin_idx++] = b;
      if (s_bin_idx >= sizeof(PointReqPacket)) {
        memcpy(&g_proto_req, s_bin_buf, sizeof(PointReqPacket));
        g_proto_ready = 1;
        s_bin_active = 0;
        s_bin_idx = 0;
      }
    }
  }

  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  return (USBD_OK);
}

uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len)
{
  uint8_t result = USBD_OK;
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  if (hcdc == NULL || hcdc->TxState != 0){
    return USBD_BUSY;
  }
  SCB_CleanDCache_by_Addr((uint32_t *)Buf, (Len + 31) & ~31);
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS);
  return result;
}

void CDC_SendResponse(const char *msg)
{
  if (!msg) return;
  uint16_t len = (uint16_t)strlen(msg);
  uint32_t t0 = HAL_GetTick();
  while (CDC_Transmit_FS((uint8_t*)msg, len) == USBD_BUSY) {
    if (HAL_GetTick() - t0 > 1000) break;
  }
}

uint8_t CDC_SendBinary(const uint8_t *data, uint16_t len)
{
  if (!data || len == 0) return USBD_OK;
  uint32_t t0 = HAL_GetTick();
  while (CDC_Transmit_FS((uint8_t*)data, len) == USBD_BUSY) {
    if (HAL_GetTick() - t0 > 2000) return USBD_BUSY;
  }
  return USBD_OK;
}

static int8_t CDC_TransmitCplt_FS(uint8_t *Buf, uint32_t *Len, uint8_t epnum)
{
  UNUSED(Buf);
  UNUSED(Len);
  UNUSED(epnum);
  return (USBD_OK);
}

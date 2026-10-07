#ifndef __NEURO_PROTOCOL_H__
#define __NEURO_PROTOCOL_H__

#include <stdint.h>

#define PROTO_REQ_SYNC0  0xAA
#define PROTO_REQ_SYNC1  0x55
#define PROTO_RESP_SYNC0 0x55
#define PROTO_RESP_SYNC1 0xAA

#define CMD_INVERT_POINT 0x01

#define STATUS_OK        0x00
#define STATUS_CRC_ERR   0x01
#define STATUS_NOT_INIT  0x02
#define STATUS_MATH_ERR  0x03

#pragma pack(push, 1)
typedef struct {
  uint8_t  sync[2];       /* 0xAA, 0x55 */
  uint8_t  cmd;           /* 0x01 */
  uint8_t  reserved;      /* 0x00 for 4-byte alignment */
  uint32_t point_id;      /* Point ID */
  float    signals[40];   /* 40 raw probe signals (4-byte aligned) */
  uint16_t crc;           /* CRC16 of cmd..signals (166 bytes) */
} PointReqPacket;

typedef struct {
  uint8_t  sync[2];       /* 0x55, 0xAA */
  uint8_t  status;        /* 0=OK, 1=CRC Error, 2=Not Init */
  uint8_t  reserved;      /* 0x00 for 4-byte alignment */
  uint32_t point_id;      /* Echo point ID */
  uint32_t elapsed_ms;    /* Time taken in ms */
  float    geo_params[14];/* Inverted geometry params (4-byte aligned) */
  uint16_t crc;           /* CRC16 of status..geo_params (66 bytes) */
  uint16_t pad;           /* Padding to 72 bytes */
} PointRespPacket;
#pragma pack(pop)

uint16_t CRC16(const uint8_t *nData, uint16_t wLength);
int      signum(float val);

#endif /* __NEURO_PROTOCOL_H__ */

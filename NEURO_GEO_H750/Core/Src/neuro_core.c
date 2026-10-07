#include "neuro_core.h"
#include "fatfs.h"
#include "w25qxx_qspi.h"
#include "main.h"
#include "usbd_cdc_if.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

static const float D_MAX = 4.0f;
static const float NOISE_THRESHOLD = 0.045f;
static const int AMP_INDICES[16] = { 4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38 };

float g_mean_X[7];
float g_scale_X[7];
float g_mean_Y[40];
float g_scale_Y[40];
bool g_neuro_ready = false;
bool g_is_fp16 = false;
bool g_qspi_dtr = false;
uint32_t g_dtr_test_magic = 0;

static const float  *FWD_W32[5];
static const __fp16 *FWD_W16[5];
static const float  *FWD_b[5];

/* Fast high-precision polynomial GELU & derivative (Abramowitz & Stegun 7.1.26, max error < 6e-7) */
static inline float gelu(float x) {
  if (x > 5.0f) return x;
  if (x < -5.0f) return 0.0f;

  float ax = fabsf(x);
  float z = ax * 0.7071067811865475f; /* x / sqrt(2) */
  float t = 1.0f / (1.0f + 0.3275911f * z);
  float poly = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
  float ex = expf(-z * z);
  float erf_z = 1.0f - poly * ex;
  if (x < 0.0f) erf_z = -erf_z;

  return 0.5f * x * (1.0f + erf_z);
}

static inline float gelu_deriv(float x) {
  if (x > 5.0f) return 1.0f;
  if (x < -5.0f) return 0.0f;

  float ax = fabsf(x);
  float z = ax * 0.7071067811865475f;
  float t = 1.0f / (1.0f + 0.3275911f * z);
  float poly = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
  float ex = expf(-z * z); /* exp(-0.5 * x * x) */
  float erf_z = 1.0f - poly * ex;
  if (x < 0.0f) erf_z = -erf_z;

  float phi = 0.5f * (1.0f + erf_z);
  return phi + (x * ex) * 0.3989422804014327f; /* 1 / sqrt(2*pi) */
}

static void init_fwd_pointers(const uint8_t *base) {
  uint32_t magic = *(const uint32_t *)base;
  const uint8_t *p = base + 64;
  int in_dim[5] = {7, 512, 1024, 512, 256};
  int out_dim[5] = {512, 1024, 512, 256, 40};

  if (magic == FWD16_MAGIC) {
    g_is_fp16 = true;
    for (int i = 0; i < 5; ++i) {
      FWD_W16[i] = (const __fp16 *)p;
      p += in_dim[i] * out_dim[i] * sizeof(__fp16);
      FWD_b[i] = (const float *)p;
      p += out_dim[i] * sizeof(float);
    }
  } else {
    g_is_fp16 = false;
    for (int i = 0; i < 5; ++i) {
      FWD_W32[i] = (const float *)p;
      p += in_dim[i] * out_dim[i] * sizeof(float);
      FWD_b[i] = (const float *)p;
      p += out_dim[i] * sizeof(float);
    }
  }
}

/* Activation state for FWD */
typedef struct {
  float z0[512], a0[512];
  float z1[1024], a1[1024];
  float z2[512], a2[512];
  float z3[256], a3[256];
  float z4[40];
} FwdState;

static FwdState s_fwd;
static float s_da[1024];
static float s_dz[1024];

static void forward_fp32(const float *x, float *out_y)
{
  FwdState *s = &s_fwd;

  /* Layer 0: 7 -> 512 (contiguous linear streaming) */
  memcpy(s->z0, FWD_b[0], 512 * sizeof(float));
  for (int k = 0; k < 7; ++k) {
    float x_k = x[k];
    const float *w_row = &FWD_W32[0][k * 512];
    for (int j = 0; j < 512; ++j) {
      s->z0[j] += x_k * w_row[j];
    }
  }
  for (int j = 0; j < 512; ++j) {
    s->a0[j] = gelu(s->z0[j]);
  }

  /* Layer 1: 512 -> 1024 (contiguous linear streaming) */
  memcpy(s->z1, FWD_b[1], 1024 * sizeof(float));
  for (int k = 0; k < 512; ++k) {
    float a_k = s->a0[k];
    const float *w_row = &FWD_W32[1][k * 1024];
    for (int j = 0; j < 1024; ++j) {
      s->z1[j] += a_k * w_row[j];
    }
  }
  for (int j = 0; j < 1024; ++j) {
    s->a1[j] = gelu(s->z1[j]);
  }

  /* Layer 2: 1024 -> 512 (contiguous linear streaming) */
  memcpy(s->z2, FWD_b[2], 512 * sizeof(float));
  for (int k = 0; k < 1024; ++k) {
    float a_k = s->a1[k];
    const float *w_row = &FWD_W32[2][k * 512];
    for (int j = 0; j < 512; ++j) {
      s->z2[j] += a_k * w_row[j];
    }
  }
  for (int j = 0; j < 512; ++j) {
    s->a2[j] = gelu(s->z2[j]);
  }

  /* Layer 3: 512 -> 256 (contiguous linear streaming) */
  memcpy(s->z3, FWD_b[3], 256 * sizeof(float));
  for (int k = 0; k < 512; ++k) {
    float a_k = s->a2[k];
    const float *w_row = &FWD_W32[3][k * 256];
    for (int j = 0; j < 256; ++j) {
      s->z3[j] += a_k * w_row[j];
    }
  }
  for (int j = 0; j < 256; ++j) {
    s->a3[j] = gelu(s->z3[j]);
  }

  /* Layer 4: 256 -> 40 (linear) */
  memcpy(s->z4, FWD_b[4], 40 * sizeof(float));
  for (int k = 0; k < 256; ++k) {
    float a_k = s->a3[k];
    const float *w_row = &FWD_W32[4][k * 40];
    for (int j = 0; j < 40; ++j) {
      s->z4[j] += a_k * w_row[j];
    }
  }
  memcpy(out_y, s->z4, 40 * sizeof(float));
}

static void forward_fp16(const float *x, float *out_y)
{
  FwdState *s = &s_fwd;

  /* Layer 0: 7 -> 512 (contiguous linear streaming, HW vcvtb) */
  memcpy(s->z0, FWD_b[0], 512 * sizeof(float));
  for (int k = 0; k < 7; ++k) {
    float x_k = x[k];
    const __fp16 *w_row = &FWD_W16[0][k * 512];
    for (int j = 0; j < 512; ++j) {
      s->z0[j] += x_k * (float)w_row[j];
    }
  }
  for (int j = 0; j < 512; ++j) {
    s->a0[j] = gelu(s->z0[j]);
  }

  /* Layer 1: 512 -> 1024 (contiguous linear streaming, HW vcvtb) */
  memcpy(s->z1, FWD_b[1], 1024 * sizeof(float));
  for (int k = 0; k < 512; ++k) {
    float a_k = s->a0[k];
    const __fp16 *w_row = &FWD_W16[1][k * 1024];
    for (int j = 0; j < 1024; ++j) {
      s->z1[j] += a_k * (float)w_row[j];
    }
  }
  for (int j = 0; j < 1024; ++j) {
    s->a1[j] = gelu(s->z1[j]);
  }

  /* Layer 2: 1024 -> 512 (contiguous linear streaming, HW vcvtb) */
  memcpy(s->z2, FWD_b[2], 512 * sizeof(float));
  for (int k = 0; k < 1024; ++k) {
    float a_k = s->a1[k];
    const __fp16 *w_row = &FWD_W16[2][k * 512];
    for (int j = 0; j < 512; ++j) {
      s->z2[j] += a_k * (float)w_row[j];
    }
  }
  for (int j = 0; j < 512; ++j) {
    s->a2[j] = gelu(s->z2[j]);
  }

  /* Layer 3: 512 -> 256 (contiguous linear streaming, HW vcvtb) */
  memcpy(s->z3, FWD_b[3], 256 * sizeof(float));
  for (int k = 0; k < 512; ++k) {
    float a_k = s->a2[k];
    const __fp16 *w_row = &FWD_W16[3][k * 256];
    for (int j = 0; j < 256; ++j) {
      s->z3[j] += a_k * (float)w_row[j];
    }
  }
  for (int j = 0; j < 256; ++j) {
    s->a3[j] = gelu(s->z3[j]);
  }

  /* Layer 4: 256 -> 40 (linear, HW vcvtb) */
  memcpy(s->z4, FWD_b[4], 40 * sizeof(float));
  for (int k = 0; k < 256; ++k) {
    float a_k = s->a3[k];
    const __fp16 *w_row = &FWD_W16[4][k * 40];
    for (int j = 0; j < 40; ++j) {
      s->z4[j] += a_k * (float)w_row[j];
    }
  }
  memcpy(out_y, s->z4, 40 * sizeof(float));
}

void Neuro_PredictForward(const float *x, float *out_y)
{
  if (g_is_fp16) {
    forward_fp16(x, out_y);
  } else {
    forward_fp32(x, out_y);
  }
}

static void backward_fp32(const float *y_pred, const float *y_true, float *grad_x)
{
  FwdState *s = &s_fwd;
  float dy[40];

  for (int i = 0; i < 40; ++i) {
    dy[i] = (2.0f / 40.0f) * (y_pred[i] - y_true[i]);
  }

  /* Layer 4 backward: dy (40) * W4^T (40 x 256) -> da3 (256) */
  for (int k = 0; k < 256; ++k) {
    float sum = 0.0f;
    for (int j = 0; j < 40; ++j) {
      sum += dy[j] * FWD_W32[4][k * 40 + j];
    }
    s_da[k] = sum;
  }

  /* Layer 3 backward: da3 (256) * deriv(z3) -> dz3 (256), dz3 * W3^T -> da2 (512) */
  for (int j = 0; j < 256; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z3[j]);
  }
  for (int k = 0; k < 512; ++k) {
    float sum = 0.0f;
    for (int j = 0; j < 256; ++j) {
      sum += s_dz[j] * FWD_W32[3][k * 256 + j];
    }
    s_da[k] = sum;
  }

  /* Layer 2 backward: da2 (512) * deriv(z2) -> dz2 (512), dz2 * W2^T -> da1 (1024) */
  for (int j = 0; j < 512; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z2[j]);
  }
  for (int k = 0; k < 1024; ++k) {
    float sum = 0.0f;
    for (int j = 0; j < 512; ++j) {
      sum += s_dz[j] * FWD_W32[2][k * 512 + j];
    }
    s_da[k] = sum;
  }

  /* Layer 1 backward: da1 (1024) * deriv(z1) -> dz1 (1024), dz1 * W1^T -> da0 (512) */
  for (int j = 0; j < 1024; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z1[j]);
  }
  for (int k = 0; k < 512; ++k) {
    float sum = 0.0f;
    for (int j = 0; j < 1024; ++j) {
      sum += s_dz[j] * FWD_W32[1][k * 1024 + j];
    }
    s_da[k] = sum;
  }

  /* Layer 0 backward: da0 (512) * deriv(z0) -> dz0 (512), dz0 * W0^T -> grad_x (7) */
  for (int j = 0; j < 512; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z0[j]);
  }
  for (int k = 0; k < 7; ++k) {
    float sum = 0.0f;
    for (int j = 0; j < 512; ++j) {
      sum += s_dz[j] * FWD_W32[0][k * 512 + j];
    }
    grad_x[k] = sum;
  }
}

static void backward_fp16(const float *y_pred, const float *y_true, float *grad_x)
{
  FwdState *s = &s_fwd;
  float dy[40];

  for (int i = 0; i < 40; ++i) {
    dy[i] = (2.0f / 40.0f) * (y_pred[i] - y_true[i]);
  }

  /* Layer 4 backward: dy (40) * W4^T (40 x 256) -> da3 (256) */
  for (int k = 0; k < 256; ++k) {
    float sum = 0.0f;
    const __fp16 *w_row = &FWD_W16[4][k * 40];
    for (int j = 0; j < 40; ++j) {
      sum += dy[j] * (float)w_row[j];
    }
    s_da[k] = sum;
  }

  /* Layer 3 backward: da3 (256) * deriv(z3) -> dz3 (256), dz3 * W3^T -> da2 (512) */
  for (int j = 0; j < 256; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z3[j]);
  }
  for (int k = 0; k < 512; ++k) {
    float sum = 0.0f;
    const __fp16 *w_row = &FWD_W16[3][k * 256];
    for (int j = 0; j < 256; ++j) {
      sum += s_dz[j] * (float)w_row[j];
    }
    s_da[k] = sum;
  }

  /* Layer 2 backward: da2 (512) * deriv(z2) -> dz2 (512), dz2 * W2^T -> da1 (1024) */
  for (int j = 0; j < 512; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z2[j]);
  }
  for (int k = 0; k < 1024; ++k) {
    float sum = 0.0f;
    const __fp16 *w_row = &FWD_W16[2][k * 512];
    for (int j = 0; j < 512; ++j) {
      sum += s_dz[j] * (float)w_row[j];
    }
    s_da[k] = sum;
  }

  /* Layer 1 backward: da1 (1024) * deriv(z1) -> dz1 (1024), dz1 * W1^T -> da0 (512) */
  for (int j = 0; j < 1024; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z1[j]);
  }
  for (int k = 0; k < 512; ++k) {
    float sum = 0.0f;
    const __fp16 *w_row = &FWD_W16[1][k * 1024];
    for (int j = 0; j < 1024; ++j) {
      sum += s_dz[j] * (float)w_row[j];
    }
    s_da[k] = sum;
  }

  /* Layer 0 backward: da0 (512) * deriv(z0) -> dz0 (512), dz0 * W0^T -> grad_x (7) */
  for (int j = 0; j < 512; ++j) {
    s_dz[j] = s_da[j] * gelu_deriv(s->z0[j]);
  }
  for (int k = 0; k < 7; ++k) {
    float sum = 0.0f;
    const __fp16 *w_row = &FWD_W16[0][k * 512];
    for (int j = 0; j < 512; ++j) {
      sum += s_dz[j] * (float)w_row[j];
    }
    grad_x[k] = sum;
  }
}

static inline void forward_net_backward(const float *y_pred, const float *y_true, float *grad_x)
{
  if (g_is_fp16) {
    backward_fp16(y_pred, y_true, grad_x);
  } else {
    backward_fp32(y_pred, y_true, grad_x);
  }
}

/* SRAM streaming buffer for Inverse network weights */
#define CHUNK_FLOATS 32768  /* 128 KB buffer */
static float s_chunk[CHUNK_FLOATS];

static float inv_act_prev[1024];
static float inv_act_next[1024];

uint8_t Neuro_PredictInverseFromSD(const float *y_scaled_40, float *opt_geo_out_7)
{
  FRESULT res;
  UINT bytes_read;
  FIL inv_file;

  res = f_open(&inv_file, "INV_FP32.BIN", FA_READ);
  if (res != FR_OK) return NEURO_ERROR;

  /* Skip 64-byte header */
  f_lseek(&inv_file, 64);

  /* --- Layer 0: 40 -> 512 --- */
  /* Read W0: 40 * 512 = 20,480 floats */
  f_read(&inv_file, s_chunk, 40 * 512 * 4, &bytes_read);
  /* Read b0: 512 floats */
  f_read(&inv_file, inv_act_next, 512 * 4, &bytes_read);

  for (int j = 0; j < 512; ++j) {
    float sum = inv_act_next[j];
    for (int k = 0; k < 40; ++k) {
      sum += y_scaled_40[k] * s_chunk[k * 512 + j];
    }
    inv_act_prev[j] = gelu(sum);
  }

  /* --- Layer 1: 512 -> 1024 --- */
  /* Read W1 (512 x 1024 = 524,288 floats) in chunks of 16 rows = 16,384 floats */
  /* b1: 1024 floats is at offset: 64 + (40*512 + 512 + 512*1024)*4 */
  DWORD w1_offset = f_tell(&inv_file);
  DWORD b1_offset = w1_offset + 512 * 1024 * 4;

  /* Read b1 first into inv_act_next */
  f_lseek(&inv_file, b1_offset);
  f_read(&inv_file, inv_act_next, 1024 * 4, &bytes_read);

  /* Stream W1 in 32 chunks of 16 rows (16 * 1024 = 16,384 floats = 64 KB) */
  f_lseek(&inv_file, w1_offset);
  for (int chunk = 0; chunk < 32; ++chunk) {
    f_read(&inv_file, s_chunk, 16 * 1024 * 4, &bytes_read);
    int row_start = chunk * 16;
    for (int r = 0; r < 16; ++r) {
      float a_val = inv_act_prev[row_start + r];
      const float *w_row = &s_chunk[r * 1024];
      for (int j = 0; j < 1024; ++j) {
        inv_act_next[j] += a_val * w_row[j];
      }
    }
  }
  for (int j = 0; j < 1024; ++j) {
    inv_act_prev[j] = gelu(inv_act_next[j]);
  }
  /* Skip past b1 */
  f_lseek(&inv_file, b1_offset + 1024 * 4);

  /* --- Layer 2: 1024 -> 512 --- */
  DWORD w2_offset = f_tell(&inv_file);
  DWORD b2_offset = w2_offset + 1024 * 512 * 4;

  /* Read b2 first into inv_act_next */
  f_lseek(&inv_file, b2_offset);
  f_read(&inv_file, inv_act_next, 512 * 4, &bytes_read);

  /* Stream W2 in 32 chunks of 32 rows (32 * 512 = 16,384 floats = 64 KB) */
  f_lseek(&inv_file, w2_offset);
  for (int chunk = 0; chunk < 32; ++chunk) {
    f_read(&inv_file, s_chunk, 32 * 512 * 4, &bytes_read);
    int row_start = chunk * 32;
    for (int r = 0; r < 32; ++r) {
      float a_val = inv_act_prev[row_start + r];
      const float *w_row = &s_chunk[r * 512];
      for (int j = 0; j < 512; ++j) {
        inv_act_next[j] += a_val * w_row[j];
      }
    }
  }
  for (int j = 0; j < 512; ++j) {
    inv_act_prev[j] = gelu(inv_act_next[j]);
  }
  f_lseek(&inv_file, b2_offset + 512 * 4);

  /* --- Layer 3: 512 -> 256 --- */
  DWORD w3_offset = f_tell(&inv_file);
  DWORD b3_offset = w3_offset + 512 * 256 * 4;

  f_lseek(&inv_file, b3_offset);
  f_read(&inv_file, inv_act_next, 256 * 4, &bytes_read);

  /* Stream W3 in 8 chunks of 64 rows (64 * 256 = 16,384 floats = 64 KB) */
  f_lseek(&inv_file, w3_offset);
  for (int chunk = 0; chunk < 8; ++chunk) {
    f_read(&inv_file, s_chunk, 64 * 256 * 4, &bytes_read);
    int row_start = chunk * 64;
    for (int r = 0; r < 64; ++r) {
      float a_val = inv_act_prev[row_start + r];
      const float *w_row = &s_chunk[r * 256];
      for (int j = 0; j < 256; ++j) {
        inv_act_next[j] += a_val * w_row[j];
      }
    }
  }
  for (int j = 0; j < 256; ++j) {
    inv_act_prev[j] = gelu(inv_act_next[j]);
  }
  f_lseek(&inv_file, b3_offset + 256 * 4);

  /* --- Layer 4: 256 -> 7 (linear) --- */
  /* Read W4 (256 x 7 = 1792 floats) */
  f_read(&inv_file, s_chunk, 256 * 7 * 4, &bytes_read);
  /* Read b4 (7 floats) directly into opt_geo_out_7 */
  f_read(&inv_file, opt_geo_out_7, 7 * 4, &bytes_read);

  for (int j = 0; j < 7; ++j) {
    float sum = opt_geo_out_7[j];
    for (int k = 0; k < 256; ++k) {
      sum += inv_act_prev[k] * s_chunk[k * 7 + j];
    }
    opt_geo_out_7[j] = sum;
  }

  f_close(&inv_file);
  return NEURO_OK;
}

static void pgd_core(const float *y_true_scaled,
                     const float *min_scaled,
                     const float *max_scaled,
                     int topology_mode,
                     int steps,
                     const float *m1_transformed,
                     const float *opt_geo_init,
                     float *out_transformed)
{
  float opt_geo[7];
  memcpy(opt_geo, opt_geo_init, 7 * sizeof(float));

  /* Initial bounds clip */
  for (int i = 0; i < 7; ++i) {
    if (opt_geo[i] < min_scaled[i]) opt_geo[i] = min_scaled[i];
    if (opt_geo[i] > max_scaled[i]) opt_geo[i] = max_scaled[i];
  }

  /* Noise check */
  bool active = true;
  if (topology_mode >= 0) {
    float amp_mean = 0.0f;
    for (int i = 0; i < 16; ++i) amp_mean += y_true_scaled[AMP_INDICES[i]];
    amp_mean /= 16.0f;
    float sq = 0.0f;
    for (int i = 0; i < 16; ++i) {
      float d = y_true_scaled[AMP_INDICES[i]] - amp_mean;
      sq += d * d;
    }
    float amp_std = sqrtf(sq / 15.0f);
    active = (amp_std > NOISE_THRESHOLD);
  }

  bool up_closer = false;
  float m1_scaled[7];
  if (topology_mode == 2 && m1_transformed != NULL) {
    for (int i = 0; i < 7; ++i) {
      m1_scaled[i] = (m1_transformed[i] - g_mean_X[i]) / g_scale_X[i];
    }
    up_closer = (m1_transformed[4] <= m1_transformed[5]);
    if (up_closer) {
      opt_geo[4] = m1_scaled[4];
      opt_geo[0] = m1_scaled[0];
    } else {
      opt_geo[5] = m1_scaled[5];
      opt_geo[3] = m1_scaled[3];
    }
  }

  float y_pred[40];
  float grad_x[7];
  float m[7] = {0}, v[7] = {0};
  const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f, lr = 0.015f;

  for (int step = 1; step <= steps; ++step) {
    if (step % 25 == 0 || step == steps) {
      HAL_GPIO_TogglePin(PE3_GPIO_Port, PE3_Pin);
    }
    Neuro_PredictForward(opt_geo, y_pred);
    forward_net_backward(y_pred, y_true_scaled, grad_x);

    if (topology_mode >= 0 && !active) {
      grad_x[4] = 0.0f;
      grad_x[5] = 0.0f;
    }
    if (topology_mode == 2 && m1_transformed != NULL) {
      if (up_closer) {
        grad_x[4] = 0.0f;
        grad_x[0] = 0.0f;
      } else {
        grad_x[5] = 0.0f;
        grad_x[3] = 0.0f;
      }
    }

    float p1 = 1.0f - powf(beta1, (float)step);
    float p2 = 1.0f - powf(beta2, (float)step);

    for (int i = 0; i < 7; ++i) {
      m[i] = beta1 * m[i] + (1.0f - beta1) * grad_x[i];
      v[i] = beta2 * v[i] + (1.0f - beta2) * (grad_x[i] * grad_x[i]);
      float m_hat = m[i] / p1;
      float v_hat = v[i] / p2;
      opt_geo[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);

      /* Clip to scaled bounds */
      if (opt_geo[i] < min_scaled[i]) opt_geo[i] = min_scaled[i];
      if (opt_geo[i] > max_scaled[i]) opt_geo[i] = max_scaled[i];
    }

    if (topology_mode >= 0) {
      float geo_real[7];
      for (int i = 0; i < 7; ++i) {
        geo_real[i] = opt_geo[i] * g_scale_X[i] + g_mean_X[i];
      }

      if (topology_mode == 0) {
        geo_real[4] = D_MAX;
        geo_real[5] = D_MAX;
        geo_real[0] = geo_real[1];
        geo_real[3] = geo_real[1];
      } else if (topology_mode == 1) {
        if (geo_real[4] <= geo_real[5]) {
          geo_real[5] = D_MAX;
          geo_real[3] = geo_real[1];
        } else {
          geo_real[4] = D_MAX;
          geo_real[0] = geo_real[1];
        }
      } else if (topology_mode == 2 && m1_transformed != NULL) {
        if (up_closer) {
          geo_real[4] = m1_transformed[4];
          geo_real[0] = m1_transformed[0];
        } else {
          geo_real[5] = m1_transformed[5];
          geo_real[3] = m1_transformed[3];
        }
      }

      for (int i = 0; i < 7; ++i) {
        opt_geo[i] = (geo_real[i] - g_mean_X[i]) / g_scale_X[i];
        if (opt_geo[i] < min_scaled[i]) opt_geo[i] = min_scaled[i];
        if (opt_geo[i] > max_scaled[i]) opt_geo[i] = max_scaled[i];
      }
    }
  }

  for (int i = 0; i < 7; ++i) {
    out_transformed[i] = opt_geo[i] * g_scale_X[i] + g_mean_X[i];
  }
}

void Neuro_RunTopologyInversion(const float *raw_signals_40,
                                const float *min_bounds_7,
                                const float *max_bounds_7,
                                float *out14,
                                bool verbose)
{
  float y_true[40];
  for (int i = 0; i < 40; ++i) y_true[i] = raw_signals_40[i];
  for (int i = 0; i < 16; ++i) {
    int idx = AMP_INDICES[i];
    y_true[idx] = log10f(fmaxf(y_true[idx], 1e-8f));
  }
  for (int i = 0; i < 40; ++i) {
    y_true[i] = (y_true[i] - g_mean_Y[i]) / g_scale_Y[i];
  }

  float min_b[7], max_b[7], min_scaled[7], max_scaled[7];
  for (int i = 0; i < 7; ++i) {
    min_b[i] = min_bounds_7[i];
    max_b[i] = max_bounds_7[i];
  }
  for (int i = 0; i < 4; ++i) {
    min_b[i] = log10f(fmaxf(min_b[i], 1e-5f));
    max_b[i] = log10f(fmaxf(max_b[i], 1e-5f));
  }
  for (int i = 0; i < 7; ++i) {
    min_scaled[i] = (min_b[i] - g_mean_X[i]) / g_scale_X[i];
    max_scaled[i] = (max_b[i] - g_mean_X[i]) / g_scale_X[i];
  }

  float opt_geo_init[7];
  Neuro_PredictInverseFromSD(y_true, opt_geo_init);

  float res0[7], res1[7], res2[7];
  if (verbose) CDC_SendResponse("  [INVERT] Running Model 0 (0 boundaries, 25 steps)...\r\n");
  pgd_core(y_true, min_scaled, max_scaled, 0, ADAM_STEPS_M0, NULL, opt_geo_init, res0);
  if (verbose) CDC_SendResponse("  [INVERT] Running Model 1 (1 boundary, 40 steps)...\r\n");
  pgd_core(y_true, min_scaled, max_scaled, 1, ADAM_STEPS_M1, NULL, opt_geo_init, res1);
  if (verbose) CDC_SendResponse("  [INVERT] Running Model 2 (2 boundaries, 40 steps)...\r\n");
  pgd_core(y_true, min_scaled, max_scaled, 2, ADAM_STEPS_M2, res1, opt_geo_init, res2);

  out14[0]  = powf(10.0f, res0[1]);
  out14[1]  = powf(10.0f, res0[2]);
  out14[2]  = powf(10.0f, res1[1]);
  out14[3]  = powf(10.0f, res1[2]);
  out14[4]  = powf(10.0f, res1[0]);
  out14[5]  = powf(10.0f, res1[3]);
  out14[6]  = res1[4];
  out14[7]  = res1[5];
  out14[8]  = powf(10.0f, res2[1]);
  out14[9]  = powf(10.0f, res2[2]);
  out14[10] = powf(10.0f, res2[0]);
  out14[11] = powf(10.0f, res2[3]);
  out14[12] = res2[4];
  out14[13] = res2[5];
}

uint8_t Neuro_Init(void)
{
  FRESULT res;
  UINT bytes_read;
  FIL f;

  /* 1. Read SCALERS.BIN from SD */
  res = f_open(&f, "SCALERS.BIN", FA_READ);
  if (res != FR_OK) return NEURO_ERROR;

  f_lseek(&f, 64); /* skip header */
  f_read(&f, g_mean_X, 7 * 4, &bytes_read);
  f_read(&f, g_scale_X, 7 * 4, &bytes_read);
  f_read(&f, g_mean_Y, 40 * 4, &bytes_read);
  f_read(&f, g_scale_Y, 40 * 4, &bytes_read);
  f_close(&f);

  /* 2. Check if QSPI is in memory mapped mode and has FWD magic */
  w25qxx_Init();

  /* Try DTR Mode with candidate dummy cycle values (6, 8, 4) */
  uint32_t magic = 0;
  g_qspi_dtr = false;
  static const uint8_t dtr_candidates[] = {6, 8, 4};
  for (int i = 0; i < 3; ++i) {
    w25qxx_Init();
    uint8_t dtr_res = w25qxx_StartupDTR(dtr_candidates[i]);
    SCB_CleanInvalidateDCache();
    magic = *(volatile uint32_t *)QSPI_BASE_ADDR;
    g_dtr_test_magic = magic;
    if (dtr_res == w25qxx_OK && (magic == FWD16_MAGIC || magic == FWD_MAGIC)) {
      const uint32_t *hdr = (const uint32_t *)QSPI_BASE_ADDR;
      if (hdr[1] == 5 && hdr[2] == 7 && hdr[3] == 512 && hdr[4] == 1024) {
        g_qspi_dtr = true;
        break;
      }
    }
  }

  if (!g_qspi_dtr) {
    /* Fallback to reliable Normal Mode (SDR, 60 MB/s) */
    w25qxx_Init();
    w25qxx_Startup(w25qxx_NormalMode);
    SCB_CleanInvalidateDCache();
    magic = *(volatile uint32_t *)QSPI_BASE_ADDR;
  }

  if (magic != FWD16_MAGIC && magic != FWD_MAGIC) {
    /* Need to burn FWD to QSPI */
    if (Neuro_BurnFwdToQspi() != NEURO_OK) {
      return NEURO_ERROR;
    }
  }

  SCB_CleanInvalidateDCache();
  init_fwd_pointers((const uint8_t *)QSPI_BASE_ADDR);
  g_neuro_ready = true;
  return NEURO_OK;
}

uint8_t Neuro_BurnFwdToQspi(void)
{
  FRESULT res;
  UINT bytes_read;
  FIL f;
  char msg[80];
  const char *burn_file = "FWD_FP16.BIN";

  res = f_open(&f, burn_file, FA_READ);
  if (res != FR_OK) {
    burn_file = "FWD_FP32.BIN";
    res = f_open(&f, burn_file, FA_READ);
  }
  if (res != FR_OK) {
    CDC_SendResponse("[QSPI] Error: Cannot open FWD_FP16.BIN or FWD_FP32.BIN!\r\n");
    return NEURO_ERROR;
  }

  /* Initialize QSPI in indirect mode for programming */
  w25qxx_Init();

  uint32_t total_size = f_size(&f);
  uint32_t block_count = (total_size + 65535) / 65536;

  snprintf(msg, sizeof(msg), "[QSPI] Flashing %s (%lu KB)...\r\n", burn_file, total_size / 1024);
  CDC_SendResponse(msg);

  snprintf(msg, sizeof(msg), "[QSPI] Erasing %lu blocks (64KB each)...\r\n", block_count);
  CDC_SendResponse(msg);

  /* Erase 64KB blocks */
  for (uint32_t b = 0; b < block_count; ++b) {
    HAL_GPIO_TogglePin(PE3_GPIO_Port, PE3_Pin);
    W25qxx_EraseBlock(b * 65536);
    if ((b + 1) % 10 == 0 || b + 1 == block_count) {
      snprintf(msg, sizeof(msg), "  [QSPI] Erased block %lu/%lu (%lu%%)\r\n",
               b + 1, block_count, (b + 1) * 100 / block_count);
      CDC_SendResponse(msg);
    }
  }

  snprintf(msg, sizeof(msg), "[QSPI] Programming %lu KB...\r\n", total_size / 1024);
  CDC_SendResponse(msg);

  /* Program in 4096-byte sector chunks */
  uint8_t buf[4096];
  uint32_t write_addr = 0;
  uint32_t last_percent = 0;

  while (write_addr < total_size) {
    f_read(&f, buf, sizeof(buf), &bytes_read);
    if (bytes_read == 0) break;

    /* Write in 256-byte pages */
    for (uint32_t p = 0; p < bytes_read; p += 256) {
      uint32_t chunk = (bytes_read - p >= 256) ? 256 : (bytes_read - p);
      W25qxx_PageProgram(&buf[p], write_addr + p, chunk);
    }
    write_addr += bytes_read;

    HAL_GPIO_TogglePin(PE3_GPIO_Port, PE3_Pin);

    uint32_t percent = write_addr * 100 / total_size;
    if (percent >= last_percent + 15 || write_addr >= total_size) {
      last_percent = percent;
      snprintf(msg, sizeof(msg), "  [QSPI] Written %lu/%lu KB (%lu%%)\r\n",
               write_addr / 1024, total_size / 1024, percent);
      CDC_SendResponse(msg);
    }
  }
  f_close(&f);

  /* Switch back to Memory-Mapped Mode */
  if (g_qspi_dtr) {
    w25qxx_Startup(w25qxx_DTRMode);
  } else {
    w25qxx_Startup(w25qxx_NormalMode);
  }
  SCB_CleanInvalidateDCache();
  return NEURO_OK;
}

uint8_t Neuro_RunBenchmarkPoint(uint32_t point_idx, float *out14, uint32_t *time_ms_out)
{
  FRESULT res;
  UINT bytes_read;
  FIL f;
  float row[48];

  CDC_SendResponse("[INVERT] Opening SYNTHLOG.BIN...\r\n");
  res = f_open(&f, "SYNTHLOG.BIN", FA_READ);
  if (res != FR_OK) {
    CDC_SendResponse("[INVERT] Error: SYNTHLOG.BIN not found on SD card!\r\n");
    return NEURO_ERROR;
  }

  DWORD offset = 64 + point_idx * 48 * 4;
  res = f_lseek(&f, offset);
  if (res != FR_OK) {
    f_close(&f);
    CDC_SendResponse("[INVERT] Error: seek failed in SYNTHLOG.BIN!\r\n");
    return NEURO_ERROR;
  }

  res = f_read(&f, row, 48 * 4, &bytes_read);
  f_close(&f);
  if (bytes_read != 48 * 4) {
    CDC_SendResponse("[INVERT] Error: read failed in SYNTHLOG.BIN!\r\n");
    return NEURO_ERROR;
  }

  CDC_SendResponse("[INVERT] Point 0 loaded. Inverting 40 signals...\r\n");

  float min7[7] = { 1.0f, 1.0f, 1.0f, 1.0f, 0.05f, 0.05f, 60.0f };
  float max7[7] = { 1000.0f, 1000.0f, 1000.0f, 1000.0f, 4.0f, 4.0f, 120.0f };

  uint32_t t0 = HAL_GetTick();
  Neuro_RunTopologyInversion(&row[8], min7, max7, out14, true);
  *time_ms_out = HAL_GetTick() - t0;

  return NEURO_OK;
}

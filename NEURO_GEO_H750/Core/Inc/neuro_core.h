#ifndef __NEURO_CORE_H
#define __NEURO_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include <stdint.h>
#include <stdbool.h>

#define NEURO_OK      0
#define NEURO_ERROR   1

#define FWD_MAGIC     0x46574431  /* 'FWD1' */
#define FWD16_MAGIC   0x46573136  /* 'FW16' */
#define INV_MAGIC     0x494E5631  /* 'INV1' */
#define SCAL_MAGIC    0x5343414C  /* 'SCAL' */
#define SYN1_MAGIC    0x53594E31  /* 'SYN1' */

#define QSPI_BASE_ADDR 0x90000000U

/* Optimization steps per topology with Early Stopping */
#define ADAM_MAX_STEPS_M0         25
#define ADAM_MAX_STEPS_M1         40
#define ADAM_MAX_STEPS_M2         40

#define ADAM_MIN_STEPS_M0         10
#define ADAM_MIN_STEPS_M1         20
#define ADAM_MIN_STEPS_M2         20

#define ADAM_EARLY_STOP_PATIENCE  2
#define ADAM_GRAD_NORM_SQ_EPS     0.0001f  /* ||∇x|| < 0.010 -> ||∇x||^2 < 1e-4 */

/* Scaler vectors */
extern float g_mean_X[7];
extern float g_scale_X[7];
extern float g_mean_Y[40];
extern float g_scale_Y[40];
extern bool g_neuro_ready;
extern bool g_is_fp16;
extern bool g_qspi_dtr;
extern uint32_t g_dtr_test_magic;

/* Core lifecycle */
uint8_t Neuro_Init(void);
uint8_t Neuro_BurnFwdToQspi(void);

/* Forward and Inverse Networks */
uint8_t Neuro_PredictInverseFromSD(const float *y_scaled_40, float *opt_geo_out_7);
void    Neuro_PredictForward(const float *x_scaled_7, float *y_scaled_out_40);

/* Inversion algorithms */
void    Neuro_RunPGD(const float *raw_signals_40,
                     const float *min_bounds_7,
                     const float *max_bounds_7,
                     int topology_mode,
                     const float *m1_geo_7,
                     float *raw_geo_out_7);

void    Neuro_RunTopologyInversion(const float *raw_signals_40,
                                   const float *min_bounds_7,
                                   const float *max_bounds_7,
                                   float *topology_out_14,
                                   bool verbose);

/* Benchmark and Verification */
uint8_t Neuro_RunBenchmarkPoint(uint32_t point_idx, float *out14, uint32_t *time_ms_out);

#ifdef __cplusplus
}
#endif

#endif /* __NEURO_CORE_H */

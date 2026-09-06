"""Проверка тождества, на котором построена аналитическая аугментация в генераторе.

Для зеркальной модели (Rh_up<->Rh_dn, D_up<->D_dn, alpha->180-alpha):
    ZZ-каналы       не меняются
    амплитуды  |S| -> 1/|S|
    фазы       arg -> -arg

Скрипт считает физику для случайных LHS-точек и их зеркал и сравнивает с формулой.
Запуск:  python tools/verify_mirror_symmetry.py [N]
"""
import os
import sys

for _v in ['OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(_v, '1')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'))

import numpy as np
from GEO_40sig_7p_dataset_gen import (AMP_COLS, PH_COLS, calculate_point, mirror_params,
                                      mirror_signals, sample_parameters)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
X = sample_parameters(N, seed=7)
Xm = mirror_params(X)

Y = np.array([calculate_point(x) for x in X])
Y_mirror_physics = np.array([calculate_point(x) for x in Xm])
Y_mirror_formula = mirror_signals(Y)

zz = [i for i in range(40) if i not in set(AMP_COLS) | set(PH_COLS)]
print(f'точек: {N}')
print('max |ZZ_mirror - ZZ_orig|          =', np.abs(Y_mirror_physics[:, zz] - Y[:, zz]).max(), 'град')
print('max |Amp_mirror - 1/Amp_orig| отн. =',
      (np.abs(Y_mirror_physics[:, AMP_COLS] - 1 / Y[:, AMP_COLS]) * np.abs(Y[:, AMP_COLS])).max())
print('max |Phase_mirror + Phase_orig|    =',
      np.abs(Y_mirror_physics[:, PH_COLS] + Y[:, PH_COLS]).max(), 'град')
print('max |физика - формула| по всем 40  =', np.abs(Y_mirror_physics - Y_mirror_formula).max())

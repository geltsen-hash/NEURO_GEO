import numpy as np
from scipy.optimize import least_squares
import empymod

CALIBRATION_HEIGHTS = np.array([0.5, 0.8, 1.2, 1.5, 1.85], dtype=np.float64)
ROTATION_ANGLES_DEG = np.arange(16, dtype=np.float64) * 22.5
TRANSMITTERS_Z = {
    'Tzz1': -0.480,
    'Tzz2': 0.840,
    'Tzz3': -1.200,
    'Tzz4': 1.560,
    'Tzz5': 0.180,
}
RECEIVERS_Z = {
    'Rzx1': -1.440,
    'Rzx2': 1.800,
}
CHANNELS = [
    ('Tzz1', 'Rzx1'),
    ('Tzz2', 'Rzx1'),
    ('Tzz4', 'Rzx1'),
    ('Tzz5', 'Rzx1'),
    ('Tzz1', 'Rzx2'),
    ('Tzz2', 'Rzx2'),
    ('Tzz3', 'Rzx2'),
    ('Tzz5', 'Rzx2'),
]
TILT_NAMES = ['Tzz1', 'Tzz2', 'Tzz3', 'Tzz4', 'Tzz5', 'Rzx1', 'Rzx2']
XY_TILT_NAMES = [f'{name}_{axis}' for name in TILT_NAMES for axis in ('x', 'y')]
IMAGE_MOMENT_REFLECTION = np.diag([-1.0, -1.0, 1.0])
MEASURED_HARMONICS = None
MEASURED_HEIGHTS = None

EMPYMOD_FREQ = 2000000.0
EMPYMOD_RES_AIR = 2e14
EMPYMOD_RES_METAL = 1e-7
EMPYMOD_MU_METAL = 100.0
USE_EMPYMOD = True
H2_WEIGHT = 1.0

# Катушки: внешний диаметр 94 мм, сердечник 72 мм
COIL_DIAMETER = 0.094  # м
USE_FINITE_DIPOLES = False  # переключатель для тестов
USE_EMPYMOD_LOOP = False  # использовать empymod.loop вместо bipole

# Коррекция артефакта энкодера: H2_true = H2_meas - H2_ENCODER_BIAS * H1_meas
# Физически: эксцентриситет посадки энкодера даёт H2/H1 = const
H2_ENCODER_BIAS = 0.0  # установить ~0.060 для коррекции

# Эмпирическая поправка на отклонение H1/H0 от теории в зависимости от L:
# H1/H0_theor_corr = H1/H0_theor / f(L),  f(L) = FL_A * L^FL_B
# Подобрана совместно по Прибору1 и Прибору2, без ch5 (Tzz2-Rzx2)
FL_A = 2.2605
FL_B = 0.5863
USE_FL_CORRECTION = False  # включить для использования поправки f(L)

# Калиброванные углы приёмников (с поправкой f(L), без ch5):
# Прибор1: Rzx1=78.44 deg, Rzx2=77.28 deg
# Прибор2: Rzx1=79.03 deg, Rzx2=78.05 deg
# Без поправки f(L): Rzx1=59.09 deg, Rzx2=56.06 deg (Прибор2)


def rotation_xz_to_lab(phi_rad):
    c = np.cos(phi_rad)
    s = np.sin(phi_rad)
    return np.array(
        [
            [c, 0.0],
            [s, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_xyz_to_lab(phi_rad):
    c = np.cos(phi_rad)
    s = np.sin(phi_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def dipole_tensor_3d(r_vec):
    r = np.linalg.norm(r_vec)
    eye = np.eye(3, dtype=np.float64)
    outer = np.outer(r_vec, r_vec)
    return 3.0 * outer / r**5 - eye / r**3


def moment_xz_from_tilt(tilt_coeff):
    moment = np.array([tilt_coeff, 1.0], dtype=np.float64)
    return moment / np.linalg.norm(moment)


def moment_xyz_from_tilts(x_tilt, y_tilt):
    moment = np.array([x_tilt, y_tilt, 1.0], dtype=np.float64)
    return moment / np.linalg.norm(moment)


def tilts_from_vector(tilt_vector):
    return dict(zip(TILT_NAMES, tilt_vector))


def xy_tilts_from_vector(tilt_vector):
    return dict(zip(XY_TILT_NAMES, tilt_vector))


def vector_from_angles_deg(angles_deg):
    return np.array([np.tan(np.deg2rad(angles_deg[name])) for name in TILT_NAMES], dtype=np.float64)


def tilt_coeff_to_angle_deg(tilt_coeff):
    return np.rad2deg(np.arctan(tilt_coeff))


def channel_iq_at_rotation(height, tx_name, rx_name, tilt_coeffs, phi_deg):
    phi_rad = np.deg2rad(phi_deg)
    rot = rotation_xz_to_lab(phi_rad)
    tx_moment_lab = rot @ moment_xz_from_tilt(tilt_coeffs[tx_name])
    rx_moment_lab = rot @ moment_xz_from_tilt(tilt_coeffs[rx_name])
    tx_z = TRANSMITTERS_Z[tx_name]
    rx_z = RECEIVERS_Z[rx_name]
    direct_r = np.array([0.0, 0.0, rx_z - tx_z], dtype=np.float64)
    image_r = np.array([0.0, 2.0 * height, rx_z - tx_z], dtype=np.float64)
    direct_tensor = dipole_tensor_3d(direct_r)
    image_tensor = dipole_tensor_3d(image_r)
    image_tx_moment_lab = IMAGE_MOMENT_REFLECTION @ tx_moment_lab
    direct_signal = rx_moment_lab @ direct_tensor @ tx_moment_lab
    image_signal = rx_moment_lab @ image_tensor @ image_tx_moment_lab
    return direct_signal, image_signal


def xy_channel_iq_at_rotation(height, tx_name, rx_name, tilt_coeffs, phi_deg):
    phi_rad = np.deg2rad(phi_deg)
    rot = rotation_xyz_to_lab(phi_rad)
    tx_moment = moment_xyz_from_tilts(tilt_coeffs[f'{tx_name}_x'], tilt_coeffs[f'{tx_name}_y'])
    rx_moment = moment_xyz_from_tilts(tilt_coeffs[f'{rx_name}_x'], tilt_coeffs[f'{rx_name}_y'])
    tx_moment_lab = rot @ tx_moment
    rx_moment_lab = rot @ rx_moment
    tx_z = TRANSMITTERS_Z[tx_name]
    rx_z = RECEIVERS_Z[rx_name]
    direct_r = np.array([0.0, 0.0, rx_z - tx_z], dtype=np.float64)
    image_r = np.array([0.0, 2.0 * height, rx_z - tx_z], dtype=np.float64)
    direct_tensor = dipole_tensor_3d(direct_r)
    image_tensor = dipole_tensor_3d(image_r)
    image_tx_moment_lab = IMAGE_MOMENT_REFLECTION @ tx_moment_lab
    direct_signal = rx_moment_lab @ direct_tensor @ tx_moment_lab
    image_signal = rx_moment_lab @ image_tensor @ image_tx_moment_lab
    return direct_signal, image_signal


def instrument_harmonics_from_iq(re_samples, im_samples):
    harm_i = np.fft.fft(re_samples)[:3] * (2.0 / len(re_samples))
    harm_q = np.fft.fft(im_samples)[:3] * (2.0 / len(im_samples))
    harm_i[0] /= 2.0
    harm_q[0] /= 2.0
    vzz = np.real(harm_i[0]) + 1j * np.real(harm_q[0])
    vxx = np.abs(harm_i[2]) + 1j * np.abs(harm_q[2])
    fi_i = np.arctan2(np.imag(harm_i[1]), np.real(harm_i[1]))
    fi_q = np.arctan2(np.imag(harm_q[1]), np.real(harm_q[1]))
    weight_sum = np.abs(harm_i[1]) + np.abs(harm_q[1])
    if weight_sum > 0.0:
        fi = fi_i * np.abs(harm_i[1]) / weight_sum + fi_q * np.abs(harm_q[1]) / weight_sum
    else:
        fi = 0.0
    vzx = (
        np.cos(fi) * (np.real(harm_i[1]) + 1j * np.real(harm_q[1]))
        + np.sin(fi) * (np.imag(harm_i[1]) + 1j * np.imag(harm_q[1]))
    )
    return np.array([np.abs(vzz), np.abs(vzx), np.abs(vxx)], dtype=np.float64)


def _lab_moment_to_empymod_azimuth_dip(m_lab):
    mx_e = m_lab[2]
    my_e = m_lab[1]
    mz_e = m_lab[0]
    norm = np.sqrt(mx_e**2 + my_e**2 + mz_e**2)
    mx_e, my_e, mz_e = mx_e / norm, my_e / norm, mz_e / norm
    dip = np.rad2deg(np.arcsin(mz_e))
    azimuth = np.rad2deg(np.arctan2(my_e, mx_e))
    return azimuth, dip


def empymod_channel_iq_all_rotations(height, tx_name, rx_name, tilt_coeffs):
    tx_tilt = tilt_coeffs[tx_name]
    rx_tilt = tilt_coeffs[rx_name]
    tx_x = TRANSMITTERS_Z[tx_name]
    rx_x = RECEIVERS_Z[rx_name]
    N = len(ROTATION_ANGLES_DEG)
    
    if USE_FINITE_DIPOLES:
        # Конечные диполи: отрезки с наклоном в плоскости xz
        half_len = COIL_DIAMETER / 2.0
        
        # Углы наклона от вертикали
        tx_angle = np.arctan(tx_tilt)
        rx_angle = np.arctan(rx_tilt)
        
        # Передатчик: концы отрезка
        src_x0 = np.full(N, tx_x - half_len * np.cos(tx_angle))
        src_x1 = np.full(N, tx_x + half_len * np.cos(tx_angle))
        src_y0 = np.zeros(N)
        src_y1 = np.zeros(N)
        src_z0 = np.full(N, -half_len * np.sin(tx_angle))
        src_z1 = np.full(N, half_len * np.sin(tx_angle))
        
        # Приёмник: концы отрезка
        rec_x0 = np.full(N, rx_x - half_len * np.cos(rx_angle))
        rec_x1 = np.full(N, rx_x + half_len * np.cos(rx_angle))
        rec_y0 = np.zeros(N)
        rec_y1 = np.zeros(N)
        rec_z0 = np.full(N, -half_len * np.sin(rx_angle))
        rec_z1 = np.full(N, half_len * np.sin(rx_angle))
        
        src = [src_x0, src_x1, src_y0, src_y1, src_z0, src_z1]
        rec = [rec_x0, rec_x1, rec_y0, rec_y1, rec_z0, rec_z1]
        srcpts = 5
        recpts = 5
        msrc = False
        mrec = False
    else:
        # Точечные диполи с моментами
        tx_az = np.zeros(N)
        tx_dip = np.zeros(N)
        rx_az = np.zeros(N)
        rx_dip = np.zeros(N)
        for i, phi_deg in enumerate(ROTATION_ANGLES_DEG):
            rot = rotation_xz_to_lab(np.deg2rad(phi_deg))
            tx_az[i], tx_dip[i] = _lab_moment_to_empymod_azimuth_dip(rot @ moment_xz_from_tilt(tx_tilt))
            rx_az[i], rx_dip[i] = _lab_moment_to_empymod_azimuth_dip(rot @ moment_xz_from_tilt(rx_tilt))
        src = [np.full(N, tx_x), np.zeros(N), np.zeros(N), tx_az, tx_dip]
        rec = [np.full(N, rx_x), np.zeros(N), np.zeros(N), rx_az, rx_dip]
        srcpts = None
        recpts = None
        msrc = True
        mrec = True
    
    response = empymod.bipole(
        src=src,
        rec=rec,
        depth=[height],
        res=[EMPYMOD_RES_AIR, EMPYMOD_RES_METAL],
        freqtime=EMPYMOD_FREQ,
        mpermH=[1.0, EMPYMOD_MU_METAL],
        srcpts=srcpts,
        recpts=recpts,
        msrc=msrc,
        mrec=mrec,
        verb=0,
    )
    signals = np.diag(response)
    return signals.real, signals.imag


def empymod_loop_channel_iq_all_rotations(height, tx_name, rx_name, tilt_coeffs):
    """Альтернативная реализация через empymod.loop для магнитных катушек"""
    tx_tilt = tilt_coeffs[tx_name]
    rx_tilt = tilt_coeffs[rx_name]
    tx_x = TRANSMITTERS_Z[tx_name]
    rx_x = RECEIVERS_Z[rx_name]
    N = len(ROTATION_ANGLES_DEG)
    
    # Для loop: источник - электрическая петля (задаётся как bipole с electric=True)
    # Используем тот же синтаксис что и в bipole, но с electric=True
    if USE_FINITE_DIPOLES:
        # Конечные диполи: отрезки с наклоном в плоскости xz
        half_len = COIL_DIAMETER / 2.0
        
        # Углы наклона от вертикали
        tx_angle = np.arctan(tx_tilt)
        rx_angle = np.arctan(rx_tilt)
        
        # Передатчик: концы отрезка
        src_x0 = np.full(N, tx_x - half_len * np.cos(tx_angle))
        src_x1 = np.full(N, tx_x + half_len * np.cos(tx_angle))
        src_y0 = np.zeros(N)
        src_y1 = np.zeros(N)
        src_z0 = np.full(N, -half_len * np.sin(tx_angle))
        src_z1 = np.full(N, half_len * np.sin(tx_angle))
        
        # Приёмник: концы отрезка
        rec_x0 = np.full(N, rx_x - half_len * np.cos(rx_angle))
        rec_x1 = np.full(N, rx_x + half_len * np.cos(rx_angle))
        rec_y0 = np.zeros(N)
        rec_y1 = np.zeros(N)
        rec_z0 = np.full(N, -half_len * np.sin(rx_angle))
        rec_z1 = np.full(N, half_len * np.sin(rx_angle))
        
        src = [src_x0, src_x1, src_y0, src_y1, src_z0, src_z1]
        rec = [rec_x0, rec_x1, rec_y0, rec_y1, rec_z0, rec_z1]
        srcpts = 5
        recpts = 5
    else:
        # Точечные диполи с моментами
        src = [np.full(N, tx_x), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        rec = [np.full(N, rx_x), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]
        
        src_az = np.zeros(N)
        src_dip = np.zeros(N)
        rec_az = np.zeros(N)
        rec_dip = np.zeros(N)
        
        for i, phi_deg in enumerate(ROTATION_ANGLES_DEG):
            rot = rotation_xz_to_lab(np.deg2rad(phi_deg))
            src_moment = rot @ moment_xz_from_tilt(tx_tilt)
            src_az[i], src_dip[i] = _lab_moment_to_empymod_azimuth_dip(src_moment)
            rec_moment = rot @ moment_xz_from_tilt(rx_tilt)
            rec_az[i], rec_dip[i] = _lab_moment_to_empymod_azimuth_dip(rec_moment)
        
        src[3] = src_az
        src[4] = src_dip
        rec[3] = rec_az
        rec[4] = rec_dip
        srcpts = None
        recpts = None
    
    response = empymod.loop(
        src=src,
        rec=rec,
        depth=[height],
        res=[EMPYMOD_RES_AIR, EMPYMOD_RES_METAL],
        freqtime=EMPYMOD_FREQ,
        mpermH=[1.0, EMPYMOD_MU_METAL],
        srcpts=srcpts,
        recpts=recpts,
        verb=0,
    )
    signals = np.diag(response)
    return signals.real, signals.imag


def channel_harmonics(height, tx_name, rx_name, tilt_coeffs):
    if USE_EMPYMOD:
        if USE_EMPYMOD_LOOP:
            re_samples, im_samples = empymod_loop_channel_iq_all_rotations(height, tx_name, rx_name, tilt_coeffs)
        else:
            re_samples, im_samples = empymod_channel_iq_all_rotations(height, tx_name, rx_name, tilt_coeffs)
    else:
        samples = np.array(
            [channel_iq_at_rotation(height, tx_name, rx_name, tilt_coeffs, phi) for phi in ROTATION_ANGLES_DEG],
            dtype=np.float64,
        )
        re_samples, im_samples = samples[:, 0], samples[:, 1]
    return instrument_harmonics_from_iq(re_samples, im_samples)


def xy_channel_harmonics(height, tx_name, rx_name, tilt_coeffs):
    samples = np.array(
        [xy_channel_iq_at_rotation(height, tx_name, rx_name, tilt_coeffs, phi) for phi in ROTATION_ANGLES_DEG],
        dtype=np.float64,
    )
    return instrument_harmonics_from_iq(samples[:, 0], samples[:, 1])


def harmonic_model(tilt_vector, heights=CALIBRATION_HEIGHTS):
    tilt_coeffs = tilts_from_vector(tilt_vector)
    harmonics = np.zeros((len(heights), len(CHANNELS), 3), dtype=np.float64)
    for i, height in enumerate(heights):
        for j, (tx_name, rx_name) in enumerate(CHANNELS):
            harmonics[i, j] = channel_harmonics(height, tx_name, rx_name, tilt_coeffs)
    return harmonics


def xy_harmonic_model(tilt_vector, heights=CALIBRATION_HEIGHTS):
    tilt_coeffs = xy_tilts_from_vector(tilt_vector)
    harmonics = np.zeros((len(heights), len(CHANNELS), 3), dtype=np.float64)
    for i, height in enumerate(heights):
        for j, (tx_name, rx_name) in enumerate(CHANNELS):
            harmonics[i, j] = xy_channel_harmonics(height, tx_name, rx_name, tilt_coeffs)
    return harmonics


RX_NAMES = ['Rzx1', 'Rzx2']
RX9_TILT_NAMES = (
    [f'{n}_x' for n in ['Tzz1', 'Tzz2', 'Tzz3', 'Tzz4', 'Tzz5']]
    + [f'{n}_{ax}' for n in RX_NAMES for ax in ('x', 'y')]
)


def rx9_tilts_from_vector(v):
    return dict(zip(RX9_TILT_NAMES, v))


def rx9_channel_harmonics(height, tx_name, rx_name, tilt_coeffs):
    tx_tilt = tilt_coeffs[f'{tx_name}_x']
    rx_x = tilt_coeffs[f'{rx_name}_x']
    rx_y = tilt_coeffs[f'{rx_name}_y']
    tx_z = TRANSMITTERS_Z[tx_name]
    rx_z = RECEIVERS_Z[rx_name]
    re_samples = np.zeros(16, dtype=np.float64)
    im_samples = np.zeros(16, dtype=np.float64)
    for i, phi_deg in enumerate(ROTATION_ANGLES_DEG):
        phi_rad = np.deg2rad(phi_deg)
        rot2 = rotation_xz_to_lab(phi_rad)
        rot3 = rotation_xyz_to_lab(phi_rad)
        tx_moment_lab = rot2 @ moment_xz_from_tilt(tx_tilt)
        rx_moment_lab = rot3 @ moment_xyz_from_tilts(rx_x, rx_y)
        direct_r = np.array([0.0, 0.0, rx_z - tx_z], dtype=np.float64)
        image_r = np.array([0.0, 2.0 * height, rx_z - tx_z], dtype=np.float64)
        image_tx = IMAGE_MOMENT_REFLECTION @ tx_moment_lab
        sig = (rx_moment_lab @ dipole_tensor_3d(direct_r) @ tx_moment_lab
               + rx_moment_lab @ dipole_tensor_3d(image_r) @ image_tx)
        re_samples[i] = sig
        im_samples[i] = 0.0
    return instrument_harmonics_from_iq(re_samples, im_samples)


def rx9_harmonic_model(tilt_vector, heights=CALIBRATION_HEIGHTS):
    tc = rx9_tilts_from_vector(tilt_vector)
    harmonics = np.zeros((len(heights), len(CHANNELS), 3), dtype=np.float64)
    for i, height in enumerate(heights):
        for j, (tx_name, rx_name) in enumerate(CHANNELS):
            harmonics[i, j] = rx9_channel_harmonics(height, tx_name, rx_name, tc)
    return harmonics


def rx9_harmonic_residuals(tilt_vector, measured_harmonics, heights):
    calculated_ratios = harmonic_ratios(rx9_harmonic_model(tilt_vector, heights=heights))
    measured_ratios = harmonic_ratios(measured_harmonics)
    scale = np.maximum(np.abs(measured_ratios), 1e-6)
    residuals = (calculated_ratios - measured_ratios) / scale
    residuals[:, :, 1] *= H2_WEIGHT
    return residuals.ravel()


def estimate_rx9_tilts_from_harmonics(measured_harmonics, heights):
    tx_starts = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 8.0, 12.0, 10.0, 8.0],
        [-5.0, -8.0, -12.0, -10.0, -8.0],
    ]
    rx_starts = [
        [45.0, 0.0, 45.0, 0.0],
        [50.0, 5.0, 50.0, 5.0],
        [55.0, -5.0, 55.0, -5.0],
    ]
    best_result = None
    for tx_ang in tx_starts:
        for rx_ang in rx_starts:
            x0 = np.tan(np.deg2rad(np.array(tx_ang + rx_ang, dtype=np.float64)))
            from scipy.optimize import least_squares as _ls
            result = _ls(
                rx9_harmonic_residuals,
                x0,
                args=(measured_harmonics, heights),
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                max_nfev=20000,
            )
            if best_result is None or np.linalg.norm(result.fun) < np.linalg.norm(best_result.fun):
                best_result = result
    tc = rx9_tilts_from_vector(best_result.x)
    return tc, best_result


def harmonic_ratios(harmonics):
    h0 = np.maximum(harmonics[:, :, 0], 1e-30)
    h1 = harmonics[:, :, 1]
    h2 = harmonics[:, :, 2]
    return np.stack((h1 / h0, h2 / h0), axis=2)


def harmonic_residuals(tilt_vector, measured_harmonics, heights):
    calculated_ratios = harmonic_ratios(harmonic_model(tilt_vector, heights=heights))
    measured_ratios = harmonic_ratios(measured_harmonics)
    scale = np.maximum(np.abs(measured_ratios), 1e-6)
    residuals = (calculated_ratios - measured_ratios) / scale
    residuals[:, :, 1] *= H2_WEIGHT
    return residuals.ravel()


RX2_NAMES = ['Rzx1', 'Rzx2']


def rx2_harmonic_model(rx_angles_vec, heights=CALIBRATION_HEIGHTS):
    tc = {n: 0.0 for n in TILT_NAMES}
    tc['Rzx1'] = rx_angles_vec[0]
    tc['Rzx2'] = rx_angles_vec[1]
    harmonics = np.zeros((len(heights), len(CHANNELS), 3), dtype=np.float64)
    for i, height in enumerate(heights):
        for j, (tx_name, rx_name) in enumerate(CHANNELS):
            harmonics[i, j] = channel_harmonics(height, tx_name, rx_name, tc)
    return harmonics


def correct_encoder_bias(measured_harmonics, bias=None):
    if bias is None:
        bias = H2_ENCODER_BIAS
    if bias == 0.0:
        return measured_harmonics
    corrected = measured_harmonics.copy()
    corrected[:, :, 2] = measured_harmonics[:, :, 2] - bias * measured_harmonics[:, :, 1]
    return corrected


def rx2_harmonic_residuals(rx_angles_vec, measured_harmonics, heights):
    calc = rx2_harmonic_model(rx_angles_vec, heights=heights)
    h0_c = np.maximum(calc[:, :, 0], 1e-30)
    h0_m = np.maximum(measured_harmonics[:, :, 0], 1e-30)
    r1_calc = calc[:, :, 1] / h0_c
    r1_meas = measured_harmonics[:, :, 1] / h0_m
    scale = np.maximum(np.abs(r1_meas), 1e-6)
    return ((r1_calc - r1_meas) / scale).ravel()


def estimate_rx2_tilts(measured_harmonics, heights):
    best_result = None
    for rzx1_deg in [45.0, 50.0, 55.0, 60.0]:
        for rzx2_deg in [45.0, 50.0, 55.0, 60.0]:
            x0 = np.tan(np.deg2rad(np.array([rzx1_deg, rzx2_deg])))
            from scipy.optimize import least_squares as _ls
            result = _ls(
                rx2_harmonic_residuals, x0,
                args=(measured_harmonics, heights),
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                max_nfev=10000,
            )
            if best_result is None or np.linalg.norm(result.fun) < np.linalg.norm(best_result.fun):
                best_result = result
    rv = best_result.x
    return {'Rzx1': rv[0], 'Rzx2': rv[1]}, best_result


def harmonic_residuals_with_offset(param_vector, measured_harmonics, heights):
    n_tilts = len(TILT_NAMES)
    tilt_vector = param_vector[:n_tilts]
    offsets = param_vector[n_tilts:]
    calc = harmonic_model(tilt_vector, heights=heights)
    h0_meas_raw = measured_harmonics[:, :, 0]
    h0_meas_corr = np.maximum(h0_meas_raw - offsets[np.newaxis, :], 1e-30)
    h0_calc = np.maximum(calc[:, :, 0], 1e-30)
    r10_meas = measured_harmonics[:, :, 1] / h0_meas_corr
    r20_meas = measured_harmonics[:, :, 2] / h0_meas_corr
    r10_calc = calc[:, :, 1] / h0_calc
    r20_calc = calc[:, :, 2] / h0_calc
    sc10 = np.maximum(np.abs(r10_meas), 1e-6)
    sc20 = np.maximum(np.abs(r20_meas), 1e-6)
    res = np.stack(((r10_calc - r10_meas) / sc10,
                    (r20_calc - r20_meas) / sc20), axis=2)
    return res.ravel()


def estimate_tilts_with_offset(measured_harmonics, heights):
    if USE_EMPYMOD:
        tx_start_angles = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 8.0, 14.0, 12.0, 10.0],
        ]
        rx_start_angles = [
            [45.0, 45.0],
            [55.0, 53.0],
        ]
    else:
        tx_start_angles = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, -5.0, 7.0, -9.0, 1.0],
            [-3.0, 5.0, -7.0, 9.0, -1.0],
        ]
        rx_start_angles = [
            [35.0, 35.0],
            [45.0, 45.0],
            [50.0, 50.0],
        ]
    h0_min_per_channel = np.min(measured_harmonics[:, :, 0], axis=0)
    offset_upper = 0.5 * h0_min_per_channel
    tilt_bound = np.tan(np.deg2rad(89.0))
    lb = np.concatenate([np.full(len(TILT_NAMES), -tilt_bound), np.zeros(len(CHANNELS))])
    ub = np.concatenate([np.full(len(TILT_NAMES),  tilt_bound), offset_upper])
    best_result = None
    for tx_angles in tx_start_angles:
        for rx_angles in rx_start_angles:
            tilt_x0 = np.tan(np.deg2rad(np.array(tx_angles + rx_angles, dtype=np.float64)))
            offset_x0 = np.zeros(len(CHANNELS), dtype=np.float64)
            x0 = np.concatenate([tilt_x0, offset_x0])
            result = least_squares(
                harmonic_residuals_with_offset,
                x0,
                args=(measured_harmonics, heights),
                bounds=(lb, ub),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=20000,
            )
            if best_result is None or np.linalg.norm(result.fun) < np.linalg.norm(best_result.fun):
                best_result = result
    result = best_result
    n_tilts = len(TILT_NAMES)
    tilt_dict = tilts_from_vector(result.x[:n_tilts])
    offsets = result.x[n_tilts:]
    return tilt_dict, offsets, result


def xy_harmonic_residuals(tilt_vector, measured_harmonics, heights):
    calculated_ratios = harmonic_ratios(xy_harmonic_model(tilt_vector, heights=heights))
    measured_ratios = harmonic_ratios(measured_harmonics)
    scale = np.maximum(np.abs(measured_ratios), 1e-6)
    return ((calculated_ratios - measured_ratios) / scale).ravel()


def estimate_tilts_from_harmonics(measured_harmonics, heights):
    if USE_EMPYMOD:
        tx_start_angles = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 8.0, 14.0, 12.0, 10.0],
        ]
        rx_start_angles = [
            [45.0, 45.0],
            [55.0, 53.0],
        ]
    else:
        tx_start_angles = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, -5.0, 7.0, -9.0, 1.0],
            [-3.0, 5.0, -7.0, 9.0, -1.0],
            [5.0, -5.0, 5.0, -5.0, 2.0],
            [-5.0, 5.0, -5.0, 5.0, -2.0],
        ]
        rx_start_angles = [
            [35.0, 35.0],
            [40.0, 45.0],
            [45.0, 45.0],
            [45.0, 50.0],
            [50.0, 50.0],
        ]
    best_result = None
    for tx_angles in tx_start_angles:
        for rx_angles in rx_start_angles:
            x0 = np.tan(np.deg2rad(np.array(tx_angles + rx_angles, dtype=np.float64)))
            result = least_squares(
                harmonic_residuals,
                x0,
                args=(measured_harmonics, heights),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=20000,
            )
            if best_result is None or np.linalg.norm(result.fun) < np.linalg.norm(best_result.fun):
                best_result = result
    result = best_result
    return tilts_from_vector(result.x), result


def estimate_xy_tilts_from_harmonics(measured_harmonics, heights):
    tx_start_angles = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 8.0, 14.0, 12.0, 10.0],
        [-10.0, -8.0, -14.0, -12.0, -10.0],
    ]
    rx_start_angles = [
        [45.0, 45.0],
        [55.0, 53.0],
    ]
    y_start_angles = [
        [0.0] * len(TILT_NAMES),
        [5.0] * len(TILT_NAMES),
        [-5.0] * len(TILT_NAMES),
    ]
    best_result = None
    for tx_angles in tx_start_angles:
        for rx_angles in rx_start_angles:
            x_angles = tx_angles + rx_angles
            for y_angles in y_start_angles:
                xy_angles = []
                for x_angle, y_angle in zip(x_angles, y_angles):
                    xy_angles.extend([x_angle, y_angle])
                x0 = np.tan(np.deg2rad(np.array(xy_angles, dtype=np.float64)))
                result = least_squares(
                    xy_harmonic_residuals,
                    x0,
                    args=(measured_harmonics, heights),
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=40000,
                )
                if best_result is None or np.linalg.norm(result.fun) < np.linalg.norm(best_result.fun):
                    best_result = result
    result = best_result
    return xy_tilts_from_vector(result.x), result


def print_rank_diagnostics(result, parameter_names):
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    tolerance = np.finfo(float).eps * max(result.jac.shape) * singular_values[0]
    rank = np.sum(singular_values > tolerance)
    condition = singular_values[0] / singular_values[-1] if singular_values[-1] > 0.0 else np.inf
    print(f'Ранг Якобиана: {rank} из {len(parameter_names)}')
    print(f'Число обусловленности Якобиана: {condition:.6e}')


def print_calibration_setup(heights=CALIBRATION_HEIGHTS):
    print('=== КАЛИБРОВКА ПО МОДУЛЯМ ГАРМОНИК ВРАЩЕНИЯ ===')
    print('Высоты калибровки, м:', heights)
    print('Углы вращения, град:', ROTATION_ANGLES_DEG)
    print('Формат данных: 5 высот x 8 каналов x 3 гармоники: |H0|, |H1|, |H2|')
    print('Смысл гармоник: |H0| - zz+xx, |H1| - перекрестные zx+xz, |H2| - отраженная xx')
    print('Используемые отношения: |H1|/|H0| и |H2|/|H0|')
    print('Неизвестные наклоны mx/mz:', TILT_NAMES)
    print('Рабочие пары каналов:')
    for i, (tx_name, rx_name) in enumerate(CHANNELS):
        print(f'  {i}: {tx_name} -> {rx_name}')


def print_synthetic_calibration_demo():
    true_angles_deg = {
        'Tzz1': 3.0,
        'Tzz2': -5.0,
        'Tzz3': 7.5,
        'Tzz4': -9.0,
        'Tzz5': 1.5,
        'Rzx1': 44.0,
        'Rzx2': 47.0,
    }
    true_vector = vector_from_angles_deg(true_angles_deg)
    measured_harmonics = harmonic_model(true_vector, heights=CALIBRATION_HEIGHTS)
    estimated_tilts, result = estimate_tilts_from_harmonics(measured_harmonics, CALIBRATION_HEIGHTS)
    print()
    print('=== СИНТЕТИЧЕСКИЙ ТЕСТ ПО ГАРМОНИКАМ ===')
    print(' name | true, deg | estimated, deg | error, deg | estimated mx/mz')
    print('------|-----------|----------------|------------|----------------')
    for name in TILT_NAMES:
        estimated_angle = tilt_coeff_to_angle_deg(estimated_tilts[name])
        error = estimated_angle - true_angles_deg[name]
        print(
            f'{name:5s} | {true_angles_deg[name]:9.4f} | {estimated_angle:14.4f} | '
            f'{error:10.4e} | {estimated_tilts[name]:14.6e}'
        )
    print(f'Норма невязки: {np.linalg.norm(result.fun):.6e}')
    print_rank_diagnostics(result, TILT_NAMES)


def run_real_calibration_if_data_present():
    if MEASURED_HARMONICS is None:
        print()
        print('MEASURED_HARMONICS не задан. Вставьте измерения формы (5, 8, 3): |H0|, |H1|, |H2|.')
        return
    measured = np.asarray(MEASURED_HARMONICS, dtype=np.float64)
    heights = CALIBRATION_HEIGHTS if MEASURED_HEIGHTS is None else np.asarray(MEASURED_HEIGHTS, dtype=np.float64)
    expected_shape = (len(heights), len(CHANNELS), 3)
    if measured.shape != expected_shape:
        raise ValueError(f'MEASURED_HARMONICS должен иметь форму {expected_shape}')
    if np.any(~np.isfinite(heights)) or np.any(heights <= 0.0):
        raise ValueError('MEASURED_HEIGHTS должен содержать положительные конечные высоты.')
    estimated_tilts, result = estimate_tilts_from_harmonics(measured, heights)
    print()
    print('=== ОЦЕНКА УГЛОВ ПО РЕАЛЬНЫМ ГАРМОНИКАМ ===')
    print('Высоты измерений, м:', heights)
    print('Оцененные углы элементов:')
    for name in TILT_NAMES:
        print(f'{name}: {tilt_coeff_to_angle_deg(estimated_tilts[name]):.6f} deg, mx/mz={estimated_tilts[name]:.6e}')
    print(f'Норма невязки: {np.linalg.norm(result.fun):.6e}')
    print_rank_diagnostics(result, TILT_NAMES)


if __name__ == '__main__':
    print_calibration_setup()
    print_synthetic_calibration_demo()
    run_real_calibration_if_data_present()

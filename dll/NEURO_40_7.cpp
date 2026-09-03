/**
 * NEURO_40_7P_INVERSION_ENGINE
 * Автономная библиотека для инверсии данных электромагнитного каротажа.
 *
 * Правки относительно исходной версии:
 *  - защита от вызова RunPGDInversion до успешного InitEngine (раньше — зависание);
 *  - строгая проверка чтения весов: битый или обрезанный файл больше не проходит
 *    как валидный (раньше InitEngine возвращал true и инверсия выдавала мусор);
 *  - проверка входных указателей;
 *  - экспортированы IsEngineReady() и GetLastErrorText() для диагностики.
 *
 * Каталог, передаваемый в InitEngine, — общий каталог движка (engine_weights):
 * FWD_W1..5.txt, FWD_b1..5.txt, INV_W1..5.txt, INV_b1..5.txt и четыре файла скейлеров.
 */
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
using namespace Eigen;
// Глобальные переменные среды
MatrixXf FWD_W[5], INV_W[5];
RowVectorXf FWD_b[5], INV_b[5];
RowVectorXf mean_X, scale_X, mean_Y, scale_Y;
int g_max_steps = 150;
float g_lr = 0.015f;
bool g_initialized = false;
std::string g_last_error;
const float SQRT_2 = 1.41421356237f;
const float SQRT_2PI = 2.50662827463f;
const int AMP_INDICES[16] = { 4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 30, 32, 34, 36, 38 };
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x / SQRT_2)); }
inline float gelu_deriv(float x) { return 0.5f * (1.0f + std::erf(x / SQRT_2)) + (x * std::exp(-0.5f * x * x)) / SQRT_2PI; }
// Чтение ровно count значений: меньше значений, мусор вместо числа или лишние
// значения в конце файла считаются ошибкой.
static bool read_exact(const std::string& path, float* dst, int count) {
	std::ifstream file(path);
	if (!file.is_open()) {
		g_last_error = "не открывается файл " + path;
		return false;
	}
	for (int i = 0; i < count; ++i) {
		if (!(file >> dst[i])) {
			g_last_error = path + ": прочитано " + std::to_string(i) +
				" значений вместо " + std::to_string(count);
			return false;
		}
		if (!std::isfinite(dst[i])) {
			g_last_error = path + ": нечисловое значение в позиции " + std::to_string(i);
			return false;
		}
	}
	float extra;
	if (file >> extra) {
		g_last_error = path + ": в файле больше " + std::to_string(count) + " значений";
		return false;
	}
	return true;
}
bool load_matrix(const std::string& path, MatrixXf& mat, int rows, int cols) {
	std::vector<float> buf(static_cast<size_t>(rows) * cols);
	if (!read_exact(path, buf.data(), rows * cols)) return false;
	mat.resize(rows, cols);
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			mat(i, j) = buf[static_cast<size_t>(i) * cols + j];
	return true;
}
bool load_vector(const std::string& path, RowVectorXf& vec, int cols) {
	std::vector<float> buf(static_cast<size_t>(cols));
	if (!read_exact(path, buf.data(), cols)) return false;
	vec.resize(cols);
	for (int j = 0; j < cols; ++j) vec(j) = buf[j];
	return true;
}
// ==========================================================
// ИНИЦИАЛИЗАЦИЯ (Вызывать 1 раз при старте программы)
// ==========================================================
/**
 * @brief Загружает конфигурацию нейросетей в память.
 * @param dir_path Путь к конфигурационной папке с 24 текстовыми файлами (веса и скейлеры).
 * @param max_steps Максимальное количество итераций оптимизации (Рекомендуется: 150).
 * @param lr Скорость обучения оптимайзера (Рекомендуется: 0.015).
 * @return true - успешно, false - ошибка чтения файлов (текст ошибки: GetLastErrorText).
 */
extern "C" __declspec(dllexport) bool InitEngine(const char* dir_path, int max_steps, float lr) {
	g_initialized = false;
	g_last_error.clear();
	if (dir_path == nullptr) {
		g_last_error = "dir_path == nullptr";
		return false;
	}
	if (max_steps <= 0) {
		g_last_error = "max_steps должен быть положительным";
		return false;
	}
	if (!(lr > 0.0f) || !std::isfinite(lr)) {
		g_last_error = "lr должен быть положительным конечным числом";
		return false;
	}
	std::string dir = dir_path;
	g_max_steps = max_steps;
	g_lr = lr;
	int f_sz[] = { 7, 512, 1024, 512, 256, 40 };
	int i_sz[] = { 40, 512, 1024, 512, 256, 7 };
	for (int i = 0; i < 5; ++i) {
		if (!load_matrix(dir + "/FWD_W" + std::to_string(i + 1) + ".txt", FWD_W[i], f_sz[i], f_sz[i + 1])) return false;
		if (!load_vector(dir + "/FWD_b" + std::to_string(i + 1) + ".txt", FWD_b[i], f_sz[i + 1])) return false;
		if (!load_matrix(dir + "/INV_W" + std::to_string(i + 1) + ".txt", INV_W[i], i_sz[i], i_sz[i + 1])) return false;
		if (!load_vector(dir + "/INV_b" + std::to_string(i + 1) + ".txt", INV_b[i], i_sz[i + 1])) return false;
	}
	if (!load_vector(dir + "/scaler_X_mean.txt", mean_X, 7)) return false;
	if (!load_vector(dir + "/scaler_X_scale.txt", scale_X, 7)) return false;
	if (!load_vector(dir + "/scaler_Y_mean.txt", mean_Y, 40)) return false;
	if (!load_vector(dir + "/scaler_Y_scale.txt", scale_Y, 40)) return false;
	if ((scale_X.array() == 0.0f).any() || (scale_Y.array() == 0.0f).any()) {
		g_last_error = "в скейлерах есть нулевой масштаб";
		return false;
	}
	g_initialized = true;
	return true;
}
/** @brief true, если движок готов к работе (InitEngine отработал успешно). */
extern "C" __declspec(dllexport) bool IsEngineReady() { return g_initialized; }
/** @brief Текст последней ошибки инициализации (пустая строка, если ошибок не было). */
extern "C" __declspec(dllexport) const char* GetLastErrorText() { return g_last_error.c_str(); }
struct FwdState { RowVectorXf z[5], a[5]; };
void forward_net_predict(const RowVectorXf& x, RowVectorXf& out_y, FwdState& s) {
	s.z[0] = x * FWD_W[0] + FWD_b[0];
	s.a[0] = s.z[0].unaryExpr([](float v) { return gelu(v); });
	for (int i = 1; i < 4; ++i) {
		s.z[i] = s.a[i - 1] * FWD_W[i] + FWD_b[i];
		s.a[i] = s.z[i].unaryExpr([](float v) { return gelu(v); });
	}
	s.z[4] = s.a[3] * FWD_W[4] + FWD_b[4];
	out_y = s.z[4];
}
void forward_net_backward(const RowVectorXf& y_pred, const RowVectorXf& y_true, const FwdState& s, RowVectorXf& grad_x) {
	RowVectorXf dy = (2.0f / 40.0f) * (y_pred - y_true);
	RowVectorXf da, dz;
	da = dy * FWD_W[4].transpose();
	for (int i = 3; i >= 0; --i) {
		dz = da.cwiseProduct(s.z[i].unaryExpr([](float v) { return gelu_deriv(v); }));
		if (i == 0) grad_x = dz * FWD_W[i].transpose();
		else da = dz * FWD_W[i].transpose();
	}
}
void inverse_net_predict(const RowVectorXf& y_in, RowVectorXf& x_out) {
	RowVectorXf a = y_in;
	for (int i = 0; i < 4; ++i) {
		RowVectorXf z = a * INV_W[i] + INV_b[i];
		a = z.unaryExpr([](float v) { return gelu(v); });
	}
	x_out = a * INV_W[4] + INV_b[4];
}
// ==========================================================
// РАБОЧИЙ ЦИКЛ (Вызывать для каждой точки инверсии)
// ==========================================================
/**
 * @brief Выполняет расчет параметров среды (7 шт) на основе показаний прибора (40 шт).
 * ВСЕ ДАННЫЕ ПЕРЕДАЮТСЯ И ВОЗВРАЩАЮТСЯ В СЫРОМ ФИЗИЧЕСКОМ ВИДЕ! Внутренняя предобработка автоматизирована.
 *
 * Если движок не инициализирован (InitEngine не вызывался или вернул false),
 * функция заполняет raw_geo_7_out значениями NaN и сразу возвращает управление.
 *
 * @param raw_signals_40 [ВХОД] Показания ЭМ-прибора. 40 элементов.
 *
 * СТРУКТУРА МАССИВА raw_signals_40 (Амплитуды - линейные единицы, Фазы - градусы):
 * --- Блок 400 кГц (Индексы 0 - 19) ---
 * [0...3]   Фазы ZZ (T1_Simm, T2_Simm, T3_Simm, T4_Simm)
 * [4...7]   Зонд ZX L=0.96м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * [8...11]  Зонд ZX L=1.62м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * [12...15] Зонд ZX L=2.28м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * [16...19] Зонд ZX L=3.00м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * --- Блок 2 МГц (Индексы 20 - 39) ---
 * [20...23] Фазы ZZ (T1_Simm, T2_Simm, T3_Simm, T4_Simm)
 * [24...27] Зонд ZX L=0.96м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * [28...31] Зонд ZX L=1.62м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * [32...35] Зонд ZX L=2.28м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 * [36...39] Зонд ZX L=3.00м: Сигнал A (Ампл, Фаза), Сигнал B (Ампл, Фаза)
 *
 * @param min_bounds_7   [ВХОД] Нижний лимит поиска для 7 геологических параметров.
 * @param max_bounds_7   [ВХОД] Верхний лимит поиска для 7 геологических параметров.
 * @param raw_geo_7_out  [ВЫХОД] Результат инверсии.
 *
 * ПОРЯДОК ЭЛЕМЕНТОВ В ГЕОЛОГИЧЕСКИХ МАССИВАХ (min_bounds, max_bounds, raw_geo_7_out):
 * [0] УЭС кровли (Ом*м)
 * [1] УЭС пласта гор. (Ом*м)
 * [2] УЭС пласта верт. (Ом*м)
 * [3] УЭС подошвы (Ом*м)
 * [4] Расстояние до кровли (Метры)
 * [5] Расстояние до подошвы (Метры)
 * [6] Зенитный угол (Градусы)
 */
extern "C" __declspec(dllexport) void RunPGDInversion(
	const float* raw_signals_40,
	const float* min_bounds_7,
	const float* max_bounds_7,
	float* raw_geo_7_out
) {
	if (raw_geo_7_out == nullptr) return;
	if (!g_initialized || raw_signals_40 == nullptr ||
		min_bounds_7 == nullptr || max_bounds_7 == nullptr) {
		if (g_last_error.empty()) g_last_error = "RunPGDInversion вызван до успешного InitEngine";
		for (int i = 0; i < 7; ++i) raw_geo_7_out[i] = std::numeric_limits<float>::quiet_NaN();
		return;
	}
	RowVectorXf y_true = Map<const RowVectorXf>(raw_signals_40, 40);

	// Внутренняя техническая нормализация
	for (int idx : AMP_INDICES) y_true(idx) = std::log10(std::fmax(y_true(idx), 1e-8f));
	y_true = (y_true - mean_Y).cwiseQuotient(scale_Y);
	RowVectorXf min_b = Map<const RowVectorXf>(min_bounds_7, 7);
	RowVectorXf max_b = Map<const RowVectorXf>(max_bounds_7, 7);

	for (int i = 0; i < 4; ++i) {
		min_b(i) = std::log10(std::fmax(min_b(i), 1e-5f));
		max_b(i) = std::log10(std::fmax(max_b(i), 1e-5f));
	}
	RowVectorXf min_scaled = (min_b - mean_X).cwiseQuotient(scale_X);
	RowVectorXf max_scaled = (max_b - mean_X).cwiseQuotient(scale_X);
	RowVectorXf opt_geo(7);
	inverse_net_predict(y_true, opt_geo);
	opt_geo = opt_geo.cwiseMax(min_scaled).cwiseMin(max_scaled);
	FwdState fwd_state;
	RowVectorXf y_pred(40), grad_x(7);
	RowVectorXf m = RowVectorXf::Zero(7), v = RowVectorXf::Zero(7);
	const float beta1 = 0.9f, beta2 = 0.999f, epsilon = 1e-8f;
	for (int step = 1; step <= g_max_steps; ++step) {
		forward_net_predict(opt_geo, y_pred, fwd_state);
		forward_net_backward(y_pred, y_true, fwd_state, grad_x);

		m = beta1 * m + (1.0f - beta1) * grad_x;
		v = beta2 * v + (1.0f - beta2) * grad_x.array().square().matrix();

		RowVectorXf m_hat = m / (1.0f - std::pow(beta1, step));
		RowVectorXf v_hat = v / (1.0f - std::pow(beta2, step));

		opt_geo.array() -= g_lr * m_hat.array() / (v_hat.array().sqrt() + epsilon);
		opt_geo = opt_geo.cwiseMax(min_scaled).cwiseMin(max_scaled);
	}
	RowVectorXf final_geo = opt_geo.cwiseProduct(scale_X) + mean_X;
	for (int i = 0; i < 4; ++i) final_geo(i) = std::pow(10.0f, final_geo(i));

	Map<RowVectorXf>(raw_geo_7_out, 7) = final_geo;
}

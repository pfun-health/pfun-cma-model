/*
  pfun_cma_engine.h: Header definitions for the PFun CMA Model Engine
*/

#ifndef PFUN_CMA_ENGINE_H
#define PFUN_CMA_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// Low-level numerical methods
double exp_clipped(double x);
double expit(double x);
double calc_vdep_current(double v, double v1, double v2, double A, double B);
double E_pfun(double x);
double Light_pfun(double x);
double K_pfun(double x);
double meal_distr_pfun(double Cm, double t, double toff);

// Signal calculations (vectorized)
void calc_L(const double* t, int N, double d, double taup, double eps, double* out);
void calc_M(const double* t, int N, const double* L, double d, double eps, int* seed, double* out);
void calc_c(const double* t, int N, const double* L, const double* m, double d, double taup, double* out);
void calc_a(const double* t, int N, const double* c, const double* m, const double* L, double d, double taup, double eps, double* out);
void calc_I_S(int N, const double* c, const double* m, double* out);
void calc_I_E(int N, const double* a, const double* I_S, double* out);

/**
 * Calculate post-prandial glucose dynamics.
 *
 * @param t Time vector (size N)
 * @param N Number of time points
 * @param I_E Extracellular insulin vector (size N)
 * @param tM Meal times vector (size n_meals)
 * @param n_meals Number of meals
 * @param taug Meal durations vector (size n_meals)
 * @param B Bias constant
 * @param Cm Cortisol temporal sensitivity coefficient
 * @param toff Meal-relative time offset
 * @param out_G_instant Output for instantaneous glucose (size N, optional, can be NULL)
 * @param out_g_components Output for per-meal components (size n_meals * N, row-major: meal x time, optional, can be NULL)
 */
void calc_G(const double* t, int N, const double* I_E, const double* tM, int n_meals, const double* taug, double B, double Cm, double toff, int include_bias_in_components, double* out_G_instant, double* out_g_components);

/**
 * Run the full CMA model.
 * All output buffers (L, m, c, a, I_S, I_E, G, g) should be pre-allocated.
 */
void run_cma_model(
    const double* t, int N,
    double d, double taup, double taug_val, // taug_val used if taug_vec is NULL
    const double* taug_vec,
    double B, double Cm, double toff,
    const double* tM, int n_meals,
    int* seed, double eps,
    double* out_L, double* out_m, double* out_c, double* out_a,
    double* out_I_S, double* out_I_E, double* out_G, double* out_g
);

#ifdef __cplusplus
}
#endif

#endif // PFUN_CMA_ENGINE_H

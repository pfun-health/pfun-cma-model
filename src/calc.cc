#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

// Clipping function for overflow control
double clip(double x, double min_val, double max_val) {
    return std::fmin(std::fmax(x, min_val), max_val);
}

// Exponential function with clipping to avoid overflow
double exp_clipped(double x) {
    x = clip(x, -709.0, 709.0);  // To avoid overflow
    return std::exp(x);
}

// Sigmoid function (expit)
double expit(double x) {
    return 1.0 / (1.0 + exp_clipped(-2.0 * x));
}

// Voltage-dependent current calculation
double calc_vdep_current(double v, double v1, double v2, double A, double B) {
    return A * expit(B * (v - v1) / v2);
}

// Normalize a flat 1D array between a and b
void normalize(double* x, int n, double a, double b) {
    double xmin = x[0], xmax = x[0];
    for (int i = 1; i < n; ++i) {
        xmin = std::fmin(xmin, x[i]);
        xmax = std::fmax(xmax, x[i]);
    }

    if (xmax == xmin) {
        for (int i = 0; i < n; ++i) {
            x[i] = (a + b) / 2.0;
        }
        return;
    }

    for (int i = 0; i < n; ++i) {
        x[i] = a + (b - a) * (x[i] - xmin) / (xmax - xmin);
    }
}

// Glucose normalization
double normalize_glucose(double G, double g0, double g1, double g_s) {
    double numer = 8.95 * std::pow((G - g_s), 3) + std::pow((G - g0), 2) - std::pow((G - g1), 2);
    return 2.0 * expit(1e-4 * numer / (g1 - g0));
}

// Exponential function
double E(double x) {
    return 1.0 / (1.0 + exp_clipped(-2.0 * x));
}

// Meal distribution function
void meal_distr(double Cm, const double* t, int n, double toff, double* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = std::pow(std::cos(2 * M_PI * Cm * (t[i] + toff) / 24.0), 2);
    }
}

// Glucose response function K
void K(const double* x, int n, double* out) {
    for (int i = 0; i < n; ++i) {
        if (x[i] > 0.0) {
            out[i] = exp_clipped(-std::pow(std::log(2.0 * x[i]), 2));
        } else {
            out[i] = 0.0;
        }
    }
}

// Vectorized G calculation
void vectorized_G(const double* t, int tn, double I_E, const double* tm, const double* taug, int m,
                  double B, double Cm, double toff, double** out) {
    double* k_G = new double[tn];
    double* meal_dis = new double[tn];

    // Compute meal distribution
    meal_distr(Cm, t, tn, toff, meal_dis);

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < tn; ++i) {
            double normalized_time = (t[i] - tm[j]) / std::pow(taug[j], 2);
            if (normalized_time > 0.0) {
                k_G[i] = exp_clipped(-std::pow(std::log(2.0 * normalized_time), 2));
            } else {
                k_G[i] = 0.0;
            }
            out[j][i] = 1.3 * k_G[i] / (1.0 + I_E) + B * (1.0 + meal_dis[i]);
        }
    }

    delete[] k_G;
    delete[] meal_dis;
}

extern "C" {
    void run_calc() {
        // Example usage
        double time[] = {0.0, 1.0, 2.0, 3.0};   // Example time vector
        double meal_times[] = {1.0, 2.0};       // Example meal times
        double meal_duration[] = {1.5, 2.0};    // Example meal durations
        double I_E = 0.1;                       // Example insulin level
        double B = 0.5;                         // Bias constant
        double Cm = 1.2;                        // Cortisol coefficient
        double toff = 1.0;                      // Time offset

        int tn = sizeof(time) / sizeof(time[0]);
        int m = sizeof(meal_times) / sizeof(meal_times[0]);

        double** G_values = new double*[m];
        for (int j = 0; j < m; ++j) {
            G_values[j] = new double[tn];
        }

        vectorized_G(time, tn, I_E, meal_times, meal_duration, m, B, Cm, toff, G_values);

        // Output example result
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < tn; ++i) {
                std::cout << G_values[j][i] << " ";
            }
            std::cout << std::endl;
        }

        // Clean up
        for (int j = 0; j < m; ++j) {
            delete[] G_values[j];
        }
        delete[] G_values;
    }
}
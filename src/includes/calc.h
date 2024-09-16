#ifndef CALC_H
#define CALC_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to clip values between a min and max range
double clip(double x, double min_val, double max_val);

// Exponential function with clipping to avoid overflow
double exp_clipped(double x);

// Sigmoid function (expit)
double expit(double x);

// Calculate voltage-dependent current
double calc_vdep_current(double v, double v1, double v2, double A, double B);

// Normalize a flat 1D array between a and b
void normalize(double* x, int n, double a, double b);

// Glucose normalization function
double normalize_glucose(double G, double g0, double g1, double g_s);

// Exponential function
double E(double x);

// Meal distribution function
void meal_distr(double Cm, double* t, int n, double toff, double* out);

// Glucose response function K
void K(double* x, int n, double* out);

// Vectorized G calculation
void vectorized_G(double* t, int tn, double I_E, double* tm, double* taug, int m,
                  double B, double Cm, double toff, double** out);

#endif // CALC_H

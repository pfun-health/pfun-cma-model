#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

// Clipping function for overflow control
double clip(double x, double min_val, double max_val)
{
    return std::max(min_val, std::min(x, max_val));
}

// Exponential function with clipping to avoid overflow
double exp_clipped(double x)
{
    x = clip(x, -709.0, 709.0); // to avoid overflow
    return std::exp(x);
}

// Sigmoid function (expit in your Python code)
double expit(double x)
{
    return 1.0 / (1.0 + exp_clipped(-2.0 * x));
}

// Calculate voltage-dependent current
double calc_vdep_current(double v, double v1, double v2, double A = 1.0, double B = 1.0)
{
    return A * expit(B * (v - v1) / v2);
}

// Normalize a vector between a and b
std::vector<double> normalize(const std::vector<double> &x, double a = 0.0, double b = 1.0)
{
    double xmin = *std::min_element(x.begin(), x.end());
    double xmax = *std::max_element(x.begin(), x.end());
    std::vector<double> normalized(x.size());

    std::transform(x.begin(), x.end(), normalized.begin(), [&](double xi)
                   { return a + (b - a) * (xi - xmin) / (xmax - xmin); });

    return normalized;
}

// Glucose normalization (simplified version of your original normalize_glucose)
double normalize_glucose(double G, double g0 = 70, double g1 = 180, double g_s = 90)
{
    double numer = 8.95 * std::pow((G - g_s), 3) + std::pow((G - g0), 2) - std::pow((G - g1), 2);
    return 2.0 * expit(1e-4 * numer / (g1 - g0));
}

// Exponential function for the given input
double E(double x)
{
    return 1.0 / (1.0 + exp_clipped(-2.0 * x));
}

// Meal distribution function
std::vector<double> meal_distr(double Cm, const std::vector<double> &t, double toff)
{
    std::vector<double> distribution(t.size());

    std::transform(t.begin(), t.end(), distribution.begin(), [&](double ti)
                   { return std::pow(std::cos(2 * M_PI * Cm * (ti + toff) / 24), 2); });

    return distribution;
}

// Glucose response function K
std::vector<double> K(const std::vector<double> &x)
{
    std::vector<double> result(x.size());

    std::transform(x.begin(), x.end(), result.begin(), [](double xi)
                   {
        if (xi > 0.0) {
            return exp_clipped(-std::pow(std::log(2.0 * xi), 2));
        } else {
            return 0.0;
        } });

    return result;
}

// Vectorized G function
std::vector<std::vector<double>> vectorized_G(const std::vector<double> &t, double I_E,
                                              const std::vector<double> &tm, const std::vector<double> &taug,
                                              double B, double Cm, double toff)
{
    size_t m = tm.size();
    size_t n = t.size();
    std::vector<std::vector<double>> out(m, std::vector<double>(n, 0.0));

    for (size_t j = 0; j < m; ++j)
    {
        std::vector<double> k_G = K((t - tm[j]) / std::pow(taug[j], 2));
        std::transform(k_G.begin(), k_G.end(), out[j].begin(), [&](double kg)
                       { return 1.3 * kg / (1.0 + I_E); });
    }

    // Adding bias constant with meal distribution
    std::vector<double> meal_dis = meal_distr(Cm, t, toff);
    for (size_t j = 0; j < m; ++j)
    {
        std::transform(out[j].begin(), out[j].end(), meal_dis.begin(), out[j].begin(), [&](double outj, double meal_d)
                       { return outj + B * (1.0 + meal_d); });
    }

    return out;
}

int main()
{
    // Example usage:
    std::vector<double> time = {0.0, 1.0, 2.0, 3.0}; // Example time vector
    std::vector<double> meal_times = {1.0, 2.0};     // Example meal times
    std::vector<double> meal_duration = {1.5, 2.0};  // Example meal durations
    double I_E = 0.1;                                // Example insulin level
    double B = 0.5;                                  // Bias constant
    double Cm = 1.2;                                 // Cortisol coefficient
    double toff = 1.0;                               // Time offset

    std::vector<std::vector<double>> G_values = vectorized_G(time, I_E, meal_times, meal_duration, B, Cm, toff);

    // Output example result
    for (const auto &row : G_values)
    {
        for (const auto &val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

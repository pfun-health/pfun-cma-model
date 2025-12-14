/* This is sloppy AI-generated code, not yet an actual implementation.
*/
// pfun_cma_engine.c
// Build as a CPython extension named "pfun_cma_engine"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>

// Helper: convert a Python sequence to an array of doubles.
// Returns newly allocated double* (caller must free) and sets *len, or returns NULL on error.
static double *
seq_to_doubles(PyObject *seq, Py_ssize_t *out_len)
{
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of numbers");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    PyObject **items = PySequence_Fast_ITEMS(fast);

    double *arr = (double*)malloc(sizeof(double) * (size_t)n);
    if (!arr) {
        PyErr_NoMemory();
        Py_DECREF(fast);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *it = items[i];
        double v = PyFloat_AsDouble(it);
        if (PyErr_Occurred()) { free(arr); Py_DECREF(fast); return NULL; }
        arr[i] = v;
    }

    *out_len = n;
    Py_DECREF(fast);
    return arr;
}

// Simple Gaussian random generator (Box-Muller) using seedable xorshift32
static unsigned int rng_state = 0x12345678u;
static void rng_seed(unsigned int s) { rng_state = s ? s : 0x12345678u; }
static unsigned int xorshift32(void) {
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static double uniform_rand() {
    return (double)(xorshift32() & 0xFFFFFFu) / (double)0x1000000u;
}
static double gaussian_rand() {
    // Box-Muller
    double u1 = uniform_rand();
    double u2 = uniform_rand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// calc_model Python API
// calc_model(t_seq, d, taup, taug, B, Cm, toff, tM_seq, seed=None, eps=1e-18)
// - t_seq: sequence of times (hours)
// - d: timezone offset (hours)
// - taup: photoperiod length param (unused complicatedly; used to scale melatonin)
// - taug: glucose response time constant (hours)
// - B: glucose bias (baseline)
// - Cm: cortisol sensitivity multiplier
// - toff: solar noon offset
// - tM_seq: sequence of meal times (hours) (any length)
// - seed: optional integer seed, if provided will seed RNG
// - eps: small noise scale
//
// returns: Python list of floats (same length as t_seq) representing modeled glucose
static PyObject *
py_calc_model(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *t_seq = NULL;
    double d = 0.0, taup = 1.0, taug = 1.0, B = 0.05, Cm = 0.0, toff = 0.0, eps = 1e-18;
    PyObject *tM_seq = NULL;
    PyObject *seed_obj = Py_None;

    static char *kwlist[] = {"t_seq", "d", "taup", "taug", "B", "Cm", "toff", "tM_seq", "seed", "eps", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OdddddoO|Od", kwlist,
                                     &t_seq, &d, &taup, &taug, &B, &Cm, &toff, &tM_seq, &seed_obj, &eps)) {
        return NULL;
    }

    Py_ssize_t N;
    double *t = seq_to_doubles(t_seq, &N);
    if (!t) return NULL;

    // meal times
    Py_ssize_t M = 0;
    double *tM = NULL;
    if (tM_seq && tM_seq != Py_None) {
        tM = seq_to_doubles(tM_seq, &M);
        if (!tM) { free(t); return NULL; }
    }

    // seed
    if (seed_obj != Py_None) {
        long s = PyLong_AsLong(seed_obj);
        if (PyErr_Occurred()) { if (tM) free(tM); free(t); return NULL; }
        rng_seed((unsigned int)s);
    } else {
        // default seed from time-ish (not ideal but deterministic if not set)
        rng_seed(0x12345678u);
    }

    PyObject *out_list = PyList_New(N);
    if (!out_list) { if (tM) free(tM); free(t); return NULL; }

    // Precompute meal gaussians scale factor
    double meal_scale = 1.0;
    // Use taup to scale melatonin amplitude (toy model)
    double melatonin_scale = 0.5 * taup;

    for (Py_ssize_t i = 0; i < N; ++i) {
        double ti = t[i];

        // Meal response: sum of gaussian bumps centered at each meal time
        double meal_resp = 0.0;
        if (M > 0) {
            for (Py_ssize_t j = 0; j < M; ++j) {
                double dt = ti - tM[j];
                // wrap-around across 24h (choose minimal circular distance)
                if (dt > 12.0) dt -= 24.0;
                if (dt < -12.0) dt += 24.0;
                double g = exp(- (dt * dt) / (2.0 * taug * taug));
                meal_resp += g;
            }
            meal_resp *= meal_scale;
        }

        // Cortisol: a shifted sinusoid; cortisol peak near morning ~ 8-9h.
        double cortisol_phase = (ti + d + toff) * (2.0 * M_PI / 24.0);
        double cortisol = Cm * (0.5 * (1.0 + sin(cortisol_phase))); // positive bump scaled by Cm

        // Melatonin: antiphasic to day; scaled by taup param
        double mel_phase = (ti + d) * (2.0 * M_PI / 24.0);
        double melatonin = melatonin_scale * (0.5 * (1.0 - sin(mel_phase)));

        // Adiponectin / baseline mixing - simple bias B
        double model_val = B + meal_resp + cortisol - melatonin;

        // small gaussian noise
        if (eps > 0.0) {
            double gnoise = gaussian_rand() * eps;
            model_val += gnoise;
        }

        PyObject *val = PyFloat_FromDouble(model_val);
        if (!val) { Py_DECREF(out_list); if (tM) free(tM); free(t); return NULL; }
        PyList_SET_ITEM(out_list, i, val); // steals ref
    }

    if (tM) free(tM);
    free(t);
    return out_list;
}

static PyMethodDef PfunCmaMethods[] = {
    {"calc_model", (PyCFunction)py_calc_model, METH_VARARGS | METH_KEYWORDS,
     "calc_model(t_seq, d, taup, taug, B, Cm, toff, tM_seq, seed=None, eps=1e-18)\n\n"
     "Return modeled glucose series as a list of floats."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pfun_cma_engine_module = {
    PyModuleDef_HEAD_INIT,
    "pfun_cma_engine",   /* name of module */
    "C implementation of a simple CMA engine (pfun)", /* module documentation */
    -1,
    PfunCmaMethods
};

PyMODINIT_FUNC
PyInit_pfun_cma_engine(void)
{
    PyObject *m = PyModule_Create(&pfun_cma_engine_module);
    if (m == NULL) return NULL;
    // seed default RNG state
    rng_seed(0x12345678u);
    return m;
}


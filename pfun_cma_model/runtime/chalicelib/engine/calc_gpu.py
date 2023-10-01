from numba import cuda, jit, float32
import numpy as np

@cuda.jit(device=True)
def exp_cuda(x):
    return np.exp(x)

@cuda.jit(device=True)
def expit_pfun_cuda(x):
    return 1.0 / (1.0 + exp_cuda(-2.0 * x))

@cuda.jit
def calc_vdep_current_cuda(v, v1, v2, A, B, result):
    i = cuda.grid(1)
    if i < v.shape[0]:
        result[i] = A * expit_pfun_cuda(B * (v[i] - v1) / v2)

@cuda.jit
def E_norm_cuda(x, result):
    i = cuda.grid(1)
    if i < x.shape[0]:
        result[i] = 2.0 * (expit_pfun_cuda(2.0 * x[i]) - 0.5)

@cuda.jit
def normalize_glucose_cuda(G, g0, g1, g_s, result):
    i = cuda.grid(1)
    if i < G.shape[0]:
        numer = 8.95 * ((G[i] - g_s) ** 3) + ((G[i] - g0) ** 2) - ((G[i] - g1) ** 2)
        result[i] = 2.0 * expit_pfun_cuda(1e-4 * numer / (g1 - g0))


def main():
    # Example usage
    n = 1000
    v = np.random.rand(n).astype(np.float32)
    v1 = np.float32(0.5)
    v2 = np.float32(1.0)
    A = np.float32(1.0)
    B = np.float32(1.0)
    result = np.zeros(n, dtype=np.float32)

    threadsperblock = 32
    blockspergrid = (v.shape[0] + (threadsperblock - 1)) // threadsperblock

    calc_vdep_current_cuda[blockspergrid, threadsperblock](v, v1, v2, A, B, result)


if __name__ == "__main__":
    main()
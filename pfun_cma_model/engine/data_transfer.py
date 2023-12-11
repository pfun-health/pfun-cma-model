from numba import cuda
import numpy as np


class DataTransferHandler:
    def __init__(self, stream):
        self.stream = stream

    def to_device_async(self, cpu_data):
        gpu_data = cuda.device_array_like(cpu_data, stream=self.stream)
        cuda.to_device(cpu_data, to=gpu_data, stream=self.stream)
        return gpu_data

    def to_host_async(self, gpu_data, cpu_data):
        cuda.copy_to_host(gpu_data, to=cpu_data, stream=self.stream)

    def execute_function(self, func, *args):
        func[self.grid_size, self.block_size, self.stream](*args)


def main():
    """example usage"""
    # Initialize CUDA stream and DataTransferHandler
    stream = cuda.stream()
    handler = DataTransferHandler(stream)

    # Example usage
    n = 1000
    cpu_data = np.random.rand(n).astype(np.float32)
    gpu_data = handler.to_device_async(cpu_data)

    # Assuming `calc_vdep_current_cuda` is your CUDA kernel
    from pfun_cma_model.runtime.src.engine.calc_gpu import calc_vdep_current_cuda
    handler.execute_function(calc_vdep_current_cuda, gpu_data, ...)

    # Transfer result back to CPU
    result_cpu = np.empty_like(cpu_data)
    handler.to_host_async(gpu_data, result_cpu)

    # Synchronize stream to make sure data transfer is complete
    stream.synchronize()

    print("Result:", result_cpu)


if __name__ == "__main__":
    main()

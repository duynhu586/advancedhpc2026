from numba import cuda

cuda.detect()
gpu = cuda.select_device(0)

print("Multiprocessor count: ", gpu.MULTIPROCESSOR_COUNT)
mem = cuda.current_context().get_memory_info()

print(mem)
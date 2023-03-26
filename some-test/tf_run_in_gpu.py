import tensorflow as tf

tf.test.is_gpu_available()

# Tensorflow 2.10是最后一个在本地windows上支持GPU的版本。从2.11版本开始，需要在windows WLS2（适用于 Linux 的 Windows 子系统）上安装才能使用GPU。所以要在native-windows上使用GPU，就只能安装2.10.0版本及以下的版本，或者安装老版的tensorflow-gpu.
# 指定在cpu上运行
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000, 1000])
    cpu_b = tf.random.normal([1000, 2000])
    cpu_c = tf.matmul(cpu_a, cpu_b)
print("cpu_a:", cpu_a.device)
print("cpu_b:", cpu_b.device)
print("cpu_c:", cpu_c.device)
# 查看gpu是否可用
print(tf.config.list_physical_devices('GPU'))
print('asdfa')
# 指定在gpu上运行
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000, 1000])
    gpu_b = tf.random.normal([1000, 2000])
    gpu_c = tf.matmul(gpu_a, gpu_b)
print("gpu_a:", gpu_a.device)
print("gpu_b:", gpu_b.device)
print("gpu_c:", gpu_c.device)

import tensorflow as tf
import timeit


def cpu_run():
    with tf.device('/cpu:0'):
        cpu_a = tf.random.normal([10000, 1000])
        cpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(cpu_a, cpu_b)
    return c


def gpu_run():
    with tf.device('/gpu:0'):
        gpu_a = tf.random.normal([10000, 1000])
        gpu_b = tf.random.normal([1000, 2000])
        c = tf.matmul(gpu_a, gpu_b)
    return c


cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print("cpu:", cpu_time, "  gpu:", gpu_time)
# 检测 tensorflow 能使用的设备情况
from tensorflow.python.client import device_lib

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 这个可以指定使用哪个设备
print(device_lib.list_local_devices())

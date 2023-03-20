import pynvml

pynvml.nvmlInit()
pynvml.nvmlSystemGetDriverVersion()
pynvml.nvmlDeviceGetCount()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

name=pynvml.nvmlDeviceGetName(handle)

print(name)
# 每MB包含的字节数
NUM_EXPAND = 1024 * 1024

gpu_id=0
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)

gpu_Total = info.total   # 总显存
gpu_Free = info.free
gpu_Used = info.used

print(gpu_Total)  # 显卡总的显存大小,6442450944Bit
print(gpu_Free)  # 显存使用大小,4401950720Bit
print(gpu_Used)  # 显卡剩余显存大小,2040500224Bit

print(gpu_Total / NUM_EXPAND)
print(gpu_Free / NUM_EXPAND)
print(gpu_Used / NUM_EXPAND)

# meminfo.used / 1024 / 1024
#    4198 M


# 可以通过这个获取显存的使用情况
pynvml.nvmlShutdown()
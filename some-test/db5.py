import pywt
import matplotlib.pyplot as plt

# 生成DB5小波
wavelet = pywt.Wavelet('db5')

# 获取尺度函数和小波函数
phi, psi, x = wavelet.wavefun(level=5)

# 绘制尺度函数和小波函数
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, phi)
plt.title('DB5 Scaling Function (Phi)')
plt.subplot(2, 1, 2)
plt.plot(x, psi)
plt.title('DB5 Wavelet Function (Psi)')
plt.tight_layout()
plt.show()

h0=wavelet.dec_lo
h1=wavelet.dec_hi
g0=wavelet.rec_lo
g1=wavelet.rec_hi

print('h0:', h0)
print('h1:', h1)
print('g0:', g0)
print('g1:', g1)
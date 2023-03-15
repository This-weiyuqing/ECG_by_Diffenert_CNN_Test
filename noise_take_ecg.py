import numpy as np
import pywt
import wfdb
from matplotlib import pyplot as plt

#from ECG_read_weiyuqing_without_wfdb import PATH

record=wfdb.rdrecord('./mit-bih-arrhythmia-database-1.0.0/'+100,channel_names=['MLII'])
data=record.p_signal.flatten()

coeffs=pywt.wavedec(data=data,wavelet='db5',level=9)
cA9,cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

threshold=(np.median(np.abs(cD1))/0.6745)*(np.sqrt(2*np.log(len(cD1))))
cD1.fill(0)
cD2.fill(0)
for i in range(1,len(coeffs)-2):
    coeffs[i]=pywt.threshold(coeffs[i],threshold)

rdata=pywt.waverec(coeffs=coeffs,wavelet='db5')

plt.figure(figsize=(20,4))
plt.plot(data)
plt.show()
plt.plot(rdata)
plt.show()
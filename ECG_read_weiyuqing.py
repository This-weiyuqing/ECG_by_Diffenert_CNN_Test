import numpy as np
import matplotlib.pyplot as plt
from numpy import reshape

# arr =
PATH = "mit-bih-arrhythmia-database-1.0.0/"
HEADERFILE = "100.hea"
ATRFILE = '100.atr'
DATAFILE = "100.dat"

# Obtain the information of each part from the hea file and fill in
f = open(PATH + HEADERFILE, "r")
z = f.readline().split()
nosig, sfreq = int(z[1]), int(z[2])  # single number,samp number
SAMPLES2READ = 10 * sfreq  # number of data points

dformat, gain, bitres, zerovalue, firstvalue = [], [], [], [], []

for i in range(nosig):
    z = f.readline().split()
    dformat.append(int(z[1]))
    gain.append(int(z[2]))
    bitres.append(int(z[3]))
    zerovalue.append(int(z[4]))
    firstvalue.append(int(z[5]))
f.close()

# Obtain the information of each part from the dat file and fill in
f=open(PATH+DATAFILE, "rb")
b = f.read()
f.close()

A_init = np.frombuffer(b, dtype=np.uint8)
A_shape0 = int(A_init.shape[0] / 3)
A = A_init.reshape(A_shape0, 3)[:SAMPLES2READ]

M = np.zeros((SAMPLES2READ, 2))

M2H = A[:, 1] >> 4
M1H = A[:, 1] & 15

PRL = (A[:, 1] & 8) * (2 ** 9)
PRR = A[:, 1] & 128 << 5

M1H = M1H * (2 ** 8)
M2H = M2H * (2 ** 8)

M[:, 0] = A[:, 0] + M1H - PRL
M[:, 1] = A[:, 2] + M2H - PRR

if ((M[1, :] != firstvalue).any()):
    print("inconsistent in the first bit value")

if nosig == 2:
    M[:, 0] = (M[:, 0] - zerovalue[0]) / gain[0]
    M[:, 1] = (M[:, 1] - zerovalue[1]) / gain[1]
    TIME = np.linspace(0, SAMPLES2READ - 1, SAMPLES2READ) / sfreq
elif nosig == 1:
    M2 = []
    M[:, 0] = M[:, 0] - zerovalue[0]
    M[:, 1] = M[:, 1] - zerovalue[1]
    for i in range(M.shape[0]):
        M2.append(M[:, 0][i])
        M2.append(M[:, 1][i])
    M2.append(0)
    del M2[0]
    M2 = np.array(M2) / gain[0]
    TIME = np.linspace(0, 2 * SAMPLES2READ - 1, 2 * SAMPLES2READ) / sfreq
else:
    print("Sorting algorithm for more than 2 signals not programmed yet")

# atr  deal
f = open(PATH + ATRFILE, "rb")
b = f.read()
f.close()
A_init = np.frombuffer(b, dtype=np.uint8)
A_shape0 = int(A_init.shape[0] / 2)
A = A_init.reshape(A_shape0, 2)

ANNOT, ATRTIME = [], []
i = 0
while i < A.shape[0]:
    annoth = A[i, 1] >> 2
    if annoth == 59:
        ANNOT.append(A[i + 3, 1] >> 2)
        ATRTIME.append(A[i + 2, 0] + A[i + 2, 1] * (2 ** 8) + A[i + 1, 0] * (2 ** 16) + A[i + 1, 1] * (2 ** 24))
        i += 3
    elif annoth == 60:
        pass
    elif annoth == 61:
        pass
    elif annoth == 62:
        pass
    elif annoth == 63:
        hilfe = (A[i, 1] & 3) * (2 * 8) + A[i, 0]
        hilfe = hilfe + hilfe % 2
        i += int(hilfe / 2)
    else:
        ATRTIME.append((A[i, 1] & 3) * (2 * 8) + A[i, 0])
        ANNOT.append((A[i,1]>>2))
    i+=1
del ANNOT[len(ANNOT)-1]
del ATRTIME[len(ATRTIME)-1]

ATRTIME=np.array(ATRTIME)
ATRTIME=np.cumsum(ATRTIME)/sfreq

ind=np.where(ATRTIME<=TIME[-1][0])
ATRTIMED=ATRTIME[ind]

ANNOT=np.round(ANNOT)
ANNOTD=ANNOT[ind]
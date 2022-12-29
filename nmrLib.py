import numpy as np

# function definition with one offset
def lorentzAbsFct(f, f0, A, T2, o):
    return A * T2 / (1 + (f-f0)**2 * T2**2) + o

def lorentzDisFct(f, f0, A, T2, o):
    return -A * (f-f0) * T2**2 / (1 + (f-f0)**2 * T2**2) + o

def lorentzFct(f, f0, A, T2, p, o):
    res = lorentzAbsFct(f, f0, A, T2, o) + 1.j*lorentzDisFct(f, f0, A, T2, o)
    res = res*np.exp(-1.j*np.pi*p/180)
    return res

def lorentzFitFct(f, f0, A, T2, p, o):
    N = len(f)
    f_real = f[:N//2]
    f_imag = f[N//2:]
    y_real = np.real(lorentzFct(f_real, f0, A, T2, p, o))
    y_imag = np.imag(lorentzFct(f_imag, f0, A, T2, p, o))
    return np.hstack([y_real, y_imag])

def relaxationTime(t, P0, T2, A):
    return P0 * np.exp(-t/T2) + A

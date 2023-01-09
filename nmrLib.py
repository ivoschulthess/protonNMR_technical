import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def ratio (A, B):
    ratio = A[0] / B[0]
    ratioErr = np.sqrt((A[1]/B[0])**2 + (A[0]*B[1]/B[0]**2)**2)
    
    return np.array([ratio, ratioErr])

def weighted_mean(values, stds, axis=0):
    """
    Return the weighted mean and error of the mean.
    values, stds -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=1/stds**2, axis=axis)
    std = 1 / np.sqrt(np.sum(1/stds**2, axis=axis))
        
    return np.array([average, std])

# function definition with one offset
def lorentzAbsFct(f, f0, A, T2, o=0):
    return A * T2 / (1 + (f-f0)**2 * T2**2) + o

def lorentzDisFct(f, f0, A, T2, o=0):
    return -A * (f-f0) * T2**2 / (1 + (f-f0)**2 * T2**2) + o

def lorentzFct(f, f0, A, T2, p, o=0):
    res = lorentzAbsFct(f, f0, A, T2, o) + 1.j*lorentzDisFct(f, f0, A, T2, o)
    res = res*np.exp(-1.j*np.pi*p/180)
    return res

def lorentzFitFct(f, f0, A, T2, p, o=0):
    N = len(f)
    f_real = f[:N//2]
    f_imag = f[N//2:]
    y_real = np.real(lorentzFct(f_real, f0, A, T2, p, o))
    y_imag = np.imag(lorentzFct(f_imag, f0, A, T2, p, o))
    return np.hstack([y_real, y_imag])

def relaxationTime(t, P0, T2, A):
    return P0 * np.exp(-t/T2) + A

def nmrSignal (t, f, a, p, t0):
    return a * np.sin(2*np.pi*f*t - p*180/np.pi) * np.exp(-t/t0)

def gaussFct(f, f0, A, s, o):
    return A / np.sqrt(2*np.pi*s**2) * np.exp(-(f-f0)**2/(2*s**2)) + o

def sinExpFct (t, f, a, p, o, tau, t0):
    return a * np.sin(2*np.pi*f*t - p) * np.exp(-tau*(t-t0)) + o

def sinPhaseFct (t, a, p, o):
    return abs(a) * np.sin(np.pi/180*(t-p)) + o

def analyze_reference(file, specWidth, verbose=0, **kwargs):
    
    specWidth = 1e3*specWidth
    tdwell = 1/specWidth
    
    sig = pd.read_csv(file, skiprows=12)
    sig = (np.asarray(sig['[DATA]'][::2]) + 1.j*np.asarray(sig['[DATA]'][1::2]))
    time = np.linspace(0, len(sig)*tdwell, len(sig))
    
    if 'window' in kwargs:
        sig = sig * expWindowFct(time, kwargs.get('window'))
        
    if 'correction' in kwargs:
        correction = kwargs.get('correction')
        sig = sig * np.exp(2.j*np.pi*(correction[0]*time - correction[1]/360))
    
    freq = np.fft.fftshift(np.fft.fftfreq(len(sig), d=tdwell))
    spec = np.fft.fftshift(np.fft.fft(sig, n=len(sig), norm=None))
    noise = spec[:len(spec)//2].real.std()
    idx_max = 76

    if verbose>0:
        print('set idx_max: {}'.format(idx_max))
        print('get idx_max: {}'.format(spec.real.argmax()))

    popt = (freq[idx_max],1e3*spec[idx_max].real,0.001,0,0)
    popt = (freq[idx_max],1e3*spec[idx_max].real,0.001,0)

    try:
        popt, pcov = curve_fit(lorentzFitFct, np.hstack([freq,freq]), np.hstack([spec.real, spec.imag]), 
                               sigma=noise*np.ones(2*len(spec)), absolute_sigma=True, 
                               p0=popt, maxfev=100000)
            
        perr = np.sqrt(np.diag(pcov))
        chi2 = np.sum((np.hstack([spec.real, spec.imag])-lorentzFitFct(np.hstack([freq,freq]), *popt))**2 / (noise*np.ones(2*len(spec)))**2)
        chi2_r = chi2 / (2*len(spec) - len(popt))
    except:
        print('REFERENCE: Lorentz Fit Failed')
        plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
        plt.plot(freq/1e3, lorentzFct(freq, *popt).real, 'C0-')
        plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
        plt.plot(freq/1e3, lorentzFct(freq, *popt).imag, 'C1-')
        plt.show()
    
    if verbose>0:
        amp = popt[1]*popt[2]
        ampErr = np.sqrt(popt[1]**2*perr[2]**2 + popt[2]**2*perr[1]**2)
        print('\nREFERENCE ANALYSIS\n')
        print('peak amplitude:\t\t{:.02f} * 1e6'.format(amp/1e6))
        print('peak error:\t\t{:.02f} * 1e6'.format(ampErr/1e6))
        print('noise level f<0:\t{:.02f} * 1e6'.format(noise/1e6))
        print('signal frequency:\t{:.0f}({:.0f}) Hz'.format(popt[0], perr[0]))
        print('signal phase:\t\t{:.1f}({:.0f}) deg\n'.format(popt[3], 1e1*perr[3]))

    if verbose>1:
        print('reduced chi-squared: {:.2f}\n'.format(chi2_r))
        freqFit = np.linspace(freq.min(), freq.max(), 10001)
        fig, ax = plt.subplots()
        ax.plot(freq/1e3, spec.real, 'C0.', ms=3, lw=1)
        ax.plot(freqFit/1e3, lorentzFct(freqFit, *popt).real, 'C0-')
        ax.plot(freq/1e3, spec.imag, 'C1.', ms=3, lw=1)
        ax.plot(freqFit/1e3, lorentzFct(freqFit, *popt).imag, 'C1-')
        #ax.plot(freq/1e3, abs(spec), 'k.-', ms=3, lw=1)
        plt.show()

    return popt, perr



def analyze_signal(file, specWidth, f_ref=None, p_ref=None, t2_ref=None, o_ref=None, verbose=0):

    specWidth = 1e3*specWidth
    tdwell = 1/specWidth

    sig = pd.read_csv(file, skiprows=12)
    sig = (np.asarray(sig['[DATA]'][::2]) + 1.j*np.asarray(sig['[DATA]'][1::2]))
    time = np.linspace(0, len(sig)*tdwell, len(sig))
    freq = np.fft.fftshift(np.fft.fftfreq(len(sig), d=tdwell))
    spec = np.fft.fftshift(np.fft.fft(sig, n=len(sig), norm=None))
    noise = spec[:len(spec)//2].real.std()
    idx_max = 76

    if f_ref!=None and p_ref!=None and o_ref!=None:
        popt = (1e3*spec[idx_max].real,0.001)
        try:
            popt, pcov = curve_fit(lambda f, a, t2: lorentzFitFct(f, f_ref, a, t2, p_ref, o_ref), 
                                   np.hstack([freq,freq]), np.hstack([spec.real, spec.imag]), 
                                   sigma=noise*np.ones(2*len(spec)), absolute_sigma=True, 
                                   p0=popt, bounds=([-np.inf, 0], [np.inf, 0.002]), maxfev=100000)
            perr = np.sqrt(np.diag(pcov))
            amp = popt[0]*popt[1]
            ampErr = np.sqrt(popt[0]**2*perr[1]**2 + popt[1]**2*perr[0]**2)
        except:
            print("could not fit Lorentz resonance with fixed phase and frequency")
            plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref, o_ref).real, 'C0-')
            plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref, o_ref).imag, 'C1-')
            plt.show()
            return np.array([np.nan,np.nan])
        if verbose>1:
            print(popt)
            plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref, o_ref).real, 'C0-')
            plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref, o_ref).imag, 'C1-')
            plt.show()
        
    elif f_ref!=None and p_ref!=None and o_ref==None:
        popt = (1e3*spec[idx_max].real, 0.001, 0)
        try:
            popt, pcov = curve_fit(lambda f, a, t2, o: lorentzFitFct(f, f_ref, a, t2, p_ref, o), 
                                   np.hstack([freq,freq]), np.hstack([spec.real, spec.imag]), 
                                   sigma=noise*np.ones(2*len(spec)), absolute_sigma=True, 
                                   p0=popt, bounds=([-np.inf, 0, -np.inf], [np.inf, 0.002, np.inf]), maxfev=100000)
            perr = np.sqrt(np.diag(pcov))
            amp = popt[0]*popt[1]
            ampErr = np.sqrt(popt[0]**2*perr[1]**2 + popt[1]**2*perr[0]**2)
        except:
            print("could not fit Lorentz resonance with fixed phase and frequency")
            plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref).real, 'C0-')
            plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref).imag, 'C1-')
            plt.show()
            return np.array([np.nan,np.nan])
        if verbose>1:
            print(popt)
            plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref).real, 'C0-')
            plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
            plt.plot(freq/1e3, lorentzFct(freq, f_ref, *popt, p_ref).imag, 'C1-')
            plt.show()
    else:
        popt = (freq[idx_max],1e3*spec[idx_max].real,0.001,0)
        try:
            popt, pcov = curve_fit(lorentzFitFct, np.hstack([freq,freq]), np.hstack([spec.real,spec.imag]), p0=popt, maxfev=100000)
            perr = np.sqrt(np.diag(pcov))
            amp = popt[1]*popt[2]
            ampErr = np.sqrt(popt[1]**2*perr[2]**2 + popt[2]**2*perr[1]**2)
        except:
            print("could not fit Lorentz resonance")
            return np.array([np.nan,np.nan])

    if verbose>0:
        print('\nSIGNAL ANALYSIS\n')
        print('peak amplitude:\t\t{:.02f} * 1e6'.format(amp/1e6))
        print('peak error:\t\t{:.02f} * 1e6'.format(ampErr/1e6))
        print('noise level f<0:\t{:.02f} * 1e6'.format(noise/1e6))


    if ampErr==0:
        return np.array([amp,noise])
    else:
        return np.array([amp,ampErr])
    
    
def anaStab_ref(file):
    
    specWidth = 128e3
    tdwell = 1/specWidth
    
    sig = pd.read_csv(file, skiprows=12)
    sig = (np.asarray(sig['[DATA]'][::2]) + 1.j*np.asarray(sig['[DATA]'][1::2]))# new / 1e7
    time = np.linspace(0, len(sig)*tdwell, len(sig))
    
    freq = np.fft.fftshift(np.fft.fftfreq(len(sig), d=tdwell))
    spec = np.fft.fftshift(np.fft.fft(sig, n=len(sig), norm=None))
    noise = spec[:len(spec)//2].real.std()
    
    idx_max = 76
    popt = (freq[idx_max],1e3*spec[idx_max].real,0.001,0)
    
    try:
        popt, pcov = curve_fit(lorentzFitFct, np.hstack([freq,freq]), np.hstack([spec.real, spec.imag]), 
                               sigma=noise*np.ones(2*len(spec)), absolute_sigma=True,
                               p0=popt, maxfev=100000)
        perr = np.sqrt(np.diag(pcov))
    
    except:
        print('REFERENCE: Lorentz Fit Failed')
        plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
        plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
        plt.show()
    
    return popt, perr


def anaStab_sig(file, f0, p0):
    
    specWidth = 128e3
    tdwell = 1/specWidth
    
    sig = pd.read_csv(file, skiprows=12)
    sig = (np.asarray(sig['[DATA]'][::2]) + 1.j*np.asarray(sig['[DATA]'][1::2])) / 1e7
    time = np.linspace(0, len(sig)*tdwell, len(sig))
    
    freq = np.fft.fftshift(np.fft.fftfreq(len(sig), d=tdwell))
    spec = np.fft.fftshift(np.fft.fft(sig, n=len(sig), norm=None))
    noise = spec[:len(spec)//2].real.std()

    popt = np.array([1e4,0.001])
    perr = np.zeros_like(popt)
    
    try:
        popt, pcov = curve_fit(lambda f, a, t2: lorentzFitFct(f, f0, a, t2, p0), 
                               np.hstack([freq,freq]), np.hstack([spec.real, spec.imag]), 
                               sigma=noise*np.ones(2*len(spec)), absolute_sigma=True,
                               p0=popt, maxfev=100000)
        perr = np.sqrt(np.diag(pcov))
    
    except:
        print('Lorentz Fit Failed')
        plt.plot(freq/1e3, spec.real, 'C0.', ms=3)
        plt.plot(freq/1e3, spec.imag, 'C1.', ms=3)
        plt.show()
        popt[:] = np.nan; perr[:] = np.nan
    
    return popt, perr

def plotRabiFrequencyScan (path, **kwargs):
    
    # reference
    popt, perr = analyze_reference(path+'reference.txt', 128)
    f_ref = popt[0], perr[0]
    a_ref = popt[1]*popt[2], np.sqrt(popt[1]**2*perr[2]**2 + popt[2]**2*perr[1]**2)
    p_ref = popt[3], perr[3]
    o_ref = popt[4], perr[4]
    
    if 'lc' in kwargs:
        lc = kwargs.get('lc')
    else:
        lc = ''
        
    if 'ms' in kwargs:
        ms = float(kwargs.get('ms'))
    else:
        ms = None
        
    if 'marker' in kwargs:
        marker = kwargs.get('marker')
    else:
        marker = '.'
        
    if 'label' in kwargs:
        label = kwargs.get('label')
    else:
        label = ''
        
    if 'p0' in kwargs:
        popt = kwargs.get('p0')
    else:
        popt = (500, -100, 50, 1)
        
    if 'center' in kwargs:
        center = kwargs.get('center')
    else:
        center = False
    
    # signal
    data = np.load(path+'rabiFrequencyScan.npz')
    F_SF = data['F_SF']
    F_Fit = np.linspace(F_SF[0], F_SF[-1], 1001)
    Amp = np.zeros((2, len(F_SF)))
    for i,f_sf in enumerate(F_SF):
        Amp[:,i] = analyze_signal(path+'nmrSignal_{:02d}.txt'.format(i), 128, f_ref=f_ref[0], p_ref=p_ref[0], o_ref=o_ref[0], verbose=0)
    Amp = Amp[0]/a_ref[0], np.sqrt(Amp[1]**2/a_ref[0]**2 + Amp[0]**2*a_ref[1]**2/a_ref[0]**4)
    popt, pcov = curve_fit(gaussFct, F_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=popt)
    perr = np.sqrt(np.diag(pcov))    
    chi2 = np.sum((Amp[0]-gaussFct(F_SF, *popt))**2 / Amp[1]**2)
    chi2_r = chi2 / (len(Amp[0]) - len(popt))
    print('resonance at {:.1f}({:.0f}) Hz'.format(popt[0], 1e1*perr[0]))
    print('resonance width {:.1f}({:.0f}) Hz'.format(abs(popt[2]), 1e1*perr[2]))
    print('FWHM {:.1f}({:.0f}) Hz'.format(2.355*abs(popt[2]), 1e1*2.355*perr[2]))
    print('reduced chi-squared: {:.2f}\n'.format(chi2_r))
    
    if center==True:
        f0 = popt[0]
    else:
        f0 = 0
        
        
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        ax.errorbar(F_SF-f0, Amp[0], Amp[1], fmt='{}{}'.format(lc,marker), ms=ms, lw=1, label=label)
        ax.plot(F_Fit-f0, gaussFct(F_Fit, *popt), '{}-'.format(lc), lw=1)
    
    return popt,perr


def plotRabiAmplitudeScan (path, ax, **kwargs):
    
    # reference
    popt, perr = analyze_reference(path+'reference.txt', 128)
    f_ref = popt[0], perr[0]
    a_ref = popt[1]*popt[2], np.sqrt(popt[1]**2*perr[2]**2 + popt[2]**2*perr[1]**2)
    p_ref = popt[3], perr[3]
    o_ref = popt[4], perr[4]
    
    if 'lc' in kwargs:
        lc = kwargs.get('lc')
    else:
        lc = ''
        
    if 'ms' in kwargs:
        ms = float(kwargs.get('ms'))
    else:
        ms = None
        
    if 'marker' in kwargs:
        marker = kwargs.get('marker')
    else:
        marker = '.'
        
    if 'label' in kwargs:
        label = kwargs.get('label')
    else:
        label = ''
        
    if 'p0' in kwargs:
        popt = kwargs.get('p0')
    else:
        popt = (0.002,1,0,0.001,100) 
        
    # signal
    data = np.load(path+'rabiAmplitudeScan.npz')
    A_SF = data['A_SF']
    A_Fit = np.linspace(A_SF[0], 1000, 1001)
    Amp = np.zeros((2, len(A_SF)))
    for i,a_sf in enumerate(A_SF):
        Amp[:,i] = analyze_signal(path+'nmrSignal_{:02d}.txt'.format(i), 128, f_ref=f_ref[0], p_ref=p_ref[0], o_ref=None, verbose=0)
    Amp = Amp[0]/a_ref[0], np.sqrt(Amp[1]**2/a_ref[0]**2 + Amp[0]**2*a_ref[1]**2/a_ref[0]**4)
    mask = (Amp[0]!=0) & (A_SF<=1000)
    ax.errorbar(A_SF[mask], Amp[0][mask], Amp[1][mask], fmt='{}{}'.format(lc,marker), ms=ms, lw=1, label=label)
    popt, pcov = curve_fit(sinExpFct, A_SF[mask], Amp[0][mask], sigma=Amp[1][mask], absolute_sigma=True, p0=popt)
    perr = np.sqrt(np.diag(pcov))
    chi2 = np.sum((Amp[0][mask]-sinExpFct(A_SF[mask], *popt))**2 / Amp[1][mask]**2)
    chi2_r = chi2 / (len(Amp[0][mask]) - len(popt))
    print('pi/2 flip at {:.1f}({:.0f}) mVpp'.format(1/4/popt[0], 1e1*perr[0]/4/popt[0]**2))
    print('reduced chi-squared: {:.2f}\n'.format(chi2_r))
    ax.plot(A_Fit, sinExpFct(A_Fit, *popt), '{}-'.format(lc), lw=1)
    
    return popt, perr
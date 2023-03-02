from sunpy.map import Map
from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

def multifunc(x, a0, b0, c0, a1, b1, c1):
    return a0*np.exp(-(x-b0)**2/(2*c0**2)) + a1*np.exp(-(x-b1)**2/(2*c1**2))


minmax=100

class FitGaussian(object):
    def __call__(self, data, minmax):
        tmp = np.histogram(data.flatten(), bins=np.linspace(-minmax, minmax, 2*minmax+1))
        x = np.linspace(-minmax+0.5, minmax-0.5, 2*minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return popt, tmp
#        return amp, loc, scale

class FitMultiGaussian(object):
    def __call__(self, data, minmax):
        tmp = np.histogram(data.flatten(), bins=np.linspace(-minmax, minmax, 2*minmax+1))
        x = np.linspace(-minmax+0.5, minmax-0.5, 2*minmax)
        popt, _ = curve_fit(multifunc, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return popt, tmp



f = "hmi.M_45s.2011-01-01-00-00-00.fits"
g = "hmi.M_720s.2011-01-01-00-00-00.fits"
h = "hmi.vector.2011-01-01-00-00-00.sav"


sav = readsav(h)

m45 = Map(f).data[2048-512:2048+512, 2048-512:2048+512]
m720 = Map(g).data[2048-512:2048+512, 2048-512:2048+512]

br = sav["BR"].copy()[2048-512:2048+512, 2048-512:2048+512]
bt = sav["BT"].copy()[2048-512:2048+512, 2048-512:2048+512]
bp = sav["BP"].copy()[2048-512:2048+512, 2048-512:2048+512]

print(br.shape)
print(bt.shape)
print(bp.shape)



def plot(data, minmax):
    x = np.linspace(-minmax+0.5, minmax-0.5, 2*minmax)
    popt, tmp = FitGaussian()(data, minmax)
    print(popt)
    y = func(x, *popt)
    plt.plot(x, y, 'r-')
    plt.plot(x, tmp[0], 'b-')
    plt.show()

def multiplot(data, minmax):
    x = np.linspace(-minmax+0.5, minmax-0.5, 2*minmax)
    popt, tmp = FitMultiGaussian()(data, minmax)
    print(popt)
    y = multifunc(x, *popt)
    plt.plot(x, y, 'r-')
    plt.plot(x, tmp[0], 'b-')
    plt.show()

plot(m45, minmax)
plot(m720, minmax)
plot(br, minmax)
multiplot(bt, minmax)
multiplot(bp, minmax)



#img = np.hstack([m45, m720, br, bt, bp])
#plt.imshow(img, vmin=-30, vmax=30, cmap="gray")
#plt.show()


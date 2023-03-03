from sunpy.map import Map
from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))


class FitGaussian(object):
    def __init__(self, minmax=100):
        self.minmax=minmax
    def __call__(self, data):
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return amp, loc, scale, tmp


class FitMultiGaussian(object):
    def __init__(self, minmax=100):
        self.minmax=minmax

    def fit_negative(self, data_):
        w = np.where(data_ <= 0.)
        data = data_[w]
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return amp, loc, scale, tmp

    def fit_positive(self, data_):
        w = np.where(data_ >= 0.)
        data = data_[w]
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return amp, loc, scale, tmp

    def __call__(self, data):
        popt_negative = self.fit_negative(data)
        popt_positive = self.fit_positive(data)
        return popt_negative, popt_positive
    
def plot(data, minmax):
    x = np.linspace(-minmax+0.5, minmax-0.5, minmax)
    result = FitGaussian()(data)
    popt = (result[0], result[1], result[2])
    tmp = result[3]
    print(popt)
    y = func(x, *popt)
    plt.plot(x, y, 'r-')
    plt.plot(x, tmp[0], 'b-')
    plt.show()


def multiplot(data, minmax):
    x = np.linspace(-minmax+0.5, minmax-0.5, minmax)
    result = FitMultiGaussian(minmax)(data)
    result_negative = result[0]
    result_positive = result[1]
    popt_negative = (result_negative[0], result_negative[1], result_negative[2])
    tmp_negative = result_negative[3]
    popt_positive = (result_positive[0], result_positive[1], result_positive[2])
    tmp_positive = result_positive[3]
    print(popt_negative)
    print(popt_positive)
    y_negative = func(x, *popt_negative)
    y_positive = func(x, *popt_positive)
    plot = plt.plot(x, y_negative, 'r-')
    plt.plot(x, y_positive, 'r--')
    plt.plot(x, tmp_negative[0], 'b-')
    plt.plot(x, tmp_positive[0], 'b--')
    return plot



if __name__ == "__main__" :
    from glob import glob
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt


    list_ = sorted(glob('/data/eunsu/Dataset/denoising/B_720s/train/*.sav'))
    name_data = 'BT'
    minmax = 100

    imgs = []

    fig = plt.figure()
    for idx in range(len(list_)):
        file_ = list_[idx]
        data = readsav(file_)[name_data].copy()[2048-1024:2048+1024, 2048-1024:2048+1024]
        img = multiplot(data, minmax)
        imgs.append([img])

        if idx == 1000 :
            break

    animate = animation.ArtistAnimation(fig, imgs, interval=500, blit=True)
    animate.save('%s.gif' % (name_data))
    plt.close()


    
    




# f = "hmi.M_45s.2011-01-01-00-00-00.fits"
# g = "hmi.M_720s.2011-01-01-00-00-00.fits"
# h = "hmi.vector.2011-01-01-00-00-00.sav"


# sav = readsav(h)

# m45 = Map(f).data[2048-512:2048+512, 2048-512:2048+512]
# m720 = Map(g).data[2048-512:2048+512, 2048-512:2048+512]

# br = sav["BR"].copy()[2048-512:2048+512, 2048-512:2048+512]
# bt = sav["BT"].copy()[2048-512:2048+512, 2048-512:2048+512]
# bp = sav["BP"].copy()[2048-512:2048+512, 2048-512:2048+512]

# print(br.shape)
# print(bt.shape)
# print(bp.shape)


# plot(m45, minmax)
# plot(m720, minmax)
# plot(br, minmax)
# multiplot(bt, minmax)
# multiplot(bp, minmax)



#img = np.hstack([m45, m720, br, bt, bp])
#plt.imshow(img, vmin=-30, vmax=30, cmap="gray")
#plt.show()


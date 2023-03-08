import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec


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
        return (amp, loc, scale), tmp


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
        return (amp, loc, scale), tmp

    def fit_positive(self, data_):
        w = np.where(data_ >= 0.)
        data = data_[w]
        tmp = np.histogram(data.flatten(), bins=np.linspace(-self.minmax, self.minmax, self.minmax+1))
        x = np.linspace(-self.minmax+0.5, self.minmax-0.5, self.minmax)
        popt, _ = curve_fit(func, x, tmp[0])
        amp = popt[0]
        loc = popt[1]
        scale = abs(popt[2])
        return (amp, loc, scale), tmp

    def __call__(self, data):
        popt_negative, tmp_negative = self.fit_negative(data)
        popt_positive, tmp_positive = self.fit_positive(data)
        return popt_negative, tmp_negative, popt_positive, tmp_positive
    
def plot(data, minmax):
    x = np.linspace(-minmax+0.5, minmax-0.5, minmax)
    popt, tmp = FitGaussian()(data)
    print(popt)
    y = func(x, *popt)
    plt.plot(x, y, 'r-')
    plt.plot(x, tmp[0], 'b-')
    plt.show()


def multiplot(data, minmax):
    x = np.linspace(-minmax+0.5, minmax-0.5, minmax)
    result = FitMultiGaussian(minmax)(data)
    popt_negative, tmp_negative, popt_positive, tmp_positive = result
    amp = popt_negative[0] + popt_positive[0]
    loc = popt_negative[1] + popt_positive[1]
    scale = (abs(popt_negative[1]) + abs(popt_positive[1]))/2.
#    scale = ((abs(popt_negative[1]) + popt_negative[2]) + (abs(popt_positive[1]) + popt_positive[2]))/2.
#    scale = (popt_negative[2] + popt_positive[2])/2.
    print(popt_negative)
    print(popt_positive)
    print(amp, loc, scale)
    y_negative = func(x, *popt_negative)
    y_positive = func(x, *popt_positive)
    y_ = y_negative + y_positive
    y__ = func(x, *(amp, loc, scale))
    plt.plot(x, y_negative, 'r-')
    plt.plot(x, y_positive, 'r--')
    plt.plot(x, y_, 'g-')
    plt.plot(x, y__, 'k-')
    plt.plot(x, tmp_negative[0], 'b-')
    plt.plot(x, tmp_positive[0], 'b--')
    plt.show()



def reader(f, proc_nan=False):
    data = np.load(f)["data"][2048-1024:2048+1024, 2048-1024:2048+1024]
    if proc_nan == True :
        data[np.isnan(data)] = 0.
    return data

def analyze_los_45(f):
    data = reader(f)


def imshow(data, vmin, vmax, title):
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap="gray")
    plt.title(title)
    #plt.show()

def histshow(x, hist, title):
    plt.plot(x, hist)
    plt.title(title)
    #plt.show()


def custom_plot(data, hist, x, title, vmin, vmax):
    fig = plt.figure(figsize=(20, 8))
    plt.suptitle(title)
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1:])
    ax1.imshow(data, vmin=vmin, vmax=vmax, cmap="gray")
    ax1.set_title("Image")
    ax2.plot(x, hist)
    ax2.set_title("Histogram")
    plt.tight_layout()


def analyze_inclination(f, proc_nan=False):
    data = reader(f, proc_nan=proc_nan)
    x = np.linspace(0, 180, 180)
    hist = np.histogram(data.flatten(), bins=np.linspace(0, 180, 181))
    custom_plot(data, hist[0], x, title="Inclination", vmin=0, vmax=180)
    plt.savefig("inclination.png", dpi=200)
    plt.close()
    return data, hist[0], x

def analyze_azimuth(f, proc_nan=False):
    data = reader(f, proc_nan=proc_nan)
    x = np.linspace(0, 180, 180)
    hist = np.histogram(data.flatten(), bins=np.linspace(0, 180, 181))
    custom_plot(data, hist[0], x, title="Azimuth", vmin=0, vmax=180)
    plt.savefig("azimuth.png", dpi=200)
    plt.close()
    return data, hist[0], x

def analyze_disambig(f, proc_nan=False):
    data = reader(f, proc_nan=proc_nan)
    x = np.linspace(0, 10, 10)
    hist = np.histogram(data.flatten(), bins=np.linspace(0, 10, 11))
    custom_plot(data, hist[0], x, title="Disambig", vmin=0, vmax=10)
    plt.savefig("disambig.png", dpi=200)
    plt.close()
    return data, hist[0], x

def analyze_field(f, proc_nan=False):
    data = reader(f, proc_nan=proc_nan)
    x = np.linspace(0, 3000, 3000)
    hist = np.histogram(data.flatten(), bins=np.linspace(0, 3000, 3001))
    custom_plot(data, hist[0], x, title="Field", vmin=0, vmax=3000)
    plt.savefig("field.png", dpi=200)
    plt.close()
    return data, hist[0], x




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
        return (amp, loc, scale), tmp

if __name__ == "__main__" :
    from glob import glob
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    f_los_45 = "hmi.2011-01-01-00-00-00.los_45.npz"
    f_los_720 = "hmi.2011-01-01-00-00-00.los_720.npz"
    f_inclination = "hmi.2011-01-01-00-00-00.inclination.npz"
    f_azimuth = "hmi.2011-01-01-00-00-00.azimuth.npz"
    f_disambig = "hmi.2011-01-01-00-00-00.disambig.npz"
    f_field = "hmi.2011-01-01-00-00-00.field.npz"

    data, hist, x = analyze_inclination(f_inclination)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    print(vmin, vmax)

    data, hist, x = analyze_azimuth(f_azimuth)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    print(vmin, vmax)

    data, hist, x = analyze_disambig(f_disambig)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    print(vmin, vmax)    

    data, hist, x = analyze_field(f_field)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    print(vmin, vmax)


    # inclination = np.load(f_inclination)["data"][2048-1024:2048+1024, 2048-1024:2048+1024]
    # field = np.load(f_field)["data"][2048-1024:2048+1024, 2048-1024:2048+1024]

    # x = np.linspace(0, 180, 180)
    # tmp = inclination[np.abs(field) > 1000]
    # hist = np.histogram(tmp.flatten(), bins=np.linspace(0, 180, 181))
    # plt.plot(x, hist[0])
    # plt.show()


    # sav = readsav(h)

    # m45 = Map(f).data[2048-512:2048+512, 2048-512:2048+512]
    # m720 = Map(g).data[2048-512:2048+512, 2048-512:2048+512]

    # br = sav["BR"].copy()[2048-512:2048+512, 2048-512:2048+512]
    # bt = sav["BT"].copy()[2048-512:2048+512, 2048-512:2048+512]
    # bp = sav["BP"].copy()[2048-512:2048+512, 2048-512:2048+512]

    # print(br.shape)
    # print(bt.shape)
    # print(bp.shape)


    # # plot(m45, minmax)
    # # plot(m720, minmax)
    # # plot(br, minmax)
    # multiplot(bt, minmax)
    # multiplot(bp, minmax)



    # # img = np.hstack([m45, m720, br, bt, bp])
    # # plt.imshow(img, vmin=-30, vmax=30, cmap="gray")
    # # plt.show()


    # # imgs = []

    # # fig = plt.figure()
    # # for idx in range(len(list_)):
    # #     file_ = list_[idx]
    # #     data = readsav(file_)[name_data].copy()[2048-1024:2048+1024, 2048-1024:2048+1024]

    # #     x = np.linspace(-minmax+0.5, minmax-0.5, minmax)
    # #     result = FitMultiGaussian(minmax)(data)
    # #     result_negative = result[0]
    # #     result_positive = result[1]
    # #     popt_negative = (result_negative[0], result_negative[1], result_negative[2])
    # #     tmp_negative = result_negative[3]
    # #     popt_positive = (result_positive[0], result_positive[1], result_positive[2])
    # #     tmp_positive = result_positive[3]
    # #     print(popt_negative)
    # #     print(popt_positive)
    # #     y_negative = func(x, *popt_negative)
    # #     y_positive = func(x, *popt_positive)
    # #     plot = plt.plot(x, y_negative, 'r-', animated=True)
    # #     plt.plot(x, y_positive, 'r--')
    # #     plt.plot(x, tmp_negative[0], 'b-')
    # #     plt.plot(x, tmp_positive[0], 'b--')

    # #     plt.show()

    # #     img = multiplot(data, minmax)
    # #     imgs.append([img])

    # #     if idx == 100 :
    # #         break

    # # animate = animation.ArtistAnimation(fig, imgs, interval=500, blit=True)
    # # animate.save('%s.gif' % (name_data))
    # # plt.close()


    
    





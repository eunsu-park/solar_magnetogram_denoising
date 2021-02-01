import numpy as np

# Linear rescale
def rescale(data, imin, imax, omin, omax):
    odif = omax-omin
    idif = imax-imin
    data = (data-imin)*(odif/idif) + omin
    return data.clip(omin, omax)

def bytescale(data, imin=None, imax=None):
    if not imin:
        imin = np.min(data)
    if not imax:
        imax = np.max(data)
    data = rescale(data, imin, imax, omin=0, omax=255)
    return data.astype(np.uint8)

# Maybe almost same function to aia_intscale.pro in SSW.
class aia_intscale():
    def __init__(self, wavelnth):
        list_aia = [94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500]
        self.wavelnth = str(int(wavelnth))
        if int(self.wavelnth) not in list_aia:
            raise ValueError('%d is invalid AIA wavelength'%int(self.wavelnth))
    def aia_rescale(self, data):
        if self.wavelnth == '94':
            data = np.sqrt((data*4.99803).clip(1.5, 50.))
        elif self.wavelnth == '131':
            data = np.log10((data*6.99685).clip(7.0, 1200.))
        elif self.wavelnth == '171':
            data = np.sqrt((data*4.99803).clip(10., 6000.))
        elif self.wavelnth == '193':
            data = np.log10((data*2.99950).clip(120., 6000.))
        elif self.wavelnth == '211':
            data = np.log10((data*4.99801).clip(30., 13000.))
        elif self.wavelnth == '304':
            data = np.log10((data*4.99941).clip(50., 2000.))
        elif self.wavelnth == '335':
            data = np.log10((data*6.99734).clip(3.5, 1000.))
        elif self.wavelnth == '1600':
            data = (data*2.99911).clip(0., 1000.)
        elif self.wavelnth == '1700':
            data = (data*1.00026).clip(0., 2500.)
        elif self.wavelnth == '4500':
            data = (data*1.00026).clip(0., 26000.)
        data = bytescale(data)
        return data
    def __call__(self, data):
        data = self.aia_rescale(data)
        return data

class shake_tensor:
    def __init__(self, isize, pad=None):
        self.isize = isize
        self.pad = pad if pad else  self.isize//64-1
        self.range = 2*self.pad+1
        print(self.pad, self.range)
    def do(self, A, B):
        A = np.pad(A, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant', constant_values=-1)
        B = np.pad(B, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant', constant_values=-1)
        x, y = np.random.randint(self.range), np.random.randint(self.range)
        A = A[:, x:x+self.isize, y:y+self.isize, :]
        B = B[:, x:x+self.isize, y:y+self.isize, :]
        return A, B
    def __call__(self, batch_A, batch_B):
        A, B = self.do(batch_A, batch_B)
        return A, B
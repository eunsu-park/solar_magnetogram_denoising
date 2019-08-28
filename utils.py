import numpy as np

class make_tensor():
    def __init__(self, isize, drange):
        self.isize = isize
        self.drange = drange
    def __call__(self, file_):
        x = np.load(file_)
        x = x.clip(-self.drange, self.drange)/self.drange
        x.shape = (1, self.isize, self.isize, 1)
        return x.astype(np.float32)
    
    
class make_output():
    def __init__(self, isize, drange):
        self.isize = isize
        self.drange = drange
    def __call__(self, output):
        output.shape = (self.isize, self.isize)
        output_npy = (output*self.drange).clip(-self.drange, self.drange)
        output_png = (((output*self.drange) + 30.)*(255./60.)).clip(0, 255)
        return output_npy.astype(np.float32), output_png.astype(np.uint8)

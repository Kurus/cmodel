import scipy.io
import numpy as np

weights_raw = scipy.io.loadmat("sqz_full.mat")

for name in weights_raw:
    # skipping '__version__', '__header__', '__globals__'
    if name[0:2] != '__':
        kernels, bias = weights_raw[name][0]
        kernels = kernels.astype(np.float32)
        bias = bias.astype(np.float32)
        kernels = np.moveaxis(kernels,-1,0)
        name = name.replace("/","_")
        np.savetxt("wei/"+name+"_bias",bias.flatten())
        np.savetxt("wei/"+name+"_ker",kernels.flatten())


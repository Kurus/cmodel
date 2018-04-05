import scipy.io
import numpy as np
import sys

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

path = sys.argv[1]
print(path)

img_orig = scipy.misc.imread(path)
img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
if len(img.shape) == 2:
    # grayscale
    img = np.dstack((img,img,img))
mean_pixel = np.array([104.006, 116.669, 122.679])
img=preprocess(img,mean_pixel)
np.savetxt("wei/image",img.flatten())

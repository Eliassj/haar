import numpy as np
from skimage import io
from skimage.util import img_as_ubyte

picarray = io.imread('kvinna.jpg', as_gray=True)
picarray = img_as_ubyte(picarray)

def haar_matrix(size:int, inverse = False):
    a = np.zeros((size, size))
    rt2d2 = np.sqrt(2)/2
    indterm = int(size/2)
    for r, c in zip(range(0, int(size/2)), range(0, int(size), 2)):
        a[r, (c, c+1)] = 1
        a[r+indterm, c] = -1
        a[r+indterm, c+1] = 1
    a = a * rt2d2
    if inverse: # Return inverse matrix if specified
        return np.transpose(a)
    else:
        return a

def haartransform(imarray, split = True):
    m = imarray.shape[0]
    n = imarray.shape[1]
    if not m % 2 == 0:
        imarray = imarray[:-1]
        m -= 1
    if not n % 2 == 0:
        imarray = imarray[:,:-1]
        n -= 1
        
    marr = haar_matrix(m)
    narr = haar_matrix(n).transpose()
    
    transformed = np.matmul(marr, imarray)
    transformed = np.matmul(transformed, narr)
    
    if split:
        split = np.split(transformed, 2, 0)
        split = np.array([np.split(x, 2, 1) for x in split])
        return split
    else:
        return transformed

def inverseHaarTransformation(arrs):
    imarray = np.vstack((np.concatenate(arrs[0,:], axis=1), np.concatenate(arrs[1,:], axis=1)))
    m = imarray.shape[0]
    n = imarray.shape[1]
    if not m % 2 == 0:
        imarray = imarray[:-1]
        m -= 1
    if not n % 2 == 0:
        imarray = imarray[:,:-1]
        n -= 1
    
    marr = haar_matrix(m, True)
    narr = haar_matrix(n, True).transpose()
    
    transformed = np.matmul(marr, imarray)
    transformed = np.matmul(transformed, narr)
    
    return transformed

def saveim(arr, maxlum, path):
    arr = arr*(maxlum/np.max(arr)) # Scale the image to the max luminosity of the original
    arr = np.round(arr).astype(np.uint8) # Convert to uint8
    io.imsave(path, arr)

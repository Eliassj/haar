import numpy as np
from skimage import io
import sys
import haar
from skimage.util import img_as_ubyte


def main():
    image = io.imread(str(sys.argv[1]), as_gray = True)
    image = img_as_ubyte(image)
    maxlum = np.max(image)
    
    trans = haar.haartransform(image)
    print(str(sys.argv[1])+"first transform")
    if int(sys.argv[2]) > 1:
        for i in range(int(sys.argv[2])-1):
            trans = haar.haartransform(trans[0,0])
        
    haar.saveim(trans[0,0], maxlum, sys.argv[1]+"_compressed.jpg")
        
        
main()
from skimage.measure import label, regionprops
import numpy as np
import cv2
import torch 

def getProps(image):
    num,im = cv2.connectedComponents(image)
    im = np.uint8(im)
    if num == 1:
        return torch.tensor([0,0,0,0,0,0,0])
    elif num == 2:
        props = regionprops(image)
        ret = [props[0].area,props[0].convex_area,props[0].eccentricity,props[0].equivalent_diameter,props[0].orientation,props[0].perimeter,props[0].solidity]
        return torch.tensor(ret)
    elif num>2:
        numMayor = 0
        mayor = 1
        for i in range(1,num):
            conteo = len(im[im==i])
            if conteo>numMayor:
                numMayor = conteo
                mayor = i
        im[im!=mayor]=0
        im[im==mayor]=1
        props = regionprops(im)
        ret = [props[0].area,props[0].convex_area,props[0].eccentricity,props[0].equivalent_diameter,props[0].orientation,props[0].perimeter,props[0].solidity]
        return torch.tensor(ret)


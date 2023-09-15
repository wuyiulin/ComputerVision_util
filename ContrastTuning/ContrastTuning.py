import pdb
import time
import cv2
import numpy as np
import math

def ContrastTuning(img_path, contrast=0, brightness=0, f_BP=None):
    img = cv2.imread(img_path)
    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    f_RGB = np.array([0.59, 0.3, 0.11]) 
    if(not f_BP): f_BP = 8
    f = np.power(2, f_BP) * f_RGB
    grey = np.matmul(img, f).astype(np.uint64)
    grey = (grey >> f_BP).astype(np.uint8)
    iMax = grey.max()
    imin = grey.min()
    Cm = (iMax + imin) / 2
    img = (img - Cm * (1 - B)) * k + Cm * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)

    
    cv2.namedWindow("Contrast", 0)
    cv2.resizeWindow("Contrast", int(img.shape[1]/2), int(img.shape[0]/2))
    cv2.imshow('Contrast', img)
    
    while True:
        if cv2.getWindowProperty('Contrast', cv2.WND_PROP_VISIBLE) <= 0: 
            break
        cv2.waitKey(1)
    cv2.destroyWindow('Contrast')


    



if __name__=='__main__':
    img_path = 'Your_image_path'
    ContrastTuning(img_path)
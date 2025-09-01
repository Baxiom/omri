#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:34:29 2017
Small change - Sept 1 2025 !!
@author: maris
"""
import numpy as np
from PIL import Image

RGB_BW_VECTOR = np.array([0.3, 0.5, 0.2])

#Convert to greyscale.
#Can probably use Image.convert(mode="L") just as well
def toGrey2(im_array):
    return np.tensordot(im_array, RGB_BW_VECTOR, axes = ([2], [0])) 
    
#Display 2-d array (values 0-255) as "L" mode greyscale image.
#Handy for quick looking at an image.
def display(imArray):
    Image.fromarray(np.uint8(imArray), mode="L").show()
  
def displayRB(im1, im2):
    z = np.zeros(im1.shape)
    array = np.dstack([np.uint8(im1), np.uint8(z), np.uint8(im2)])
    Image.fromarray(array).show()
    
def displayRGB(im1, im2, im3):
    array = np.dstack([np.uint8(im1), np.uint8(im2), np.uint8(im3)])
    Image.fromarray(array).show()
      
    
#  ----------------------------------
#Test data:
mFile = Image.open("../samples/music.JPG")

#mFile.show()
width, height = mFile.size

#Skinny
skinny = mFile.crop((1000, 0, 1250, width))
#skinny.show()

linearBig = skinny.transpose(Image.ROTATE_270)
    
greyBig = toGrey2(linearBig)
#display(greyBig)

greyBig
np.amax(mFile)

#-----------------------

# Some utes for relatively global analysis of feature H or V correlation distance.
# The idea is given binary input arrays (effectively 1d - either row or columns)
# To build up the specrum/time series model of how soon after A event the next B occurs.
# And how soon after B, A occurs

def jointSpectrum(A, B):
    aToB = np.zeros(A.shape)
    i = 0
    aWaiting = 0
    for i in range(A.shape[0]):
        if aWaiting > 0:
            if B[i] > 0:
                aToB[aWaiting] += 1
                aWaiting = 0
            else: #reset if A again
                aWaiting = 1 if A[i] > 0 else aWaiting + 1
        else:
            if A[i] > 0:
                if B[i] > 0:
                    aToB[0] += 1
                else:
                    aWaiting = 1
        i += 1
    return aToB
    
#test
t1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
t2 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
#analyseBtoT(t1, t2)
jointSpectrum(t1, t2)


        


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 12:34:29 2017

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
  
    
    
    
#  ----------------------------------
#Test data:
mFile = Image.open("music.JPG")
#mFile.show()
width, height = mFile.size

#Skinny
skinny = mFile.crop((1000, 0, 1250, width))
skinny.show()

linearBig = skinny.transpose(Image.ROTATE_270)
    
greyBig = toGrey2(linearBig)
display(greyBig)

greyBig
np.amax(mFile)


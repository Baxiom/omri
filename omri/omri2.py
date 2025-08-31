#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:22:08 2024

@author: marisozols
The idea here is to just get the good bits from the hge omri experiment that I have!
"""


#import PIL
from PIL import Image
import numpy as np
import utils

mFile = Image.open("../samples/music.JPG")
#mFile.show()
width, height = mFile.size
musFile = mFile.transpose(Image.ROTATE_270)

mGrey2 = utils.toGrey2(musFile)
#utils.display(mGrey2)
#linArray = toArray(mGrey2)
#linArray.shape


# SHIFTING
def xshift(arr, m):
    """Shift the array down (with positive m) or up (negative m).
    the extra lines will be repeats of the top -or bottommost lines, respectively
    """
    res = arr
    h,w = arr.shape
    n = - m
    if n < 0:
        filler = np.repeat(arr[0,], -n, axis = 0)
        filler = np.transpose(filler.reshape(w, -n))
        res = np.vstack([filler, arr[:n,]])
        #res = np.vstack([np.repeat(arr[0,], -n, axis = 0), arr[:n,]])
    elif (n > 0):
        filler = np.repeat(arr[-1,], n, axis = 0)
        filler = np.transpose(filler.reshape(w, -n))
        res = np.vstack([arr[n:,], filler])
        #res = np.vstack([arr[n:,], np.repeat(arr[-1,], n, axis = 0)])
    return res
        
def yshift(arr, n):
    """Shift the array right (with positive m) or left (negative m).
    the extra lines will be repeats of the left- or rightmost lines, respectively
    """
    return np.transpose(xshift(np.transpose(arr), n))

def crop(inArray):
    raw = np.where(inArray < 0, 0, inArray)
    raw = np.where(raw >255, 255, raw)
    return raw

def scale(inArray):
    min = inArray.min()
    factor = (inArray.max() - min)/256
    raw = (inArray - min) if (factor == 0) else ((inArray - min)/factor)
    return raw


# EDGES
#New generic cnn - well still only 1d weights
def cnn1d(inArray, weights, offset):
    acc = np.zeros(inArray.shape)
    i = 0;
    for w in weights:
        acc += weights[i] * xshift(inArray, i - offset)
        i += 1
    return crop(acc)
    
def cnn1d_v(inArray, weights, offset):
    return np.transpose(cnn1d(np.transpose(inArray), weights, offset))

def lEdge(inArray):
    return crop(cnn1d(inArray, [0.99, -0.5, -0.5], 1).astype(int))

def uEdge(inArray):
    return np.flipud(crop(cnn1d(np.flipud(inArray), [0.99, -0.5, -0.5], 1).astype(int)))

def argScaledAnd(a, b):
    return (0.5*scale(a) + 0.5*scale(b))

def countAnds(a, b):
    return (argScaledAnd(a, b) > 150).sum()

def creatULEdgeGraph(im, t):
    ledge = lEdge(im)
    uedge = uEdge(im)
    return list(map(lambda x : countAnds(uedge, xshift(ledge, -x)), range(t)))

def creatSlideRightGraph(imA, imB, t):

    return list(map(lambda x : countAnds(imA, yshift(imB, x)), range(t)))

# Do upscaling using max:
    
def upmax_vert(a):
    od = a[1::2]
    ev = a[::2]
    res = np.maximum(od, ev)
    #[max(od1, ev1) for (od1, ev1) in zip(od, ev)]
    return res

def upmax(ab):
    ay = np.transpose(upmax_vert(ab))
    return np.transpose(upmax_vert(ay))

#upmax(ab)
    
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 3, 3])
ab = np.vstack((a, b, a + b, 2*a + 3*b))


######==== TESTS
(mGrey2 < 128).sum()
(mGrey2 < -1).sum()
utils.display(mGrey2)

utils.display(lEdge(mGrey2))
utils.display(uEdge(mGrey2))

ug2 = uEdge(mGrey2)
lg2 = lEdge(mGrey2)
lines8 = ug2 + xshift(lg2,-8)
utils.display(lines8)
utils.display(upmax(lines8))
utils.display(upmax(upmax(lines8)))
utils.display(upmax(upmax(upmax(lines8))))



countAnds(ug2, lg2) #0
countAnds(ug2, xshift(lg2,4)) # 30
countAnds(ug2, xshift(lg2,5)) # 184
countAnds(ug2, xshift(lg2,6)) # 165
countAnds(ug2, xshift(lg2,7)) # 230
countAnds(ug2, xshift(lg2,-8)) # 230

utils.display(ug2 + xshift(lg2,-8))



c = creatULEdgeGraph(mGrey2, 15)
np.amax(c)
c
np.argmax(c)


cc = creatULEdgeGraph(mGrey2, 100)
np.amax(cc)
cc
np.argmax(cc)

lines1 = ug2 + xshift(lg2,-8)

slides1 = creatSlideRightGraph(lines1, xshift(lines1, 2), 200)
np.argmax(slides1)
creatSlideRightGraph(lines1, xshift(lines1, 1), 100)

utils.display(lines1)
utils.displayRGB(lines1, lines1, yshift(xshift(lines1, 2), 120))


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:45:42 2017

@author: maris
"""


import PIL
from PIL import Image
import numpy as np
import utils

mFile = Image.open("../samples/music.JPG")
mFile.show()
width, height = mFile.size

#Skinny
skinny = mFile.crop((1000, 0, 1250, width))
#skinny.show()

linearBig = skinny.transpose(Image.ROTATE_270)

#This gets it horizontal, and has got one stave (roughtly...)
factor = 4
linearBig.thumbnail((width/factor, height/factor))
#linearBig.show()
linearBig.size

#This is a barely reasonable ... 500 seems pretty wide...
med = 500, 500
mFile.thumbnail(med)
mFile.save("mMed", "bmp")

def toGrey(image):
    w, h = image.size
    greyscale = Image.new("L", (w, h), 255)
    #This is considered slow way of dealing with pixels...
    pixels = greyscale.load()
    for x in range(w):
        for y in range(h):
            r, g, b = image.getpixel((x, y))
            pixels[x, y] = (int)(0.3*r + 0.5*g + 0.2*b)
    return greyscale

grey = toGrey(linearBig)
grey.show()
#%%
#Get a single-channel image as a numpy array:
    #In theory fast, but didn't actually work
#def toArray(image):
#    imArr = np.fromstring(image.tobytes(), dtype=np.uint8)
#    imArr = imArr.reshape(image.size[1], image.size[0])
#    return imArr
    
#linArray = toArray(grey)
#linArray[1, 4]
#
#grey.tobytes()

def toArray(image):
    w, h = image.size
    imArr = list(image.getdata())
    imArr = np.array(imArr)
    imArr = imArr.reshape(h, w)
    return imArr
#%%
def toGreyArray(image):
    return toArray(toGrey(image.transpose(Image.ROTATE_270)))

utils.display(toGreyArray(mFile))
#utils.display(xshift(toGreyArray(mFile), 100))
haydn104 = toGreyArray(mFile)
display(haydn104)
#%%

linArray = toArray(grey)
linArray.shape

a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 3, 3])
ab = np.vstack((a, b, a + b, 2*a + 3*b))
ab1 = ab[:,1:4]
ab1

c = np.vstack([ab[0,], ab[:-1,]])

ab2 = ab[2:-1, ]
ab2

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
    
res = xshift(ab, -2)
#display(res)
xshift(ab, -1)
ab[:,[1]]
yshift(ab, 2)
yshift(ab, -1)
arr = ab
n = -2

#%%

#ab[:-1, ]
#np.repeat(arr[:,0], n)[:, None].shape
#np.repeat(arr[:,0], n).transpose().shape

#This lets adding a difference across the 
def edgeArray(inArray, coeffs):
    """Combine the array with its shifts a pixel up+down according to coeffs.
    The three coeffs are are for above, original and below shifts.
    """
    t, m, b = coeffs
    top = xshift(inArray, -1)#inArray[:-2, ]
    mid = inArray#inArray[1:-1, ]
    bot = xshift(inArray, 1)#inArray[2:, ]
    res = t*top + m*mid + b*bot
    return res
    
#The y direction equivalent...
def vEdgeArray(inArray, coeffs):
    l, m, r = coeffs
    left = yshift(inArray, -1)
    right = yshift(inArray, 1)
    return l*left + m*inArray + r*right
    
    
k = edgeArray(ab, (-0.5, 1.0, -0.5))
k.astype(int)
edgeArray(ab, (0, 1.0, 0))
edgeArray(ab, (0, 0, -1))

#v edge threshold
def vEdgeThreshold(inArray):
    raw = vEdgeArray(inArray, (0.5, 0.5, -0.99)).astype(int)
    raw = np.where(raw < 0, 0, raw)
    return raw 
    
    #v edge threshold
def hEdgeThreshold(inArray):
    raw = edgeArray(inArray, (0.5, 0.5, -0.99)).astype(int)
    raw = np.where(raw < 0, 0, raw)
    return raw 
    
    #v edge threshold -inner
def hEdgeThresholdInner(inArray):
    raw = edgeArray(inArray, (0.99, -0.5, -0.5)).astype(int)
    raw = np.where(raw < 0, 0, raw)
    return raw 
    
def crop(inArray):
    raw = np.where(inArray < 0, 0, inArray)
    raw = np.where(raw >255, 255, raw)
    return raw
#Makes upper edge dark, lower edge light
#This is a good start. Probably need to look at the whole image and 
# get the scale of the lines...

def edgeArray1(inArray):
    raw = edgeArray(inArray, (-0.25, -0.25, 0.5)).astype(int)
    raw += 124
    return raw
    
def vEdgeArray1(inArray):
    raw = vEdgeArray(inArray, (-0.25, -0.25, 0.5)).astype(int)
    raw += 124
    return raw   
    
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
    
#test
cnn1d(ab, [1,2], 1)
cnn1d(ab, [1,2], 0)


vGreyEdges = vEdgeArray1(linArray)
vGreyEdgesCnn1d = cnn1d_v(linArray, [-0.25, -0.25, 0.5], 1).astype(int) + 124
v = np.identity(5)
cnn1d_v(v, [2, 3, 4], 1)
cnn1d_v(v, [0, 1, 0], 1)

#%%    
# utils.display(vGreyEdges)
utils.display(vGreyEdgesCnn1d)
# utils.display(vGreyEdgesCnn1d - vGreyEdges)
#display(greyEdges)
utils.display(linArray)
#%%
def nineLong(array):
    return cnn1d_v(array, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 5)

def scale(inArray):
    min = inArray.min()
    factor = (inArray.max() - min)/256
    raw = (inArray - min) if (factor == 0) else ((inArray - min)/factor)
    return raw

tedgeNew = crop(cnn1d(linArray, [0.99, -0.5, -0.5], 1).astype(int))
tedgeInner = tedgeNew
# utils.display(tedgeNew)
longEdgesRaw = cnn1d_v(tedgeNew, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 5)
utils.display(scale(longEdgesRaw))
#longedges = utils.display(crop(1.5*longEdgesRaw.astype(int) - 120))
#utils.display(lo80ngedges)
#longedges = utils.display(crop(2.2*longEdgesRaw.astype(int) - 180))
#utils.display(longedges)
#%%

# utils.display(scale(longEdgesRaw))
#

def invert(raw):
    return -1 * raw + 255
longBlacksRaw = nineLong(invert(linArray))

def compress(inArray, lows, highs):
    slope = 256/(256 - highs - lows)
    raw = slope * scale(inArray) - (slope * lows)
    return crop(raw)
    

def softOr(in1, in2, softness):
    raw = 0.5*(scale(in1) + scale(in2))
    return compress(raw, 40 - softness, 120 - softness)

def softAnd(in1, in2, softness):
    raw = 0.5*(scale(in1) + scale(in2))
    return compress(raw, 160 - softness, 40 + softness)  

def recl(array):
    return crop((array - 128) * 2)


#cnn1d_v(invert(linArray), [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 5)
utils.display(longBlacksRaw)
def longen(raw):
    return crop(1.3*raw.astype(int) - 120)

longBlacks = crop(1.3*longBlacksRaw.astype(int) - 120)
# utils.display(crop(longBlacksRaw))
# utils.display(longBlacks)

utils.display(scale(longBlacks))

#take the h edges, and start to go sideways
bedge = hEdgeThreshold(linArray)
#bedgeInner = hEdgeThresholdInner(linArray)
bedgeInner = np.flipud(hEdgeThresholdInner(np.flipud(linArray)))

utils.displayRGB(longBlacks, tedgeInner, bedgeInner)
utils.displayRGB(tedgeInner, tedgeInner, bedgeInner)

utils.jointSpectrum(recl(scale(tedgeInner))[:,320], recl(scale(tedgeInner))[:,320])
sum(tedgeInner[:,320])
# utils.jointSpectrum(scale(tedgeInner)[:,324], scale(xshift(tedgeInner,-1))[:,324])

def jointSelf(array):
    (a, b) = array.shape
    for i in range(0, b-10):
        if (i == 0):
            acc = utils.jointSpectrum(array[:,0], xshift(array,-1)[:,0])
        else:
            acc = acc + utils.jointSpectrum(array[:,i], xshift(array,-1)[:,i])
    return acc

jointSelf(scale(tedgeInner))

lbSpectum = jointSelf(recl(scale(longBlacks)))
np.amax(lbSpectum[1:])
result = np.where(lbSpectum[1:] == np.amax(lbSpectum[1:]))
lbSpectum[10]

#I think pooling s
def pool(inArray):
    (xo, yo) = inArray.shape
    #Skip first row/column if needed to make it even
    raw = inArray[xo % 2:,yo % 2:]
    oddrows = raw[::2,:]
    evenrows = raw[1::2,:]
    res = np.maximum(oddrows, evenrows)
    oddres = res[:,::2]
    evenres = res[:,1::2]
    return np.maximum(oddres, evenres)


teIspectrum = jointSelf(scale(pool(tedgeInner)))
np.where(teIspectrum[3:] == np.amax(teIspectrum[3:]))

#%%

#This one gives considerable white in the staff lines - basically showing they are 
#roughtly 2-3 bit thick lines
# utils.displayRGB(longBlacks, xshift(tedgeInner,1), xshift(bedgeInner,-1))
#%%
#Since that looked good, let's define as a next-order feature:
staffLines1 = crop(longBlacks + xshift(tedgeInner,1) + xshift(bedgeInner,-1))
# utils.display(staffLines1)

staffLines2 = crop(longBlacks + xshift(tedgeInner,-1) + xshift(bedgeInner,1))
# utils.display(staffLines2)

#Since that looked good, let's define as a next-order feature:
staffLines3 = crop(0.6*longBlacks + 0.4*xshift(tedgeInner,-1) + 0.4*xshift(bedgeInner,1).astype(int) - 100)
utils.display(scale(staffLines3))
#utils.display(scale(staffLines4))

longTedgeInner = longen(tedgeInner)
longBedgeInner = longen(bedgeInner)
# utils.display(longBedgeInner)

utils.display(cnn1d(longBlacks, [0.1, 0,0, 0, 0, 0, 0, 0, 0.1, 0,0, 0, 0, 0, 0, 0, 0.1, 0,0, 0, 0, 0, 0, 0, 0.1, 0,0, 0, 0, 0, 0, 0, 0, 0.1], 16))
fatWhites = cnn1d(linArray, [0.2, 0.2, 0.2, 0.2, 0.2], 2)
fw2 = softAnd(fatWhites, fatWhites, 0)
utils.display(fw2)
fw3 = nineLong(nineLong(linArray))
fw4 = softAnd(fw3, fw3, 0)
# utils.display(fw4)


utils.display(softAnd(xshift(fw2, 8), longBlacks, 0))
utils.display(softAnd(xshift(invert(fw4), 8), longBlacks, 0))




def mkStaff(m, t, b):
    return crop(0.6*scale(m).astype(int) + 0.4*scale(xshift(t,-1)) + 0.4*scale(xshift(b,1)).astype(int) - 100)

staffA = mkStaff(longBlacks, longTedgeInner, longBedgeInner)
# utils.display(scale(staffA))
# utils.display(scale(pool(staffA)))

#utils.display(scale(longen(scale(staffA))))

#utils.display(mkStaff(longBlacks,longBlacks,longBlacks))

#fghj = mkStaff(longBlacks,longBlacks,longBlacks)

#Can try to extend the staff lines left/right, provided it is both blacl and has a
# Strong staff line to its left/right.

# utils.display(crop(0.7*(0.8*invert(linArray) + yshift(staffLines3, 5) + staffLines3).astype(int) - 120))
# utils.display(crop(0.7*(0.8*invert(linArray) + yshift(staffLines3, -5) + staffLines3).astype(int) - 120))

staffExtendL = crop(0.7*(0.7*invert(linArray) + yshift(staffLines3, 7) + 0.9*staffLines3).astype(int) - 120)
staffExtendR = crop(0.7*(0.7*invert(linArray) + yshift(staffLines3, -7) + 0.9*staffLines3).astype(int) - 120)
utils.display(crop(nineLong(1.2 * (staffExtendL + staffExtendR) - 0)))

staffLines4 = crop(nineLong(1.2 * (staffExtendL + staffExtendR) - 0))
# utils.display(staffLines4)
# utils.display(scale(crop(cnn1d(staffLines4, [0, -0.3, 0.4, 0.4, -0.3], 4))))
              

staffExtendLP = pool(crop(0.7*(0.7*invert(linArray) + yshift(staffLines3, 7) + 0.9*staffLines3).astype(int) - 120))
staffExtendRP = pool(crop(0.7*(0.7*invert(linArray) + yshift(staffLines3, -7) + 0.9*staffLines3).astype(int) - 120))
utils.display(crop(nineLong(1.3 * (staffExtendLP + staffExtendRP) - 20)))
#%%

tallBlacks = np.transpose(nineLong(np.transpose(invert(linArray))))
utils.display(tallBlacks)

transI = np.transpose(linArray)
ledgeInner = np.transpose(hEdgeThresholdInner(transI))
redgeInner = np.transpose(np.flipud(hEdgeThresholdInner(np.flipud(transI))))
# utils.display(redgeInner)
# utils.display(tallBlacks)

utils.displayRGB(tallBlacks, yshift(ledgeInner,-1), yshift(redgeInner,1))
uprights1 = crop(tallBlacks + yshift(ledgeInner,-1) + yshift(redgeInner,1))


def hedgeLonger(ed):
    return crop(vEdgeArray(ed, (0.7, 0.7, 0.7)).astype(int) - 128)
    
def hedgeEnd(ed):
    return crop(vEdgeArray(ed, (-0.8, 1.0, 1.0)).astype(int) - 128)
    
def hedgeStart(ed):
    return np.fliplr(hedgeEnd(np.fliplr(ed)))

def vedgeLonger(ed):
    return np.transpose(hedgeLonger(np.tranpose(ed)))
    
def vedgeEnd(ed):
    return np.transpose(hedgeLonger(np.tranpose(ed)))

startTedgerInner = hedgeStart(tedgeInner)
endTedgerInner = hedgeEnd(tedgeInner)


startEndTedge = crop(xshift(startTedgerInner + endTedgerInner, 1))
# utils.display(startEndTedge)
utils.displayRGB(staffLines1, uprights1, startEndTedge)
utils.display(staffLines1)

uprights2 = crop(0.5*tallBlacks + 0.5*yshift(ledgeInner,-1) + 0.5*yshift(redgeInner,1))

utils.display(uprights2)
utils.display(scale(uprights2))
def fiveTall(array):
    return cnn1d(array, [0.2, 0.2, 0.2, 0.2, 0.2], 3)

# utils.display(ledgeInner)
# utils.display(scale(crop(fiveTall(ledgeInner))))
utils.displayRGB(crop(fiveTall(ledgeInner)), ledgeInner, crop(fiveTall(ledgeInner)))

    
def fiveTallMiddled(array):
    return cnn1d(array, [0.2, 0.2, 0.2], 1) + cnn1d(array, [0.2, 0.2, 0.2], -1)
    
uprights3 = crop(0.4*tallBlacks 
+ 0.4*yshift(fiveTallMiddled(ledgeInner),-1) 
+ 0.4*yshift(fiveTallMiddled(redgeInner),1))
# utils.display(uprights3)


# utils.display(softOr(staffLines3, crop(fiveTall(ledgeInner)), 20))
# utils.display(softOr(staffLines4, crop(fiveTall(ledgeInner)), 20))
# utils.display(scale(crop(fiveTall(ledgeInner))))
# utils.display(scale(staffLines3))

# utils.display(softAnd(staffLines3, crop(fiveTall(ledgeInner)), 0))
utils.display(softAnd(staffLines4, crop(fiveTall(ledgeInner)), 0))

def recl(array):
    return crop((array - 128) * 2)


# utils.display(recl(scale(1.2*uprights3 + 0.3 * staffLines1)))

uprights4 = crop(0.4*tallBlacks 
+ 0.4*yshift(fiveTallMiddled(ledgeInner),-1) 
+ 0.4*yshift(fiveTallMiddled(redgeInner),1) + 0.2 * staffLines1)

# utils.display(recl(uprights4))

uprights5 = crop(0.4*tallBlacks 
+ 0.4*yshift(fiveTallMiddled(ledgeInner + 0.4* staffLines1),-1) 
+ 0.4*yshift(fiveTallMiddled(redgeInner+ 0.4 * staffLines1),1))
utils.display(recl(uprights5))
    

#%% Pooling Start
#Every second row
def div2(inArray):
    return inArray[:,::2]
    
#I think pooling s
def pool(inArray):
    (xo, yo) = inArray.shape
    #Skip first row/column if needed to make it even
    raw = inArray[xo % 2:,yo % 2:]
    oddrows = raw[::2,:]
    evenrows = raw[1::2,:]
    res = np.maximum(oddrows, evenrows)
    oddres = res[:,::2]
    evenres = res[:,1::2]
    return np.maximum(oddres, evenres)

#pool(ab)
utils.display(pool(pool(staffLines3)))
#abp = np.reshape((np.repeat(pool(ab), 2)),(2,4))
#np.repeat(np.repeat(pool(ab), 2, axis=0), 2, axis=1)

def unpool(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)



utils.displayRGB(staffLines1, uprights3, startEndTedge)
#utils.displayRGB(unpool(pool(staffLines1)), unpool(pool(uprights5)), startEndTedge)

#%%
np.shape(unpool(pool(staffLines1)))
np.shape(pool(staffLines1))

utils.display(crop(fiveTallMiddled(ledgeInner)))
#If want to move towards 
    
utils.display(recl(uprights1))
utils.display(recl(staffLines1))



def invert(raw):
    return -1 * raw + 255
    
vGreyThresholdEdge = vEdgeThreshold(linArray)
utils.display(vGreyThresholdEdge)
hGreyThresholdEdge = -1 * hEdgeThreshold(linArray) + 255
utils.display(hGreyThresholdEdge)

def hedgeLonger(ed):
    return crop(vEdgeArray(ed, (0.7, 0.7, 0.7)).astype(int) - 128)
    
def hedgeEnd(ed):
    return crop(vEdgeArray(ed, (-0.8, 1.0, 1.0)).astype(int) - 128)
    
def hedgeStart(ed):
    return np.fliplr(hedgeEnd(np.fliplr(ed)))

def vedgeLonger(ed):
    return np.transpose(hedgeLonger(np.tranpose(ed)))
    
def vedgeEnd(ed):
    return np.transpose(hedgeLonger(np.tranpose(ed)))
    
#Start looking at curves
    ###################################
#combines tedge and ledge
#def curveNW(linArray):
#%%    
utils.display(scale(recl(0.8 * (bedgeInner + ledgeInner))))
sEast = scale(recl(0.9 * (bedgeInner + ledgeInner)))
sWest = scale(recl(0.9 * (bedgeInner + redgeInner)))
nWest = scale(recl(0.9 * (tedgeInner + redgeInner)))
nEast = scale(recl(0.9 * (tedgeInner + ledgeInner)))

def shift(a, x, y):
    return xshift(yshift(a, y), x)
    
def makeCrossHair(se, sw, nw, ne, k):
    """The bad crosshair - was mixing up north and south"""
    return scale(shift(se, k, k) + shift(sw, k, -k) +\
        shift(ne, -k, k) + shift(nw, -k, -k))
        
def makeCrossHair2(se, sw, nw, ne, k):
    """Provide the SE, SW, NW, NE edges, and the pixel distance to move
    bishop-wise to the shared centre (e.g. 4 for a circle of radius 4)"""
    return makeCrossHair(ne, nw, sw, se, k);
        
cHair0 = makeCrossHair(sEast, sWest, nWest, nEast, 0)
cHair1 = makeCrossHair(sEast, sWest, nWest, nEast, -1)
cHair2 = makeCrossHair(sEast, sWest, nWest, nEast, -2)

utils.displayRGB(linArray, bedgeInner, bedgeInner)
# utils.displayRGB(linArray, nEast, nWest)

# utils.displayRGB(sEast, nWest, cHair0)

# utils.displayRGB(cHair0, cHair1, cHair2)

cHair = makeCrossHair(sEast, sWest, nWest, nEast, -4)
cHair2 = makeCrossHair2(sEast, sWest, nWest, nEast, -4)

utils.display(cHair2)

utils.displayRGB(linArray, cHair2, cHair2)
#%%

crosses = crop(recl(0.6*(tallBlacks + longBlacks)))
utils.display(crosses)
# utils.displayRGB(linArray, crosses, crosses)

noteheads = crop(recl(0.9*(1.5*cHair + crosses)))
utils.display(noteheads)

noteheads2 = crop(recl(0.9*(1.5*cHair2 + crosses)))
# utils.display(noteheads2)
#utils.display(pool(yshift(xshift(noteheads,1),1)))
# utils.display(pool(noteheads))
# utils.display(pool(noteheads2))

# utils.displayRGB(pool(noteheads2), crop(pool(noteheads2) + pool(staffLines3)), pool(staffLines3))
# utils.displayRGB(pool(noteheads2), scale(pool(staffLines3)), pool(staffLines3))
utils.displayRGB(pool(noteheads2), crop(pool(linArray)), pool(scale(staffLines3)))

# So at this point I see that noteheads2 and staffLines are both pretty good, close to looking at
# generating note assignment - that really would be something!

#%% Staff line analysis
# Ultimate need to assign line numbers for F, D, B, G and E lines ??
# Feedin perhaps single scale and pool staffLines3?

def fLine(lines, gap):
    return softAnd(lines, 0.2*(1.6*invert(xshift(lines, gap)) + invert(xshift(lines, gap+1))+ invert(xshift(lines, gap-1))), 0)
    
l = pool(scale(staffLines3))
lo = scale(staffLines3)
# utils.displayRGB(l, fLine(l, 4), invert(xshift(l, 4)))
# utils.displayRGB(lo, fLine(lo, 8), invert(xshift(lo, 8)))

ll0 = softAnd(fLine(l, 4), pool(fLine(lo, 8)), 0)
# utils.displayRGB(l, ll0, ll0)
# utils.displayRGB(lo, fLine(lo, 8), fLine(lo, 8))

#Both 4 and 5 giving decent values now. 
utils.displayRGB(l, fLine(l, 5), fLine(l, 5))
utils.displayRGB(l, fLine(l, 4), fLine(l, 4))
#utils.displayRGB(l, fLine(l, 3), fLine(l, 3))
#utils.displayRGB(l, fLine(l, 2), fLine(l, 2))
#utils.displayRGB(l, fLine(l, 1), fLine(l, 1))

sl6 = softOr(0.8*staffLines3, softAnd(yshift(staffLines3, 8), invert(longBlacks), 0), 0)
# utils.displayRGB(staffLines3, staffLines3, staffLines3)
# utils.display(sl6)
sl7 = softOr(0.8*staffLines3, softAnd(yshift(staffLines3, -8), invert(linArray), 0), 0)
# utils.displayRGB(staffLines3, sl7, sl6)
# utils.display(sl7)
# utils.display(linArray)



#The line below

def lineBelow(lines, above, gap):
   return softAnd(lines, 0.2*(1.6*(xshift(above, gap)) + invert(xshift(above, gap+1))+ invert(xshift(above, gap-1))), 0)
     
# utils.displayRGB(l, lineBelow(l, fLine(l, 4), 4), lineBelow(l, fLine(l, 4), 4))


poolNotes = 0.25*pool(noteheads)
# utils.display(makeCrossHair(poolNotes, poolNotes, poolNotes, poolNotes, 1))
poolNotes2 = 0.25*pool(noteheads2)
# utils.display(scale(noteheads2))
utils.display(makeCrossHair(poolNotes2, poolNotes2, poolNotes2, poolNotes2, 1))

def twoByTwo(a, filter):
    xa = xshift(a, 1)
    return filter[0, 0]*a + filter[1, 0]*xa + \
    filter[0,1]*yshift(a,1) + filter[1,1]*yshift(xa,1)
#%%
#######################
# Reasonably good noteheads
smallNoteHeads = crop(5*recl(4*twoByTwo(poolNotes, np.array([[0.25,0.25],[0.25,0.25]]))))
# utils.displayRGB(smallNoteHeads, smallNoteHeads, pool(invert(linArray)))

# utils.displayRGB(smallNoteHeads, pool(staffLines1), pool(uprights5))
# utils.displayRGB(staffLines1, uprights4, unpool(smallNoteHeads))
           
# utils.display(crop(5*recl(4*twoByTwo(poolNotes, np.array([[0.25,0.25],[0.25,0.25]])))))

# utils.displayRGB(noteheads, crosses, linArray)
 
# utils.displayRGB(noteheads, noteheads, invert(linArray))
   
#take the h edges, and start to go sideways
bedge = hEdgeThreshold(linArray)
bedgeInner = hEdgeThresholdInner(linArray)
bedgeInner = np.flipud(hEdgeThresholdInner(np.flipud(linArray)))
endTedgerInner = hedgeEnd(tedgeInner)
startTedgerInner = hedgeStart(tedgeInner)
utils.displayRGB(startTedgerInner, endTedgerInner, tedgeInner)

tedge = np.flipud(hEdgeThreshold(np.flipud(linArray)))
# utils.displayRGB(bedge, tedge, bedgeInner)
# utils.displayRGB(linArray, tedgeInner, bedgeInner)
# utils.displayRGB(invert(linArray), tedgeInner, bedgeInner)

utils.display(bedge)
bedgerL = crop(vEdgeArray(bedge, (0.7, 0.7, 0.7)).astype(int) - 128)
tedgerL = crop(vEdgeArray(tedge, (0.7, 0.7, 0.7)).astype(int) - 128)
endTedger = hedgeEnd(tedge)
startTedger = hedgeStart(tedge)
utils.displayRGB(endTedger, tedgerL, startTedger)

# utils.display(tedge[:,640:651])
# utils.display(bedge[:,640:651])

def analyseBtoT(a, b):
    a1 = a.flatten()
    b1 = b.flatten()
    return utils.jointSpectrum(a1, b1)
    
def toBinary(array):
    return np.where(array < 128, 0, 255)
    
utils.display(toBinary(linArray))

binArray = toBinary(linArray)
bArray = crop(binArray - xshift(binArray, 1))
utils.display(bArray)
tArray = crop(invert(binArray) - invert(xshift(binArray, 1)))
# utils.display(tArray)
# utils.displayRGB(bArray, tArray, linArray)

analyseBtoT(bArray, tArray)[:30]
analyseBtoT(tArray, bArray)[:30]

k = analyseBtoT(np.transpose(tArray), np.transpose(bArray))[:30]
(k/sum(k))[:20]
#np.set_printoptions(suppress=True)

analyseBtoT(np.transpose(bArray), np.transpose(tArray))[:30]

def analyseA(a):
    binArray = toBinary(a)
    bArray = crop(binArray - xshift(binArray, 1))
    tArray = crop(invert(binArray) - invert(xshift(binArray, 1)))
    result = analyseBtoT(np.transpose(tArray), np.transpose(bArray))
    return result / sum(result) 
    
xBlackRes = analyseA(linArray)[:50]
xWhiteRes = analyseA(invert(linArray))[:50]
yBlackRes = analyseA(np.transpose(linArray))[:50]
yWhiteRes = analyseA(invert(np.transpose(linArray)))[:50]

def getTopIndex(array):
    maxi = max(array)
    return np.where(array == maxi)

ixB, = getTopIndex(xBlackRes) # 2
ixW, = getTopIndex(xWhiteRes) # 6
iyB, = getTopIndex(yBlackRes) # 3
iyW, = getTopIndex(yWhiteRes) # 1

tres = analyseA(np.transpose(linArray))[:50]
tres
    
analyseBtoT(tedgeInner[:,640:645], bedgeInner[:,640:645])

ts  = np.transpose(tedgerL[:,510:616]) 
bs = np.transpose(bedgerL[:,510:616]) 
#%%
tToB = analyseBtoT(ts, bs)
bToT = analyseBtoT(bs, ts)
print(tToB[:62])
print(bToT[:62])
# utils.displayRB(tedgerL[:,640:816], bedgerL[:,640:816])

# utils.displayRB(startTedger, tedgerL)

ledge = vEdgeThreshold(linArray)
utils.display(ledge)
ledgerL = np.transpose(crop(vEdgeArray(np.transpose(ledge), (0.7, 0.7, 0.7)).astype(int) - 128))

# utils.displayRGB(bedge, tedge, ledge)
# utils.displayRGB(bedgerL, tedgerL, ledgerL)

# utils.displayRB(bedgerL, tedgerL)

# utils.displayRGB(startTedger, bedgerL, tedgerL)
#%%

#diplayRGB(

utils.display(invert(bedgerL))
#utils.display(invert(pool(bedgerL)))
    
#utils.display(div2(invert(bedgerL)))
    
upper = lambda x: (x < 80)
lower = lambda x: (x > 170)

#Find next point below meeting the test function f
def findNext(im, x, y, f):
    for x1 in range(x+1, im.shape[0]):
        if f(im[x1, y]):
            return x1
    return None
    
greyEdges = edgeArray1(linArray)
greyEdges.shape
#greyEdges.transpose().shape   

r = findNext(greyEdges, 0, 0, upper)
r

r = findNext(greyEdges, 0, 502, upper)
r
   
ab.shape[1]
range(3, 9)
    
edgeArray1(ab)

def findFlip(im, x, y, f, g):
    r = findNext(im, x, y, f)
    if r is None:
        return None
    s = findNext(im, r, y, g)
    if s is None:
        return None
    return (r, s)
    
r, s = findFlip(greyEdges, 0, 616, upper, lower)
(r, s)

k = findFlip(greyEdges, 30, 616, upper, lower)
k

k = findFlip(greyEdges, 21, 566, lower, upper)
k

findFlip(greyEdges, 28, 625, lower, upper)

#Search a 100 flips, lower to upper, and see how frequent...
#def getHistMap(im, upper, lower):
    
#TODO

gredges = Image.fromarray(np.uint8(greyEdges), mode="L")

gredges.show()

np.min(greyEdges)
np.max(greyEdges)
greyEdges.shape

#ab.astype(byte)
  #%%  Very old parts

#This really needs to be adaptive on the discovered staff line thickness
#But given I've discovered its 2 in this file, I can do it easily here...
def ridgeArray(inArray):
    raw = edgeArray(inArray, (1.2, 0.0, -1.2)).astype(int)
    raw += 235
    raw = np.clip(raw, 0, 255)
    return raw
thinHlines = ridgeArray(greyEdges)

gridges = Image.fromarray(np.uint8(thinHlines), mode="L")
gridges.show()

def ridgeArray2(inArray):
    raw = edgeArray(inArray, (1.6, 0.0, -1.6)).astype(int)
    raw += 245
    raw = np.clip(raw, 0, 255)
    return raw
    
gridges2 = Image.fromarray(np.uint8(ridgeArray2(greyEdges)), mode="L")
gridges2.show()

#Ok, so given the ridge array, should be able to search for staff candidates
#E.g. does, x, y a plausible (non-lower staff line?

3 + np.array([1, 2, 4])      

#Offsets is an array.
def atOffsets(im, x, y, offsets, f, g): 
    if not f(im[x, y]):
        return None
    for x1 in x + offsets:
        if x1 < im.shape[0] and g(im[x1, y]):
            return x1
            
#Offsets at single function
def atOffsetsSelf(im, x, y, offsets, f):
    return atOffsets(im, x, y, offsets, f, f)
    
#Aimed at the gridges style feature matrix:
#Says whether we have a staff like line at x, y
def staffSlice(im, x, y, offsets):
    return atOffsetsSelf(im, x, y, offsets, upper)
    
def wholeStaffSlice(im, x, y, offsets):
    ans = [x]
    for line in range(4):
        xnew = staffSlice(im, x, y, offsets)
        if xnew is not None:
            ans.append(xnew)
        else:
            return None
        x = xnew
    return ans
    
#Vertically search for a staff slice - and across actually!
def searchStaffSlice(im, xin, yin, offsets):
    x, y = xin, yin
    while y < im.shape[1]:
        while x < im.shape[0]:
            ans = wholeStaffSlice(im, x, y, offsets)
            if ans is not None:
                return ans
    
myOffsets = np.array([8, 9, 10])

#staffStart = searchStaffSlice(thinHlines, myOffsets)
#staffStart

staffSlice(thinHlines, 0, 0, myOffsets)
atOffsets(thinHlines, 0, 0, myOffsets, lower, lower)
lower(thinHlines[0, 0])

#Plausible y's to have a staff:
staffYs = np.amin(thinHlines, axis = 0)
firstStaffY = np.argwhere(staffYs < 80)
firstStaffY[0][0]

ans = searchStaffSlice(thinHlines, 466, 0, myOffsets)
ans

#Lets, get where the staff starts...
staffStartY = np.argwhere(np.count_nonzero(thinHlines < 100, axis = 0) > 4)[0][0]

#thinHlines[:,435]
#left = thinHlines[:,0:544]
#Image.fromarray(np.uint8(left), mode="L").show()

#initilise the staff
startStartX = np.argwhere(thinHlines[:,staffStartY] < 100)
startStartX[0]
startStartX.tolist()[0][0]


lines = np.full(thinHlines.shape, 255)

def flatten(a):
    r = []
    for x in a:
        r.append(x[0])
    return r
        
sf = np.array(flatten(startStartX))
thinHlines[sf, staffStartY]
thinHlines[sf - 1, staffStartY + 1]
    
#This method takes sY to be where a staff starts, and sX to be the
#x's of the lines of the staff.
#If then tries to trace the staff to the right, jumping to line above
#or below, if that seems to beradge line, and the current line isn't.
#
#It is not perfect, but given it is based on purely local info, I think
#It is ok :)
#
def staffLines(inmat, sX, sY):
    w, h = inmat.shape
    outmat = np.full(inmat.shape, 255)
    m = np.copy(sX)
    for y in range(sY, h):
        vals = inmat[m, y]
        down = inmat[m + 1, y]
        up = inmat[m - 1, y]
        n = np.where((vals > 100) & (down < 100), m + 1, m)
        m = np.where((vals > 100) & (up < 100), m - 1, n)
        outmat[m, y] = 0
        
    return outmat

g = staffLines(thinHlines, sf, staffStartY)
Image.fromarray(np.uint8(g), mode="L").show()

#k = np.amin(g, linArray)
#k = np.where(thinHlines <  linArray, thinHlines, linArray)
k = np.where(g <  linArray, g, linArray)
Image.fromarray(np.uint8(k), mode="L").show()

#Now I am greedy to want to find notes.
#The pattern for a solid circle is:
    #1. left edge - centre
    #2. left edge, above and below
    #3. top edge, higher
    #4. btm edge lower
    #5. right edge upper, lower, 
    #6. right edge centre
    
#Lets look to see when we are past a clef:
#Lets, get where the staff starts...
cleanClefs = np.argwhere(np.count_nonzero(thinHlines < 110, axis = 0) > 4)

# #Clef
# utils.display(linArray[:, 436:469])
# #Next
# utils.display(linArray[:, 471:481])
# utils.display(thinHlines[:, 490:498])
# #Next

#From y, find a right edge...this is like the start event for finding a note etc
#This is good, but the magic number 190 is quite sensitve :( )
def findRight(vEdgeArray, y):
    w, h = vEdgeArray.shape
    for y1 in range(y, h):
        slices = vEdgeArray[:, y1] > 190
        if(np.count_nonzero(slices) > 0):
            return y1, np.argwhere(slices)
    return 0, []

            
findRight(vGreyEdges, 631)
# utils.display(linArray[:, 600:629])
# utils.display(vGreyEdges[:, 600:629])
# utils.display(vGreyEdges[:, 631:638])

utils.toGrey2(linearBig)
#pwd()
y1 = 714
vEdgeArray = vGreyEdges
        
        
        

                 

    
    













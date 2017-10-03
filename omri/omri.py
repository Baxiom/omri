#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:45:42 2017

@author: maris
"""


#import PIL
from PIL import Image
import numpy as np
#import omri.utils

mFile = Image.open("samples/music.JPG")
#mFile.show()
width, height = mFile.size

#Skinny
skinny = mFile.crop((1000, 0, 1250, width))
skinny.show()

linearBig = skinny.transpose(Image.ROTATE_270)

#This gets it horizontal, and has got one stave (roughtly...)
factor = 4
linearBig.thumbnail((width/factor, height/factor))
linearBig.show()
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
    res = arr
    n = - m
    if n < 0:
        res = np.vstack([np.repeat(arr[0,], -n, axis = 0), arr[:n,]])
    elif (n > 0):
        res = np.vstack([arr[n:,], np.repeat(arr[-1,], n, axis = 0)])
    return res
    
def yshift(arr, n):
    res = arr
    if n < 0:
        res = np.hstack([arr[:, -n:], np.repeat(arr[:,[-1]], -n, axis = 1)])
    elif n > 0:
        res = np.hstack([np.repeat(arr[:,[0]], n, axis = 1), arr[:, :-n]])
    return res  
    
xshift(ab, 2)
xshift(ab, -1)
ab[:,[1]]
yshift(ab, 2)
yshift(ab, -1)
arr = ab
n = -2

ab[:-1, ]
np.repeat(arr[:,0], n)[:, None].shape
np.repeat(arr[:,0], n).transpose().shape

#This lets adding a difference across the 
def edgeArray(inArray, coeffs):
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
    
vGreyEdges = vEdgeArray1(linArray)

    
#display(vGreyEdges)
#display(greyEdges)
    
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

#Search a 100 flips, lower to upper, and see how frequent...
#def getHistMap(im, upper, lower):
    
#TODO

gredges = Image.fromarray(np.uint8(greyEdges), mode="L")

gredges.show()

np.min(greyEdges)
np.max(greyEdges)
greyEdges.shape

#ab.astype(byte)

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
#def searchStaffSlice(im, xin, yin, offsets):
#    x, y = xin, yin
#    while y < im.shape[1]:
#        while x < im.shape[0]:
#            ans = wholeStaffSlice(im, x, y, offsets)
#            if ans is not None:
#                return ans
    
myOffsets = np.array([8, 9, 10])

#staffStart = searchStaffSlice(thinHlines, myOffsets)

staffSlice(thinHlines, 0, 0, myOffsets)
atOffsets(thinHlines, 0, 0, myOffsets, lower, lower)
lower(thinHlines[0, 0])

#Plausible y's to have a staff:
staffYs = np.amin(thinHlines, axis = 0)
firstStaffY = np.argwhere(staffYs < 80)
firstStaffY[0][0]

#ans = searchStaffSlice(thinHlines, 466, 0, myOffsets)
#ans

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

#Clef
#display(linArray[:, 436:469])
#Next
#display(linArray[:, 471:481])
#display(thinHlines[:, 490:498])
#Next

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
#display(linArray[:, 600:629])
#display(vGreyEdges[:, 600:629])
#display(vGreyEdges[:, 631:638])
#
#toGrey2(linearBig)

#y1 = 714
#        vEdgeArray = vGreyEdges
#        
        
        

                 

    
    













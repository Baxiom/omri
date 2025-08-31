#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 10:56:55 2023

Pre analysis logic for working out the size of staff lines, line spacing and rotation
of images.

So these are some ideas for how this sequence could go:
    
STEP A: Use top edge / bottom edge to determine the likely staff line width,
and (maybe) the likely interstaff width.

STEP B: Using the staff width find the long blacks (i.e.. high ceratainty).

STEP C: Then find some staff lines, and do following of them. From this we could do:
    * Determine the tilt, and then rotate the image to fix that.
    * Or live with it, confident enough that we can detect the lines and hence
    later staves accurately enough.

@author: marisozols
"""

from PIL import Image
import numpy as np
import utils

#%%Test functions section

a = np.array([[1, 2, 3, 1], [2, 1, 3, 1], [1, 3, 1, 2], [1, 3, 4, 2], [4, 4, 3, 1]])
b = a.copy()
b[0, 2] = 9
b[4,3]=3
b
np.where(a == 3)
np.where(b == 3)

(b != b[:,0][:, None]).argmax()
(b != b[:,0][:, None])
(b == 3).argmax(1)
(b==3) & (a==3)
b.argmax(1)
np.maximum.accumulate((b==3), axis=1)
bx = np.maximum.accumulate((b==3), axis=1)
np.where(a, bx==True)
a2 = a.copy()
a2[bx]=0

#%% Testing for tilt in tedgeNew

# tedgeNew
# utils.display(tedgeNew)
# t1_x = np.sum(tedgeNew, axis=0)
# t1_y = np.sum(tedgeNew, axis=1)

# t1_x
# t1_y

# #%% New 20230326. STEP A.
 
# #Edges for line width.
# utils.display(linArray)

hedge1 = crop(cnn1d(invert(linArray), [0.99, -0.5, -0.5], 1).astype(int))
# utils.displayRGB(hedge1, hedge1, invert(linArray))

# hedge2 = np.flipud(crop(cnn1d(invert(np.flipud(linArray)), [0.99, -0.5, -0.5], 1).astype(int)))
# utils.displayRGB(hedge1, hedge2, invert(linArray))

m = np.amax(hedge1)
tops = np.where(hedge1 > (m/2), axis=0)

(np.average(hedge1, axis=0)).shape

np.nonzero(hedge1)
np.where(hedge1 > (m - 20))
np.argmax(hedge1 > (m - 20), axis=0)

m2 = np.amax(hedge2)
np.where(hedge2 > (m2 - 20))

mask1 = np.maximum.accumulate(hedge1 > (m/2), axis=0)
m2args = np.argmax(np.logical_and(mask1, hedge2 > m2/2), axis=0)
#args = np.argmax(if(mask1, hedge2 > m2/2), axis=0)
m1args = np.argmax(mask1, axis=0)
diff1 = m2args - m1args
diff1

utils.displayRGB(hedge1*1, hedge2*1, invert(linArray))
utils.displayRGB(mask1*200, (np.logical_and(mask1, hedge2 > m2/2))*200, invert(linArray))
utils.displayRGB(mask1*250, (np.logical_and(mask1, hedge2 > m2/2))*250, mask1*250)
utils.displayRGB((np.logical_and(mask1, hedge2 > m2/2))*160, mask1*70,  1.1*hedge1)

def count_freq(list):
    dict = {}
    for a in list:
        dict[a] = dict.get(a, 0) + 1
    return dict

count_freq(diff1)

#Now the interline gap
mask2 = np.maximum.accumulate(hedge2 > (m2/2), axis=0)
m3args = np.argmax(np.logical_and(mask2, hedge1 > m/2), axis=0)
diff2 = m3args - m2args
diff2
result = count_freq(diff2)
result
dict(filter(lambda i : i[1] >30, result.items()))

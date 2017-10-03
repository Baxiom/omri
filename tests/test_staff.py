#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 21:37:30 2017

@author: maris
"""
from PIL import Image
from omri.omri import toGrey
from omri.utils import display


mFile = Image.open("samples/music.JPG")
greyBig = toGrey(mFile)
display(greyBig)

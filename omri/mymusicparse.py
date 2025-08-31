#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:39:37 2023

@author: marisozols
"""

from xml.dom.minidom import parse, parseString, Node
#from xml.etree.ElementTree import parse
#document = parse("../samples/Paraiso_1926.musicxml")
document = parse("../samples/Paraiso1926_v2.musicxml")
#getOnsetsFromXMLMeasure
#document = parse("../samples/Paraiso_Tango2-Milonga__Ernesto_Nazareth__1926.musicxml")

parts = document.getElementsByTagName('part')
len(parts)
part1Notes = parts[0].getElementsByTagName('note')
part2Notes = parts[1].getElementsByTagName('note')
len(part1Notes)
len(part2Notes)

#typical measure

m1_2 = parts[0].getElementsByTagName('measure')[1]
m1_2
m1_2.getElementsByTagName('note')
m1_2.getElementsByTagName('division')
n1 = m1_2.getElementsByTagName('note')[0]

def getOnsetsFromXMLMeasure(measure, division):
    onsets = [0]*division
    place = 0
    k = 0
    for element in measure.childNodes:
        if len(element.getElementsByTagName('grace')) > 0:
            #do nothing
            k += 1
        elif len(element.getElementsByTagName('chord')) > 0:
            #do nothing
            k += 1
        elif len(element.getElementsByTagName('rest')) > 0:
            #Get duration and move the 
            dur = element.getElementsByTagName('duration')
            place += int(dur[0].childNodes[0].data)
        elif len(element.getElementsByTagName('backup')) > 0:
            #Get duration and move back 
            dur = element.getElementsByTagName('duration')
            place -= int(dur[0].childNodes[0].data)
        elif element.tagName == 'note':
            #Handle like a rest, but first capture the onset
            onsets[place] = 1
            dur = element.getElementsByTagName('duration')
            place += int(dur[0].childNodes[0].data)
    return place

getOnsetsFromXMLMeasure(m1_2, 8)

#%% Revised measure parser
def getOnsetsFromXMLMeasure(measure, division):
    onsets = [0]*division
    place = 0
    k = 0
    for element in measure.childNodes:
        if element.nodeName == '#text':
            k += 1
        elif len(element.getElementsByTagName('grace')) > 0:
            #do nothing
            k += 1
        elif len(element.getElementsByTagName('chord')) > 0:
            #do nothing
            k += 1
        elif len(element.getElementsByTagName('rest')) > 0:
            #Get duration and move the 
            dur = element.getElementsByTagName('duration')
            print("place:", place, " rest", " dur:", dur[0].childNodes[0].data)
            place += int(dur[0].childNodes[0].data)
        elif element.tagName == 'backup':
            #Get duration and move back 
            dur = element.getElementsByTagName('duration')
            print("place:", place, " backup", " dur:", dur[0].childNodes[0].data)
            place -= int(dur[0].childNodes[0].data)
        elif element.tagName == 'note':
            print("place:", place, " note")
            #Handle like a rest, but first capture the onset
            onsets[place] = 1
            dur = element.getElementsByTagName('duration')
            print("place:", place, " note", " dur:", dur[0].childNodes[0].data)
            place += int(dur[0].childNodes[0].data)
    print("MEasure:", measure.getAttribute("number"), onsets)
    return onsets

#%% Revised measure parser

def isTiedToExtension(noteElement):
    isTied = False
    for tie in noteElement.getElementsByTagName('tie'):
        isTied = isTied or (tie.getAttribute('type') == 'stop')
    return isTied    
    
def getOnsetsFromXMLMeasure(measure, division):
    onsets = [0]*division
    place = 0
    k = 0
    for element in measure.childNodes:
        if element.nodeName == '#text':
            k += 1
        elif len(element.getElementsByTagName('grace')) > 0:
            #do nothing
            k += 1
        elif len(element.getElementsByTagName('chord')) > 0:
            #do nothing
            k += 1
        elif len(element.getElementsByTagName('rest')) > 0:
            #Get duration and move the 
            dur = element.getElementsByTagName('duration')
            place += int(dur[0].childNodes[0].data)
        elif element.tagName == 'backup':
            #Get duration and move back 
            dur = element.getElementsByTagName('duration')
            place -= int(dur[0].childNodes[0].data)
        elif element.tagName == 'note':
            #Handle like a rest, but first capture the onset
            #Should not do an onset if tied to an earlier note
            if isTiedToExtension(element):
                print("isTiedTo:", element)
            else:
                onsets[place] = 1
            dur = element.getElementsByTagName('duration')
            place += int(dur[0].childNodes[0].data)
    print("Measure:", measure.getAttribute("number"), onsets)
    return onsets

getOnsetsFromXMLMeasure(m1_2, 16)

def getOnsestsFromPart(part, division):
    return list(map (lambda m: getOnsetsFromXMLMeasure(m, division),  part.getElementsByTagName('measure')))

lh1_para = getOnsestsFromPart(parts[0], 8)
#list(map (lambda m: getOnsetsFromXMLMeasure(m,8),  parts[0].getElementsByTagName('measure')))
#rh1_para = list(map (lambda m: getOnsetsFromXMLMeasure(m,8),  parts[1].getElementsByTagName('measure')))
rh1_para = getOnsestsFromPart(parts[1], 8)

#%%  Pulling in some of the musicxml.py stuff so get to the same basis, whether parsed by
# music21, or by my own stuff.

def to_code_nums(strs):
    return list(map(lambda str: to_code_num(str), strs))

lh1_codes = to_code_nums(lh1_para)
paraiso_codesL_dict = {}
extend_count_dict(paraiso_codesL_dict, lh1_codes)

print_codes_dict(paraiso_codesL_dict)

rh1_codes = to_code_nums(rh1_para)
paraiso_codesR_dict = {}
extend_count_dict(paraiso_codesR_dict, rh1_codes)

print_codes_dict(paraiso_codesR_dict)

















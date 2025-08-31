#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:08:07 2023

@author: marisozols
Musicxml parsing etc, from https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S2_MusicXML.html
"""

import sys
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas as pd
import IPython.display as ipd
import music21 as m21
import itertools

sys.path.append('..')
import libfmp.c1

fn = os.path.join('..', 'samples', 'bwv69.6.xml')

with open(fn, 'r') as stream:
    xml_str = stream.read()

start = xml_str.find('<note')
end = xml_str[start:].find('</note>') + start + len('</note>')
print(xml_str[start:end])

#%% Now the function to have music21 parse it.

def xml_to_list(xml):
    """Convert a music xml file to a list of note events

    Notebook: C1/C1S2_MusicXML.ipynb

    Args:
        xml (str or music21.stream.Score): Either a path to a music xml file or a music21.stream.Score

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, duration, pitch, velocity, label]``
    """

    if isinstance(xml, str):
        xml_data = m21.converter.parse(xml)
    elif isinstance(xml, m21.stream.Score):
        xml_data = xml
    else:
        raise RuntimeError('midi must be a path to a midi file or music21.stream.Score')

    score = []

    for part in xml_data.parts:
        instrument = part.getInstrument().instrumentName

        for note in part.flat.notes:

            if note.isChord:
                start = note.offset
                duration = note.quarterLength

                for chord_note in note.pitches:
                    pitch = chord_note.ps
                    volume = note.volume.realized
                    score.append([start, duration, pitch, volume, instrument])

            else:
                start = note.offset
                duration = note.quarterLength
                pitch = note.pitch.ps
                volume = note.volume.realized
                score.append([start, duration, pitch, volume, instrument])

    score = sorted(score, key=lambda x: (x[0], x[2]))
    return score

xml_data = m21.converter.parse(fn)
xml_list = xml_to_list(xml_data)

df = pd.DataFrame(xml_list[:9], columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
html = df.to_html(index=False, float_format='%.2f', max_rows=8)
ipd.HTML(html)

#%% In theory shows piano roll:
libfmp.c1.visualize_piano_roll(xml_list, figsize=(8, 3), velocity_alpha=True,xlabel='Time (quarter lengths)');


#%% My additions
def get_instrument_part(event_list, instrument):
    return list(filter(lambda e : e[4] == instrument, event_list))
    
bach_69_6_trumpet = get_instrument_part(xml_list, "Instrument 6")

def get_onset_times(event_list):
    return list(map(lambda e : e[0], event_list))

bach_69_6_trumpet_onset_times = get_onset_times(bach_69_6_trumpet)
bach_69_6_trumpet_onset_times


def starts_to_binit_rep(onset_times, increment):
    '''
    So the important thing here is that the increment needs to allocate to the smallest
    unit. E.g. if all the times are whole numbers, then use 1. But if there are quarters
    say (5.25 or 9.75), then will need (at least) 4.
    '''
    #lastVal = 0
    result = []
    if (onset_times[0] == 0):
        result.append(1)
    for val in onset_times:
        #diff = round(increment*(val - lastVal))
        diff = round(increment*val - (len(result) - 1))
        if (diff > 0):
            result.extend([0] *  (diff - 1))
            result.append(1)
        #lastVal = val
    return result
            
bach_69_6_trumpet_onsets = starts_to_binit_rep(bach_69_6_trumpet_onset_times, 4)  
bach_69_6_trumpet_onsets   

#%%Some fun dealing directly with the xml data from music21:
bach_xml = xml_data
def print_part_names(xml):
    for part in xml.parts:
        print(part.partName)
         
print_part_names(bach_xml)
         
         
p0 = bach_xml.parts[0]
p0       
getattr(p0, "partName")

#%%First tango analysis

fn_tango1 = os.path.join('..', 'samples', 'Derecho_Viejo__Tango_milonga______Arolas.musicxml')
with open(fn_tango1, 'r') as stream2:
    xml_tango1 = stream2.read()

tango1_list = xml_to_list(xml_tango1)
#t1_PianoPart = get_instrument_part(tango1_list, "Piano")
tango1_xml_data = m21.converter.parse(xml_tango1)

t1_PianoPart = tango1_xml_data.parts[0]
t1_notes = t1_PianoPart.flat.notes
dir(t1_notes[0])
t1_notes[1].staff
libfmp.c1.visualize_piano_roll(tango1_list, figsize=(10, 3), velocity_alpha=True,xlabel='Time (quarter lengths)');

labels = list(map(lambda r : r[4], tango1_list))
set(labels)

list(map(lambda r : r[3], tango1_list[:20]))
list(map(lambda r : r[2], tango1_list[-20:]))


sinst = n1._getStoredInstrument
dir(sinst)

#%%
# 
def xml_part_to_start_list(xml_part, lowerVal, upperVal):
    '''   
    Parameters
    ----------
    xml_part : music xml part fragment as parsed by music21
        DESCRIPTION.
    lower : float
        only consider notes/chords that have at least this volume.
    upper : float
        only consider noted/chords with less than this volume.

    Returns
    -------
    list of the unique start times of the notes/chirds between lower and upper
    in their volume (proxy for not knowing which stave the notes are on)

    '''
    startOffset = -1
    start_list = []
    newstarts = sorted(list(xml_part.flat.notes), key = lambda n : n.offset )
    for note in newstarts:
        #print(startOffset)
        if (note.offset > startOffset): 
            #print(note.offset)
            if (note.volume.realized > lowerVal) & (note.volume.realized < upperVal):
                startOffset = note.offset
                #print(startOffset)
                start_list.append(startOffset)
    return start_list
 
t1_PianoPart_lh = tango1_xml_data.parts[0]
t1_PianoPart_rh = tango1_xml_data.parts[1]


leftHand_t1 = xml_part_to_start_list(t1_PianoPart_lh, 0.0, 1.1)
leftHand_t1
tango1_lh_onsets = starts_to_binit_rep(leftHand_t1, 4)  
tango1_lh_onsets
len(tango1_lh_onsets)

rightHand_t1 = xml_part_to_start_list(t1_PianoPart_rh, 0.0, 1.1)
rightHand_t1
tango1_rh_onsets = starts_to_binit_rep(rightHand_t1, 4) 

#bothHands_t1 = xml_part_to_start_list(t1_PianoPart, 0.0, 1.1)
#tango1_all_onsets = starts_to_binit_rep(leftHand_t1, 4)  


def pad_to_multiple(l, multiple): 
    g = l.copy()
    partial = len(g) % multiple
    if (partial > 0):
        g.extend([0] * (multiple - partial))
    return g


tango1_lh_onsets_padded = pad_to_multiple(tango1_lh_onsets, 16)
len(tango1_lh_onsets_padded)
lh_array = np.array(tango1_lh_onsets_padded)
lh16 = np.reshape(lh_array, ((round((len(tango1_lh_onsets_padded))/16), 16)))

tango1_rh_onsets_padded = pad_to_multiple(tango1_rh_onsets, 16)
len(tango1_rh_onsets_padded)
rh_array = np.array(tango1_rh_onsets_padded)
np.reshape(rh_array, (round(len(tango1_rh_onsets_padded)/16), 16))


#tango1_all_onsets_padded = pad_to_multiple(tango1_all_onsets, 16)

#t1_all_array = np.array(tango1_all_onsets_padded)
#np.reshape(t1_all_array, (round(len(tango1_all_onsets_padded)/16), 16))


def to_code_num(str):
    '''
    given list of 1's, 0's gives the code number for it

    '''
    counts = list(range(len(str)))
    factors = np.array(list(map(lambda x: pow(2, x), counts)))
    return np.dot(factors, str)
    
lh16_codes = to_code_num(lh16.T)

lh8 = np.reshape(lh_array, ((round((len(tango1_lh_onsets_padded))/8), 8)))
lh8_codes = to_code_num(lh8.T)
lh8_codes

def code_to_ones(code, n):
    ans = []
    div = 2
    restCode = code
    for i in range(n):
        ans.append(restCode % 2)
        restCode = restCode // 2
    return ans
                  
code_to_ones(91, 8)       
code_to_ones(185, 8)          

def extend_count_dict(counts, codes):
    for code in codes: 
        counts[code] = counts.get(code, 0) + 1
        
tango1_codes = {}

extend_count_dict(tango1_codes, lh8_codes)
tango1_codes

def print_codes_dict(counts):
    for code, count in counts.items():
        print("rhythm:", code_to_ones(code, 8), " - count:", count)
                  
                  
print_codes_dict(tango1_codes)                
                  
                  
 #%% Trying to build the processing chain here, so don't need so many intermediate
# variables

#This takes a raw part from music21 parse, and returns onsets:
xml_part_to_start_list

#Things to do after this:
# - padd
# - to array
# - reshape
# - transpose
# Put in appropriate dictionaries - one for the part, another for the global.

#One thing that could be good is to put both the LH and the RH part in a structure.

def part_to_dict(part, n):
    starts = xml_part_to_start_list(part, -1, 1.1)
    onsets = starts_to_binit_rep(starts, 4)
    padded = pad_to_multiple(onsets, n)
    shaped_array = np.reshape(np.array(padded), (round(len(padded)/n), n))
    codes = to_code_num(shaped_array.T)
    codes_dict = {}
    extend_count_dict(codes_dict, codes)
    return codes_dict
    
def part_to_dict_anacrusis(part, n, anac):
    starts = xml_part_to_start_list(part, -1, 1.1)
    onsets = [0] * anac
    onsets.extend(starts_to_binit_rep(starts, 4))
    padded = pad_to_multiple(onsets, n)
    shaped_array = np.reshape(np.array(padded), (round(len(padded)/n), n))
    codes = to_code_num(shaped_array.T)
    codes_dict = {}
    extend_count_dict(codes_dict, codes)
    return codes_dict
 

test_lh1 = part_to_dict(tango1_xml_data.parts[0], 8)
test_rh1 = part_to_dict(tango1_xml_data.parts[1], 8)
print_codes_dict(test_rh1)

#%% Tango 2 !!!

#fn_tango2 = os.path.join('..', 'samples', 'Paraiso_Tango-Milonga__Ernesto_Nazareth__1926-Piano.musicxml')
fn_tango2 = os.path.join('..', 'samples', 'Paraiso_1926.musicxml')
#fn_tango2 = os.path.join('..', 'samples', 'Tango2 ParaisoMilonga1926.musicxml.xml')
with open(fn_tango2, 'r') as stream3:
    xml_tango2 = stream3.read()
                 
tango2_xml_data = m21.converter.parse(xml_tango2)

test2_lh1 = part_to_dict(tango2_xml_data.parts[0], 8)
#Some error here
test2_rh1 = part_to_dict(tango2_xml_data.parts[1], 8)
print_codes_dict(test2_lh1)
print_codes_dict(test2_rh1)

n1 = tango2_xml_data.parts[0].flat.notes[0]
n1.staff
dir(n1)
dir(tango2_xml_data)

 # OK - I NEEED TO REMEMBER TO EXPORT FROM MUSE SCORE AS UNCOMPRESSED MUSICXML                 
                  
dir(tango2_xml_data)

dir(n1.notehead.index)

#%% Tango 3 chocla
fn_tango3 = os.path.join('..', 'samples', 'El_choclo_-_Milonga.musicxml')
#fn_tango2 = os.path.join('..', 'samples', 'Tango2 ParaisoMilonga1926.musicxml.xml')
with open(fn_tango3, 'r') as stream4:
    xml_tango3 = stream4.read()
    
tango3_xml_data = m21.converter.parse(xml_tango3)
test3_lh1 = part_to_dict_anacrusis(tango3_xml_data.parts[0], 8, 5)
#Some error here
test3_rh1 = part_to_dict_anacrusis(tango3_xml_data.parts[1], 8, 5)
print_codes_dict(test3_lh1)
print_codes_dict(test3_rh1)
both = max(test3_lh1, test3_rh1)
 
                  
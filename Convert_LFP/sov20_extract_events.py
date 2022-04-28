# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:12:06 2020

@author: Wladek
"""
import numpy as np #wczytanie biblioteki z funkcjami matematycznymi
import pylab as py #wczytanie biblioteki do rysowania
from scipy.stats import pearsonr
import DemoReadSGLXData.readSGLX as readSGLX
from scipy.signal import filtfilt, butter,detrend,argrelmin
from pathlib import Path
import scipy.io
#%%
def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': 
      return np.dot(X,np.array([[1.,  0,  0],[0 ,  c, -s],[0 ,  s,  c] ]))
  elif axis == 'y': 
      return np.dot(X,np.array([[c,  0,  -s],[0,  1,   0],[s,  0,   c]]))
  elif axis == 'z': 
      return np.dot(X,np.array([[c, -s,  0 ],[s,  c,  0 ],[0,  0,  1.]]))
  
def dan_fetch_electrodes(meta):
    imroList = meta['imroTbl'].split(sep=')')
    nChan = len(imroList) - 2
    electrode = np.zeros(nChan, dtype=int)        # default type = float
    channel = np.zeros(nChan, dtype=int)
    bank = np.zeros(nChan, dtype=int)
    for i in range(0, nChan):
        currList = imroList[i+1].split(sep=' ')
        channel[i] = int(currList[0][1:])
        bank[i] = int(currList[1])
    # Channel N => Electrode (1+N+384*A), where N = 0:383, A=0:2
    electrode = 1 + channel + 384 * bank
    return(electrode, channel)
    
def eles_to_rows(eles):
    rows = []
    for ele in eles:
        rows.append(np.int(np.ceil(ele/2)))
    return rows

def eles_to_ycoord(eles):
    rows = eles_to_rows(eles)
    y_coords = []
    for ii in rows:
        y_coords.append(int((480 - ii)*20))
    return y_coords

def eles_to_xcoord(eles):
    x_coords = []
    for ele in eles:
        off = ele%4
        if off == 1: x_coords.append(-24)
        elif off == 2: x_coords.append(8)
        elif off == 3: x_coords.append(-8)
        elif off==0: x_coords.append(24)
    return x_coords

def eles_to_coords(eles):
    xs = eles_to_xcoord(eles)
    ys = eles_to_ycoord(eles)
    return np.array((xs, ys)).T
#%% 
loadir = 'H:\\SOVy\\SOV20\\'
name='05_ep_g0_t0.imec0.ap.bin'
lid = ''
binFullPath = Path(loadir+name)
meta = readSGLX.readMeta(binFullPath)
Fs = int(readSGLX.SampRate(meta))
sig_B1 = readSGLX.makeMemMapRaw(binFullPath, meta)[:, ::3]
Fs=int(Fs/3)
electrodes, channels = dan_fetch_electrodes(meta)
ch_order = electrodes.argsort()
electrodes.sort()
ele_pos_2D = eles_to_coords(electrodes[::-1])/1000
ele_pos = np.array([ele_pos_2D[:,0]-3.1, 
                    np.zeros(384)+4,
                    ele_pos_2D[:,1]+1.7])
ttl = np.loadtxt(loadir+ 'TTL_'+name[:2]+'.txt')
trials_b = np.zeros((385,len(ttl),600))
for i in range(len(ttl)):
    t1=ttl[i]*Fs
    trials_b[:,i] = sig_B1[:,int(t1-100):int(t1+500)]
del(sig_B1)
#%%
lfp_s = trials_b.mean(axis=1)
b,a = butter(3,[1/(Fs/2),2000/(Fs/2)], btype='bandpass')
lfp_s = filtfilt(b,a, lfp_s[ch_order], padlen=100)[:,65:365]
lfp_s = np.delete(lfp_s, 287, axis=0)[:369]
ele_pos=np.delete(ele_pos,287,axis=1)[:,:369]

frags = 30
leng= 30
leap=10
ch_stay=320
cor_score = np.zeros((ele_pos.shape[1],frags))
for part in range(frags-3):
    for ch in range(ele_pos.shape[1]):
        cor_score[ch,part] = pearsonr(lfp_s[ch_stay,leap*part:leap*part+leng], lfp_s[ch,leap*part:leap*part+leng])[0]
py.figure(figsize=(8,8))
py.subplot(211)
py.title('LFP sov20'+lid)
# ele_pos_sort = np.argsort(ele_pos[:128,2])

py.imshow(lfp_s, vmin=-25, vmax=25, extent=[-5,25,ele_pos.shape[1],1], aspect='auto', cmap='PRGn', origin='lower')
py.ylabel('channels')
py.subplot(212)
py.title('Correlation sov20'+lid)
py.imshow(cor_score, vmin=-1, vmax=1, extent=[-5,25,1,ele_pos.shape[1]],aspect='auto', cmap='PiYG', origin='lower')
py.ylabel('channels'), py.xlabel('time')
py.savefig('sov20_'+lid+'_load')
scipy.io.savemat('./mats/sov20'+lid+'.mat',
                 dict(crtx=lfp_s[261:], th=lfp_s[:261],
                 cor=cor_score, ele_pos=ele_pos))
    
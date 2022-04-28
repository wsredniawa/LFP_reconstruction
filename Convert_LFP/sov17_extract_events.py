# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:09:12 2020

@author: Wladek
"""


import numpy as np
import pylab as py 
from scipy.signal import filtfilt, butter, detrend, argrelmax, argrelmin
from scipy.stats import pearsonr
import pandas as pd
import h5py
import os
import scipy.io

py.close('all')

lay = pd.read_csv('A8x8.lay', sep=',', header = None)
lay_mir = pd.read_csv('A8x8_mirror.lay', sep=' ', header = None)
mapa1 = pd.read_csv('samstac40_NeuroNexus_A1_8x8_EK.map',sep=' ', header=None)
col = lay[1].values
col_mir = lay_mir[1].values
depth = lay[2].values
ele_pos = []
for c in range(8):
    for d in range(8): ele_pos.append([0, c*.2, d*.2])
ele_pos_pre = np.asarray(ele_pos) 

def organize_eles(ele_pos, angle):
    ele_pos_crtx = np.zeros((ele_pos.shape[0],3))
    for n in range(64):
        new_x = 0
        new_y = ele_pos[n,1]
        new_z = -ele_pos[n,2]
        ele_pos_crtx[n] = new_x, new_y, new_z
    return ele_pos_crtx

def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': 
      return np.dot(X,np.array([[1.,  0,  0],[0 ,  c, -s],[0 ,  s,  c] ]))
  elif axis == 'y': 
      return np.dot(X,np.array([[c,  0,  -s],[0,  1,   0],[s,  0,   c]]))
  elif axis == 'z': 
      return np.dot(X,np.array([[c, -s,  0 ],[s,  c,  0 ],[0,  0,  1.]]))

def organize_pos(ele_pos, plot=False):
    a01 = np.array([-1.44, 4.4, 0.65])
    a0 = a01
    a1 = np.zeros(3)
    a8 = np.array([-1.44, 3.8, 1.95]) - a0
    h1 = np.array([-2.04, 5.45, 1.14]) - a0
    h8 = np.array([-2.04, 4.74, 2.42]) - a0
    xpos1, ypos1, zpos1 = [a1[0], a8[0], h1[0], h8[0]], [a1[1], a8[1], h1[1], h8[1]], [a1[2], a8[2], h1[2], h8[2]]
    ele_pos = organize_eles(ele_pos, 0)
    ele_pos_rot1 = rotate(ele_pos, -np.pi/6, 'z')
    ele_pos_rot2 = rotate(ele_pos_rot1, -np.pi/8, 'x')
    a12 = np.array([-1.44, 3.95, 1.55])-a0
    print('vector length: ', np.linalg.norm(a12))
    a82 = np.array([-1.44, 3.35, 2.8])-a0
    h12 = np.array([-2.04, 4.93, 2.07])-a0
    h82 = np.array([-2.04, 4.28, 3.3])-a0
    xpos12, ypos12, zpos12 = [a12[0], a82[0], h12[0], h82[0]], [a12[1], a82[1], h12[1], h82[1]], [a12[2], a82[2], h12[2], h82[2]]
    ele_pos_crtx1 = ele_pos_rot2 - np.repeat(ele_pos_rot2[7],64).reshape((3,64)).T
    ele_pos_crtx2 = ele_pos_crtx1 + np.repeat(a12,64).reshape((3,64)).T
    a1t = np.array([-3.6, 2.25, 6.0]) - a0
    a8t = np.array([-3.6, 2.25, 7.4]) - a0
    h1t = np.array([-3.6, 3.6, 6]) - a0
    h8t = np.array([-3.6, 3.6, 7.4]) - a0
    xpos2, ypos2, zpos2 = [a1t[0], a8t[0], h1t[0], h8t[0]], [a1t[1], a8t[1], h1t[1], h8t[1]], [a1t[2], a8t[2], h1t[2], h8t[2]]
    a8t2 = np.array([-3.6, 2.25, 6.4]) - a0
    a8t3 = np.array([-3.6, 2.25, 5.4]) - a0
    ele_pos_t1 = ele_pos + np.repeat(a8t,64).reshape((3,64)).T
    ele_pos_t2 = ele_pos + np.repeat(a8t2,64).reshape((3,64)).T
    ele_pos_t3 = ele_pos + np.repeat(a8t3,64).reshape((3,64)).T
    return np.concatenate((ele_pos_crtx1, ele_pos_crtx2, ele_pos_t3, ele_pos_t2, ele_pos_t1), axis=0)

def load_frag(file, start=0, stop=50, part_shape=20000, resamp=40):
    part = int(part_shape/resamp)
    sig_mtrx = np.zeros((129,stop-start,part), dtype=np.int16)
    for i in range(start,stop,1):
        if i%50==0: print('fragment loaded:', i)
        sig_temp =  np.array(file[str(i)]['data'])
        sig_mtrx[:128,i-start,:]=sig_temp[:128,::resamp][:,:part]
        sig_mtrx[128,i-start,:]=sig_temp[-1,::resamp][:part]
    sig_mtrx=sig_mtrx.reshape((sig_mtrx.shape[0], sig_mtrx.shape[1]*part))
    for i in range(128):
        sig_mtrx[i] = sig_mtrx[i]-np.mean(sig_mtrx[i])
    return sig_mtrx

def load_pots(n_list, trials=False, num=0):
    lista = [] 
    for n,pots in enumerate(n_list):
        pots_reordered = np.zeros((pots.shape))
        for i in range(64):
            ch=mapa1[0].values[i]-1
            if n==1:
                position = int(col_mir[ch]/200)*8 + int(depth[ch]/200)
            else:
                position = int(col[ch]/200)*8 + int(depth[ch]/200)
            pots_reordered[position,:] = pots[ch]
        lista.append(pots_reordered)
    return lista[0], lista[1]

def draw_signal(col, depth, shift=64, draw=False):
    global events
    events = np.where(sig[-1,:]>4000)[0][::3*resamp]
    py.figure()
    py.plot(sig[-1,:])
    py.plot(events, np.zeros(len(events))+4000, 'o', color='r')
    sweeps = np.zeros((64,len(events),300), dtype=np.int16)    
    if draw:
        py.figure(dpi=300, figsize=(12,10))
        py.subplot(111)
    for n in range(64):
        ch=mapa1[0].values[n]-1
        for ii,event in enumerate(events[2:-2]):
            sweeps[ch,ii] = sig[n+shift,event-45:event+255]
    return sweeps
#%%
savedir = './npys/'
datadir = 'H:/SOVy/SOV17/'
sovs = ['02wasy_', '03wasy_', '04wasy_','05wasy_']
sov=sovs[0]
savename=sov
print(savename)
resamp = 4

Fs=10000
lay = pd.read_csv('A8x8.lay', sep=',', header = None)
lay_mir = pd.read_csv('A8x8_mirror.lay', sep=' ', header = None)
mapa1 = pd.read_csv('samstac40_NeuroNexus_A1_8x8_EK.map',sep=' ', header=None)
for sov in sovs:
    savename=sov
    name = [i for i in os.listdir(datadir) if sov in i][0]
    file = h5py.File(datadir+name, 'r')
    print('file len: ', len(file.keys())/2/60, ' min')
    sig = load_frag(file, start=10, stop=800, resamp=resamp)
    ele_pos  = organize_pos(ele_pos_pre, plot=0)
    trials_b = draw_signal(col, depth, shift=64)
    trials_t = draw_signal(col, depth, shift=0)
    trials_b, trials_t = load_pots([trials_b, trials_t])
    lfp_s = np.concatenate((trials_b.mean(axis=1),trials_t.mean(axis=1)))
    frags = 30
    leng= 30
    leap=10
    ch_stay=11
    cor_score = np.zeros((128,frags))
    for part in range(frags-3):
        for ch in range(128):
            cor_score[ch,part] = pearsonr(lfp_s[ch_stay,leap*part:leap*part+leng], lfp_s[ch,leap*part:leap*part+leng])[0]
    py.figure(figsize=(8,8))
    py.subplot(211)
    py.title('LFP sov17:: '+savename)
    ele_pos_sort = np.argsort(ele_pos[:128,2])
    py.imshow(lfp_s, vmin=-100, vmax=100, extent=[-5,25,ele_pos.shape[1],1], aspect='auto', cmap='PRGn')
    py.ylabel('channels')
    py.subplot(212)
    py.title('Correlation sov17:: '+savename)
    py.imshow(cor_score[list(ele_pos_sort)], vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],1],aspect='auto', cmap='PiYG')
    py.ylabel('channels'), py.xlabel('time')
    py.savefig('sov17_'+savename+'_load')
    py.close('all')
    scipy.io.savemat('./mats/sov17_'+savename+'.mat',
                     dict(crtx=trials_b.mean(axis=1), th=trials_t.mean(axis=1),
                     cor=cor_score, ele_pos=ele_pos))
    
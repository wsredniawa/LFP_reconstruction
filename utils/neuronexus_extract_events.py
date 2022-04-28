# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:09:12 2020

@author: Wladek
"""


import numpy as np
import pylab as py 
from scipy.signal import filtfilt, butter, detrend, argrelmax, argrelmin
import pandas as pd
import h5py
import os
from kcsd import KCSD2D
import matplotlib.animation as animation

py.close('all')
def extract_events(sig,thr):
    bool_sig = sig<thr
    events = argrelmin(bool_sig*sig)
    events = events[0]
    py.figure()
    for i in events: 
        py.axvline(i, ls='--')
    py.plot(sig)
    return events[::4]

def load_frag(file, start=0, stop=50, part_shape=20000, resamp=40):
    part = int(part_shape/resamp)
    sig_mtrx = np.zeros((129,stop-start,part), dtype=np.int16)
    for i in range(start,stop,1):
        if i%50==0: print('fragment loaded:', i)
        sig_temp =  file[str(i)]['data'].value
        sig_mtrx[:128,i-start,:]=sig_temp[:128,::resamp][:,:part]
        sig_mtrx[128,i-start,:]=sig_temp[-1,::resamp][:part]
    sig_mtrx=sig_mtrx.reshape((sig_mtrx.shape[0], sig_mtrx.shape[1]*part))
    return sig_mtrx

def draw_signal(col, depth, part, shift=64, draw=False):
    global events
    if part=='th': div,shift=.5,0
    else: div=2
    time = np.linspace(0,180,int(125*crct))
    events = np.where(sig[-1,:]>4000)[0][::10*resamp]
    py.figure()
    py.plot(sig[-1,:])
    py.plot(events, np.zeros(len(events))+4000, 'o', color='r')
    sweeps = np.zeros((64,len(events),int(125*crct)), dtype=np.int16)    
    if draw:
        py.figure(dpi=300, figsize=(12,10))
        py.subplot(111)
        py.suptitle('example -25 +100 ms: '+name[:-5]+' ::: '+part)
    for n in range(64):
        ch=mapa1[0].values[n]-1
        for ii,event in enumerate(events[2:-2]):
            sweeps[ch,ii] = sig[n+shift,event-int(25*crct):event+int(100*crct)]
        norm_sig = sweeps[ch].mean(axis=0)
        if draw:
            tm = 36
            py.plot(time+col[ch],norm_sig/div+depth[ch], color='black')
            py.plot([col[ch]+tm, col[ch]+tm], [depth[ch]-tm, depth[ch]+tm], color='r', ls='--')
            py.plot([col[ch], col[ch]+time[-1]], [depth[ch], depth[ch]], color='grey', ls='--')
    if draw:
        py.ylabel('depth', fontsize=20)
        py.xlabel('column', fontsize=20)
    return sweeps

def draw_signal_npy(col, depth, sweeps, shift=64):
    time = np.linspace(0,180,int(300*crct)) 
    py.figure()
    py.subplot(111)
    py.suptitle('example 2 sec from file: '+name)
    for ch in range(64):
        norm_sig = sweeps[ch].mean(axis=0)
        py.plot(time+col[ch],norm_sig/5+depth[ch], color='black')
        py.plot([col[ch]+45, col[ch]+45], [depth[ch]-45, depth[ch]+45], color='r', ls='--')
        py.plot([col[ch], col[ch]+180], [depth[ch], depth[ch]], color='grey', ls='--')
    py.ylabel('depth', fontsize=20)
    py.xlabel('column', fontsize=20)
#%%
savedir = './npys/'
datadir = 'H:/SOVy/SOV17/'
sovs = ['01wasy_', '02wasy_', '07wasy_', '05wasy_', '06wasy_']
# sovs= ['09wasy_','10wasy_']
savename = ['17_crtx1000_th1500mua', '17_crtx2200_th5000mua', 
             '17_crtx2000_th5700mua', '17_crtx2000_th6700mua','17_crtx2000_th7700mua']
parts = ['_th', '_crtx']
resamp = 2
for nn, sov in enumerate(sovs):
    Fs=10000
    lay = pd.read_csv('A8x8.lay', sep=',', header = None)
    lay_mir = pd.read_csv('A8x8_mirror.lay', sep=' ', header = None)
    mapa1 = pd.read_csv('samstac40_NeuroNexus_A1_8x8_EK.map',sep=' ', header=None)
    name = [i for i in os.listdir(datadir) if sov in i][0]
    file = h5py.File(datadir+name, 'r')
    print('file len: ', len(file.keys())/2/60, ' min')
    sig = load_frag(file, start=20, stop=500, resamp=resamp)
    col = lay[1].values
    depth = lay[2].values
    for part in parts:
        crct = Fs/100
        if part=='th': lay, tm, vmax = lay_mir, 145, .25
        else: tm, vmax=200, 5
        pots = draw_signal(col,depth,part,draw=0)
        # pots = pots.mean(axis=1)
        np.save(savedir+savename[nn]+part,pots)
        # ele_pos = np.array([col,depth]).T
        # k = KCSD2D(ele_pos,pots,xmin=-100,xmax=1700, ymin=-200,ymax=1700)
        # k.L_curve(lambdas=np.linspace(1e-8,1e-3,1), Rs=[600])
        # csd = k.values('CSD')
        # py.contourf(k.estm_x, k.estm_y, csd[:,:,tm], cmap='bwr', vmin=-vmax, vmax=vmax)
        # for i in range(64):
            # py.plot([col[i]+tm/625*180, col[i]+tm/625*180], [depth[i]-40, depth[i]+40], color='grey', ls='--')
        # datadir2 = datadir+datadir[-6:-1]+'_profiles_and_csd/'
        # py.savefig(datadir2+sov+part)
        # py.close()
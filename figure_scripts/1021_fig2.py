# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:02:31 2020

@author: Wladek
"""
import numpy as np 
import matplotlib.pyplot as py
import matplotlib as mpl
from matplotlib.colors import LogNorm
from figure_properties import *
import matplotlib.patches as mpatches
from scipy.signal import welch, argrelmin, butter, filtfilt, spectrogram, detrend
import scipy.io

py.close('all')
# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def exp_design(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    img = py.imread(loadir+name)
    ax.imshow(img, aspect='auto')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def st_profile(po, pos, name, csd, title='', key='a', vmax=1000):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    pots = scipy.io.loadmat(loadir_mat+name)[csd]
    print(pots.shape)
    py.title(title)
    cmap ='bwr'
    x,y = np.meshgrid(np.linspace(-5,25,pots.shape[1]),
                      np.linspace(7.4,0,pots.shape[0]))
    if 'LFP' in title: cmap='PRGn'
    # py.contourf(x,y,pots, cmap=cmap, levels=np.linspace(-vmax,vmax,201))
    py.imshow(pots, cmap=cmap, vmin=-vmax, vmax=vmax, extent=[-5,25,7.4,0], 
              origin='lower', aspect='auto')
    cbar = py.colorbar(aspect=50, pad=0)
    cbar.formatter.set_scientific(False)
    cbar.ax.tick_params(size=0)
    
    py.xlabel('Time (ms)', labelpad=-1.5), 
    if po[0]=='A': 
        py.ylabel('Channel depth (mm)')
    py.axvline(0, ls='--', lw = 2, color='grey')
    if 'LFP' in title:
        cont_lfp1 = abs(pots)>vmax
        pots[:,:80]=0
        py.contour(x,y,cont_lfp1*abs(pots), levels=np.linspace(-1,2.5,10), 
                   cmap="Greys_r", linestyles='dashed', linewidth=.2)
    if 'CSD' in title:
        cont_lfp1 = abs(pots)>vmax
        pots[:,:80]=0
        py.contour(x,y,cont_lfp1*abs(pots), levels=np.linspace(-200,200,20), 
                   cmap="Greys_r", linestyles='dashed', linewidth=.1)
        py.plot([-5,-5,25.2,25.2,-5], [2.5,0,0,2.5,2.5], color='grey', lw=6)
        py.xlim(-5,25)
    #     pots[150:]=0
    #     cont_lfp2 = abs(pots)>3
    #     py.contour(x,y,cont_lfp2*abs(pots), levels=[0,1], cmap="Oranges", linestyles='dashed')
    # # py.axvline(-5, lw = 6, color='red')
    ax.spines['right'].set_visible(False)

loadir='./fig2_files/'
loadir_mat = './mats/'
fig = py.figure(figsize=(20,10))
gs = fig.add_gridspec(8,17)
# exp_design(('A',1.05), (0,10,1,6), 'hist_mtrx.png')
st_profile(('A',1.05), (0,8,0,5), 'an_sov19.mat', csd = 'pots', 
           title='LFP', vmax=.5)
st_profile(('B',1.05), (0,8,6,11), 'an_sov19.mat', csd='csd',
           title='CSD reconstruction from the cortex', vmax=20)
st_profile(('C',1.05), (0,8,12,17), 'an_sov19.mat', csd = 'pot_VC', 
           title='Volume conducted LFP from the cortex',vmax=.5)
# lfp_profile(('E',1.1), (12,20,8,13), 'sov6events_t.npy', 'thalamic', .1)
# exp_design(('F',1), (12,20,14,20), 'pipline.png')
# py.tight_layout()
py.savefig('fig2_new')
# py.close()
 
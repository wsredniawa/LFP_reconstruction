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
import matplotlib.ticker as ticker

py.close('all')
# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def exp_design(po, pos, name, text,par):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.05, po[1], letter= po[0])
    img = py.imread('./fig5_files/'+name)
    ax.imshow(img, aspect='auto', extent=[0,1,0,1])
    # if po[0]=='B':
        # py.xlim(0.2,0.7), py.ylim(0.2,.9)
    # else:
        # py.xlim(0.2,0.7), py.ylim(0.1,.9)
    ax.text(par,-5.5, text, fontsize=30)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def lfp_profile2(po, pos, name, title, tp=100, vmax=1e3,rat='16'):
    global pots,ele_pos,inds, cont_lfp1
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, -.23, po[1], letter= po[0])
    csd2 = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')[name]
    ele_pos = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['ele_pos']
    plane = scipy.io.loadmat('./mats/an_sov'+rat+'.mat')['est_plane']
    py.title(title, pad=15, fontsize=15)
    # b,a = butter(2, [2/(Fs/2), 300/(Fs/2)], btype='bandpass')
    # pots = filtfilt(b,a,pots)
    img = py.imread('fig_5_sov'+str(i)+'_histologia.png')
    if po[0]=='A1':
        py.ylabel('5 ms after stimulus', fontsize=25, labelpad=25)
    if po[0]=='A2':
        py.ylabel('10 ms after stimulus', fontsize=25, labelpad=25)
    if po[0]=='A3':
        py.ylabel('20 ms after stimulus', fontsize=25, labelpad=25)
    ymin,ymax = plane[1].min(), plane[1].max()
    zmin,zmax = plane[2].min(), plane[2].max()
    
    X,Y = np.meshgrid(np.linspace(ymin, ymax, 20), np.linspace(zmax, zmin, 38))
    cmap='bwr'
    if 'pot' in name: cmap = 'PRGn'
    plot = py.contourf(X,Y,csd2[:,::1,tp].reshape((38,20)),levels=np.linspace(-vmax,vmax,101),
                     cmap=cmap, alpha=.3)
    cbar=py.colorbar(plot, pad=0, ticks=[-vmax, 0, vmax])
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    py.scatter(ele_pos[1,th_start:], ele_pos[2,th_start:], s=.2, color='k')
    py.imshow(img[:,::1], extent=[ymin, ymax, zmax, zmin], alpha=1)
    # py.yticks([1,2,3,4,5,6,7],[7,6,5,4,3,2,1])
    if 'pot' in name:
        pots = csd2[:,::1,tp].reshape((38,20))
        cont_lfp1 = pots<-1e2
        py.contour(X,Y,cont_lfp1*pots, levels=np.linspace(-8e2,8e2,31), 
                   cmap="PRGn", linestyles='dashed', linewidth=.1)
        # py.imshow(csd_to_plot, extent=[-2.4,-.8,6.9,.1],aspect='auto',
          # origin='lower', vmax=vmax, vmin=-vmax, cmap='bwr', alpha=1)
    # ax.invert_yaxis()
    # cbar.formatter.set_powerlimits((10, 10))    
    if '3' in po[0]:
        py.xlabel('M<->L axis')
    else:
        py.xticks([])
   
    # if po[0]=='B':py.ylabel('D<->V axis')
    ax.spines['right'].set_visible(False)
    # py.axvline(0, ls='--', lw = 2, color='grey')

loadir='./fig4_files/'
loadir2='./fig6_files/'
tp=100
tp2=150
tp3=250
hght = 1.03
for i in ['16','17','18']:
    fig = py.figure(figsize=(24,24), dpi=150)
    gs = fig.add_gridspec(36,40)
    rat,th_start,vmax=i, 88, 5e3
    vmax2=4e3
    if rat=='18': 
        th_start=96
        vmax=3e3
    if rat=='16':
        th_start=86
        vmax=3e3
    exp_design(('A',1.07), (0,6,0,7), 'th_crtx_th_src.png', 
               '',par=-.3)
    exp_design(('B',1.07), (0,6,8,15),'th_th_src.png', 
                '', par=-.1)
    exp_design(('C',1.07), (0,6,17,24),'th_full_src.png', 
                '', par=-.1)
    exp_design(('D',1.07), (0,6,25,32),'th_crtx_full_src.png', 
                '', par=-.1)
    exp_design(('E',1.07), (0,6,33,42),'th_crtx_full_src.png', 
                '', par=-.1)
    
    
    lfp_profile2(('A1',hght), (7,16,0,7), 'csd_B', 'Cortical and thalamic',vmax=2e4, tp=tp,rat=i)
    lfp_profile2(('B1',hght), (7,16,8,15), 'csd_C', 'Only thalamic', vmax=vmax, tp=tp, rat=i)
    
    lfp_profile2(('A2',hght), (17,26,0,7), 'csd_B', '',vmax=1e5, tp=tp2,rat=i)
    lfp_profile2(('B2',hght), (17,26,8,15), 'csd_C', '', vmax=vmax, tp=tp2, rat=i)
    
    lfp_profile2(('A3',hght), (27,36,0,7), 'csd_B', '',vmax=4e4, tp=tp3, rat=i)
    lfp_profile2(('B3',hght), (27,36,8,15), 'csd_C', '', vmax=vmax, tp=tp3, rat=i)
    
    
    lfp_profile2(('D1',hght), (7,16,25,32), 'csd_E', 'Cortical and thalamic',vmax=vmax2, tp=tp, rat=i)
    lfp_profile2(('E1',hght), (7,16,33,40), 'csd_Fpot', 'Cortical and thalamic LFP',vmax=1.2e2, tp=tp,rat=i)
    lfp_profile2(('C1',hght), (7,16,17,24), 'csd_F', 'Only thalamic',vmax=vmax2, tp=tp, rat=i)
    
    lfp_profile2(('D2',hght), (17,26,25,32), 'csd_E', '',vmax=vmax2, tp=tp2, rat=i)
    lfp_profile2(('E2',hght), (17,26,33,40), 'csd_Fpot', '',vmax=1e2, tp=tp2, rat=i)
    lfp_profile2(('C2',hght), (17,26,17,24), 'csd_F', '',vmax=vmax2, tp=tp2, rat=i)
    
    lfp_profile2(('D3',hght), (27,36,25,32), 'csd_E', '',vmax=vmax2, tp=tp3, rat=i)
    lfp_profile2(('E3',hght), (27,36,33,40), 'csd_Fpot', '',vmax=3e2, tp=tp3, rat=i)
    lfp_profile2(('C3',hght), (27,36,17,24), 'csd_F', '',vmax=vmax2, tp=tp3, rat=i)
    
    py.savefig('fig5_new'+rat)
    py.close('all')
 
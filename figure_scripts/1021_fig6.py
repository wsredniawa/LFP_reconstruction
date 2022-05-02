# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:02:31 2020

@author: Wladek
"""

import numpy as np 
import matplotlib.pyplot as py

from matplotlib.colors import LogNorm
from figure_properties import *
import matplotlib.patches as mpatches
from scipy.signal import welch, argrelmin, butter, filtfilt, spectrogram, detrend
from scipy.stats import wilcoxon, f_oneway, ttest_rel, sem, shapiro
import scipy.io
from mlxtend.evaluate import permutation_test
py.close('all')
import matplotlib as mpl
# mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def exp_design(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    img = py.imread(name)
    ax.imshow(img, aspect='auto', extent=[0,1,0,1])
    # py.xlim(0.1,0.9)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def ex_lfp2(po, pos, name1,name2, channel=255,div=2, Fs=5000):
    global recon_th
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    pots = scipy.io.loadmat(name1)['pots']
    mn_pots,std_pots = pots.mean(axis=1), sem(pots,axis=1)
    recon_th = scipy.io.loadmat(name1)['pots_est_th']
    mn_rth,std_rth = np.mean(recon_th, axis=0), sem(recon_th,axis=0)
    potslid = scipy.io.loadmat(name2)['pots']
    mn_plid,std_plid = np.mean(potslid, axis=1), sem(potslid,axis=1)
    recon_thlid = scipy.io.loadmat(name2)['pots_est_th']
    mn_rlid,std_rlid = np.mean(recon_thlid, axis=0), sem(recon_thlid,axis=0)
    # mn_pots[channel] -= np.mean(mn_pots[channel,:25])
    # mn_rth[channel] -= np.mean(mn_rth[channel,:25])
    # mn_plid[channel] -= np.mean(mn_plid[channel,:25])
    # mn_rlid[channel] -= np.mean(mn_rlid[channel,:25])
    
    loc_time = np.linspace(-5,25, pots.shape[2])
    py.plot(loc_time, mn_pots[channel], color='grey',label='control')
    py.fill_between(loc_time, mn_pots[channel]-std_pots[channel],mn_pots[channel]+std_pots[channel], 
                    color='grey',alpha=.3)
    py.plot(loc_time, mn_rth[channel], color='black',label='recon. control', ls='--')
    py.fill_between(loc_time, mn_rth[channel]-std_rth[channel],mn_rth[channel]+std_rth[channel], 
                    color='black',alpha=.3)
    py.plot(loc_time, mn_plid[channel], color='red',label='ligno.')
    py.fill_between(loc_time, mn_plid[channel]-std_plid[channel],mn_plid[channel]+std_plid[channel], 
                    color='red',alpha=.3)
    py.plot(loc_time, mn_rlid[channel], color='darkred',label='recon. ligno.' ,ls='--')
    py.fill_between(loc_time, mn_rlid[channel]-std_rlid[channel],mn_rlid[channel]+std_rlid[channel], 
                    color='darkred',alpha=.3)
    
    
    ax.set_xlabel('Time (ms)'),ax.set_ylabel('Potential (mV)')
    ax.spines['right'].set_visible(False)
    py.legend(ncol=1,loc=4, frameon = False, fontsize = 16)

def lfp_profile(po, pos, name, title, vmax=200):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    lfp = scipy.io.loadmat(name)['pots']
    ele_pos = scipy.io.loadmat('./mats/sov19.mat')['ele_pos'].T
    lfp = lfp.mean(axis=1)
    py.title(title)
    time= np.linspace(-5,25,lfp.shape[1])
    for i in range(lfp.shape[0]-1,0,-1):
        py.plot(time, lfp[i]/2+ele_pos[i,2]-11, color='black', linewidth=.5)
        if i==channel:
            py.plot(time, lfp[i]/2+ele_pos[i,2]-11, color='black', linewidth=3)   
    py.axvline(0, ls='--', color='grey')
    py.ylabel('Electrode number')
    py.ylim(-7.5,.5)

    py.xlabel('Time (ms)'),py.ylabel('Depth (mm)')
    ax.spines['right'].set_visible(False)
    py.axvline(0, ls='--', lw = 2, color='grey')
    if po[0]=='B':
        py.yticks([]),py.ylabel('')
        py.plot([25.8,25.8],[0,-0.5], 'k',lw=3)
        py.text(26.4,-0.2, '1 mV', size=15)
    
def comparison(po, pos, name1,name2, channel=255,div=2, Fs=5000):
    global recon_th
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    pots = scipy.io.loadmat(name1)['pots']
    # mn_pots,std_pots = pots.mean(axis=1), sem(pots,axis=1)
    rth = scipy.io.loadmat(name1)['pots_est_th']
    # mn_rth,std_rth = np.mean(recon_th, axis=0), sem(recon_th,axis=0)
    plid = scipy.io.loadmat(name2)['pots']
    # mn_plid,std_plid = np.mean(potslid, axis=1), sem(potslid,axis=1)
    rlid = scipy.io.loadmat(name2)['pots_est_th']
    # mn_rlid,std_rlid = np.mean(recon_thlid, axis=0), sem(recon_thlid,axis=0)
    # mn_pots[channel] -= np.mean(mn_pots[channel,:25])
    # mn_rth[channel] -= np.mean(mn_rth[channel,:25])
    # mn_plid[channel] -= np.mean(mn_plid[channel,:25])
    # mn_rlid[channel] -= np.mean(mn_rlid[channel,:25])
    p_values=np.zeros((3, pots.shape[-1]))
    loc_time = np.linspace(-5,25, pots.shape[-1])
    for i in range(p_values.shape[1]):
        w, p_values[0,i] = f_oneway(pots[channel,:,i], rth[:,channel,i])
        w, p_values[1,i] = f_oneway(plid[channel,:,i], rlid[:,channel,i])
        w, p_values[2,i] = f_oneway(rth[:,channel,i], rlid[:,channel,i])
        # p_values[0,i] = permutation_test(pots[channel,:,i], rth[:,channel,i], 
                                           # paired=False, method="approximate", seed=0, num_rounds=1000)
        # p_values[1,i] = permutation_test(plid[channel,:,i], rlid[:,channel,i], 
                                           # paired=False, method="approximate", seed=0, num_rounds=1000)
    py.plot(loc_time, np.log10(p_values[0]), color= 'k', label='control pval.')
    py.plot(loc_time, np.log10(p_values[1]), color= 'r', label='ligno. pval.')
    py.plot(loc_time, np.log10(p_values[2]), color= 'grey', label='recon. ligno vs recon. ctrl')
    py.ylabel('log(p-value)'),py.legend()
    py.axhline(np.log10(0.05), ls='--',color='grey')
    ax.spines['right'].set_visible(False)
        
    
loadir='./fig4_files/'
loadir2='./fig6_files/'
channel=131
fig = py.figure(figsize=(23,10), dpi=220)
gs = fig.add_gridspec(10, 20)
# exp_design(('A',1.07), (0,6,0,8), 'th_space.png')
# correlation_map(('B',1.08), (7,16,0,10), 'cor_score_th18.npy')
# cor_map_stat(('A',1.05), (0,4,0,10), title='Cortical channels')
# cor_map_stat(('B',1.05), (6,10,0,10), title='Thalamic channels')
# ex_lfp(('C',1.05), (12,16,0,10))

rat='21'
# lfp_profile(('A',1.01), (0,10,0,4), './mats/an_multi_sov21.mat','Control',channel=channel)
# lfp_profile(('B',1.01), (0,10,5,9), './mats/an_multi_sov21lid.mat','Lignocaine',channel=channel)
exp_design(('C',1.03), (0,10,7,10), 'fig6_hist.png')
# exp_design(('B',1.07), (12,20,0,6), 'hist.png')
# ex_lfp(('D',1.05), (0,9,14,20), 'rat18_EP.npy')
lfp_profile(('A',1.03), (0,10,0,3), './mats/an_multi_sov21.mat', 'control')
lfp_profile(('B',1.03), (0,10,3,6), './mats/an_multi_sov21lid.mat', 'lignocaine')

ex_lfp2(('D',1.05), (0,7,11,20), './mats/an_multi_sov21.mat',
        './mats/an_multi_sov21lid.mat', channel=channel)

comparison(('E',1.05), (8,10,11,20), './mats/an_multi_sov21.mat',
           './mats/an_multi_sov21lid.mat', channel=channel)
# py.savefig('fig5_sov'+rat+'ch_'+str(channel))
py.savefig('fig6_new')
py.close()
 
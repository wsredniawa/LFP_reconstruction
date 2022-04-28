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
    img = py.imread(loadir2+name)
    ax.imshow(img, aspect='auto', extent=[0,1,0,1])
    # py.xlim(0.1,0.9)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.xticks([]), py.yticks([])

def correlation_map(po, pos, name):
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    cor_score = np.load(loadir2+name)
    print(cor_score.shape)
    ele_pos = np.load(loadir+'ele_pos18.npy')
    ele_pos_sort = np.argsort(ele_pos[:,2])
    py.title('Rolling correlation score map for all electrodes')
    py.imshow(cor_score[list(ele_pos_sort)], vmax=1, vmin=-1, cmap='PiYG',
              aspect='auto', extent=[-5,20,cor_score.shape[0],1])
    py.ylabel('channels sorted in ML axis')
    py.colorbar(orientation='horizontal', pad=.13,aspect=50)  
    py.xlabel('Time (ms)', labelpad=-1.5)
    py.axhline(103, lw = 6, color='blue')
    py.axhline(105, lw = 4, color='orange')
    py.axhline(145, ls='--', lw = 4, color='red')
    py.axhline(215, ls='--', lw = 4, color='red')
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    return list(ele_pos_sort)

def cor_map_stat(po, pos, typ='cs_th',typ2='cs_crtx', name='', title=''):
    global df, th
    tps, lp = list(np.arange(28)), 1
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    loadir='./mats/an_'
    cor_score_th = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ], 
                             scipy.io.loadmat(loadir+'sov17.mat')[typ],
                             scipy.io.loadmat(loadir+'sov18.mat')[typ], 
                             scipy.io.loadmat(loadir+'sov01.mat')[typ],
                             scipy.io.loadmat(loadir+'sov02.mat')[typ],
                             scipy.io.loadmat(loadir+'sov03.mat')[typ],
                             scipy.io.loadmat(loadir+'sov06.mat')[typ],
                             scipy.io.loadmat(loadir+'sov09.mat')[typ],
                             scipy.io.loadmat(loadir+'sov19.mat')[typ],
                             scipy.io.loadmat(loadir+'sov20.mat')[typ],
                             scipy.io.loadmat(loadir+'sov21.mat')[typ]])
    cor_score_crtx = np.array([scipy.io.loadmat(loadir+'sov16.mat')[typ2], 
                             scipy.io.loadmat(loadir+'sov17.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov18.mat')[typ2], 
                             scipy.io.loadmat(loadir+'sov01.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov02.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov03.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov06.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov09.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov19.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov20.mat')[typ2],
                             scipy.io.loadmat(loadir+'sov21.mat')[typ2]])
    n_rats = cor_score_th.shape[0]
    th_th, crtx_th = np.zeros((n_rats,len(tps)+2,lp)), np.zeros((n_rats,len(tps)+2,lp))
    th_crtx, crtx_crtx = np.zeros((n_rats,len(tps)+2,lp)), np.zeros((n_rats,len(tps)+2,lp))
    for n,i in enumerate(tps):
        crtx_th[:, n] = (cor_score_th[0][60, i:i+lp], 
                      cor_score_th[1][78, i:i+lp],
                      cor_score_th[2][78, i:i+lp],
                      cor_score_th[3][6, i:i+lp],
                      cor_score_th[4][7, i:i+lp],
                      cor_score_th[5][7, i:i+lp],
                      cor_score_th[6][5, i:i+lp],
                      cor_score_th[7][8, i:i+lp],
                      cor_score_th[8][320, i:i+lp],
                      cor_score_th[9][330, i:i+lp],
                      cor_score_th[10][320, i:i+lp])
        
        th_th[:, n] = (cor_score_th[0][114, i:i+lp], 
                    cor_score_th[1][158, i:i+lp],
                    cor_score_th[2][152, i:i+lp],
                    cor_score_th[3][28, i:i+lp],
                    cor_score_th[4][27, i:i+lp],
                    cor_score_th[5][17, i:i+lp],
                    cor_score_th[6][15, i:i+lp],
                    cor_score_th[7][26, i:i+lp],
                    cor_score_th[8][124, i:i+lp],
                    cor_score_th[9][134, i:i+lp],
                    cor_score_th[10][130, i:i+lp])  
        
        crtx_crtx[:, n] = (cor_score_crtx[0][60, i:i+lp], 
                      cor_score_crtx[1][78, i:i+lp],
                      cor_score_crtx[2][78, i:i+lp],
                      cor_score_crtx[3][6, i:i+lp],
                      cor_score_crtx[4][7, i:i+lp],
                      cor_score_crtx[5][7, i:i+lp],
                      cor_score_crtx[6][5, i:i+lp],
                      cor_score_crtx[7][8, i:i+lp],
                      cor_score_crtx[8][320, i:i+lp],
                      cor_score_crtx[9][330, i:i+lp],
                      cor_score_crtx[10][320, i:i+lp])
        
        th_crtx[:, n] = (cor_score_crtx[0][114, i:i+lp], 
                    cor_score_crtx[1][158, i:i+lp],
                    cor_score_crtx[2][152, i:i+lp],
                    cor_score_crtx[3][28, i:i+lp],
                    cor_score_crtx[4][27, i:i+lp],
                    cor_score_crtx[5][17, i:i+lp],
                    cor_score_crtx[6][15, i:i+lp],
                    cor_score_crtx[7][26, i:i+lp],
                    cor_score_crtx[8][124, i:i+lp],
                    cor_score_crtx[9][134, i:i+lp],
                    cor_score_crtx[10][130, i:i+lp])
    
    time_scale=np.linspace(-5,24,30)
    if 'Cortical' in title:
        mn_crtx, std_crtx = crtx_crtx.mean(axis=0)[:,0],crtx_crtx.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_crtx, color='navy', lw=2, label='Cortical sources')
        py.fill_between(time_scale, mn_crtx-std_crtx, mn_crtx+std_crtx, alpha=.3, color='blue')
        mn_th, std_th = crtx_th.mean(axis=0)[:,0], crtx_th.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_th, color='brown', lw=2,label='Thalamic sources')
        py.fill_between(time_scale, mn_th-std_th, mn_th+std_th, alpha=.3, color='brown')
    if 'Thalamic' in title:
        mn_crtx, std_crtx = th_crtx.mean(axis=0)[:,0],th_crtx.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_crtx, color='navy', lw=2, label='Cortical sources')
        py.fill_between(time_scale, mn_crtx-std_crtx, mn_crtx+std_crtx, alpha=.3, color='blue')
        mn_th, std_th = th_th.mean(axis=0)[:,0], th_th.std(axis=0)[:,0]/(n_rats**(.5))
        py.plot(time_scale, mn_th, color='brown', lw=2,label='Thalamic sources')
        py.fill_between(time_scale, mn_th-std_th, mn_th+std_th, alpha=.3, color='brown')
    sov_labels = [16,17,18,1,2,3,6,9,19,20,21]
    # for i in range(0,11,1): 
        # py.plot(time_scale, th[i], lw=2, color='indianred')
        # py.plot(time_scale, crtx[i], lw=2, color='navy')
    ax.set_ylabel('Correlation'),ax.set_xlabel('Time (ms)')
    py.grid()
    py.legend(loc=4) 
    py.title(title)
    py.ylim(-1.1,1.1), py.xlim(-5,22)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    
def ex_lfp(po, pos, div=1, div2=1):
    global recon_th
    ax = fig.add_subplot(gs[pos[0]:pos[1], pos[2]:pos[3]])
    set_axis(ax, 0, po[1], letter= po[0])
    import scipy.io
    pots = scipy.io.loadmat('./mats/an_sov19.mat')['pots']
    recon_crtx = scipy.io.loadmat('./mats/an_sov19.mat')['pots_est_crtx']
    recon_th = scipy.io.loadmat('./mats/an_sov19.mat')['pots_est_th']
    Fs = 10000
    ch_th,ch_crtx=124,320
    loc_time = np.linspace(-5,25, pots.shape[1])
    pots1 = pots[ch_crtx]-pots[ch_crtx,:25].mean()
    rec_cortex = recon_crtx[ch_crtx]-recon_crtx[ch_crtx,:25].mean()
    py.plot(loc_time, pots1, color='navy')
    py.plot(loc_time, rec_cortex, color='skyblue', ls='--')
    ax.tick_params(axis='y', labelcolor='navy')
    ax2 = ax.twinx()
    color = 'orange'
    pots2 = pots[ch_th]-pots[ch_th,:25].mean()
    ax2.plot(loc_time, pots2, color=color)
    ax2.plot(loc_time, recon_th[ch_th]-recon_th[ch_th,:25].mean(), color='red', ls='--')
    ax2.set_ylim(-.5,.5), ax.set_ylim(-5,5)
    ax2.tick_params(axis='y', labelcolor='orange')
    # ax2.text(3.4,-.25, '*')
    # ax2.text(9.2,-.5, '**')
    ax.set_xlabel('Time (ms)'),ax.set_ylabel('Potential (mV)')
    ax2.set_xlim(-5,25)
    py.axvline(0, ls='--', lw = 2, color='grey')
    # py.axvline(10, ls='--', lw = 2, color='purple')
    th_pot = mpatches.Patch(color=color, label='Thalamic EP')
    crtx_pot = mpatches.Patch(color='navy', label='Cortical EP')
    th_recon = mpatches.Patch(color='red', label='Thalamic recon.')
    crtx_recon = mpatches.Patch(color='skyblue', label='Cortical recon.')
    py.legend(handles=[th_pot, crtx_pot, th_recon, crtx_recon], ncol=1,loc=4, frameon = False, fontsize = 10)

loadir='./fig4_files/'
loadir2='./fig6_files/'
channel=130
fig = py.figure(figsize=(10,14), dpi=200)
gs = fig.add_gridspec(16, 10)
# exp_design(('A',1.07), (0,6,0,8), 'th_space.png')
# correlation_map(('B',1.08), (7,16,0,10), 'cor_score_th18.npy')
cor_map_stat(('B',1.05), (6,10,0,10), title='Cortical channels')
cor_map_stat(('C',1.05), (12,16,0,10), title='Thalamic channels')
ex_lfp(('A',1.05), (0,4,0,10))

# rat='21'
# lfp_profile(('D',1.01), (0,8,12,16), './mats/an_sov21.mat','Control',channel=channel)
# lfp_profile(('E',1.01), (0,8,17,21), './mats/an_sov21lid.mat','Lignocaine',channel=channel)
# ex_lfp2(('F',1.05), (10,16,13,21), './mats/an_sov21.mat','./mats/an_sov21lid.mat', channel=channel)
# py.savefig('fig5_sov'+rat+'ch_'+str(channel))
py.savefig('fig4_new')
py.close()
 
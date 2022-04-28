# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:12:06 2020

@author: Wladek
"""
import numpy as np #wczytanie biblioteki z funkcjami matematycznymi
import matplotlib.pyplot as py #wczytanie biblioteki do rysowania
from scipy.signal import filtfilt, butter, detrend
from scipy.stats import pearsonr
import scipy.io
from neo.io import Spike2IO
import mat73
#%%
crtx_ch = 22
cor_b = np.array([np.linspace(-1.7, -2.1,crtx_ch),
                  np.linspace(5.2, 4.1,crtx_ch),
                  np.linspace(1.7, 3.5, crtx_ch)])
cor_t = np.array([np.linspace(-2.5, -3, 16),
                  np.linspace(3.3, 2.5, 16),
                  np.linspace(4.9, 6.7, 16)])
#%% 
loadir = './sov/sov09/'

b1_name ='1_1400_1500srednia.mat'#2_BCX2000.smr'
b2_name ='2_2800_2900srednia.mat'#2_BCX2000.smr'
t1_name= '4_5600_5700srednia.mat'

b1 = mat73.loadmat(loadir+b1_name)
b2 = mat73.loadmat(loadir+b2_name)
t1 = mat73.loadmat(loadir+t1_name)
# t2 = mat73.loadmat(loadir+t2_name)

#%%
b1_chs = list(b1.keys())
b2_chs = list(b2.keys())
t1_chs = list(t1.keys())
# t2_chs = list(t2.keys()) 
b1b,b2b,t1b=np.zeros((3,16,300))
ch_num=16
ind=21
Fs=10000
start,stop=3955, 4255
for ch in range(16):
        true_ch = int(b1_chs[ch][ind:])
        b1b[true_ch-1] = b1[b1_chs[ch]]['values'][start:stop]
        b2b[true_ch-1] = b2[b2_chs[ch]]['values'][start:stop]
        t1b[true_ch-1] = t1[t1_chs[ch]]['values'][start:stop]
trials_b1 = np.delete(b1b, [1,2,11,12,13,14,15], axis=0)
trials_b2 = np.delete(b2b, [14,15], axis=0)
trials_b = np.concatenate((trials_b1,trials_b2))
trials_t = np.delete(t1b, [14],axis=0)
#%%
ele_pos = np.concatenate((cor_b,cor_t), axis=1)
lfp_s = np.concatenate((trials_b,trials_t))
frags = 30
leng= 50
leap=10
ch_stay=7
cor_score = np.zeros((ele_pos.shape[1],frags))
cor_score_crtx, cor_score_th = np.zeros((ele_pos.shape[1],frags)), np.zeros((ele_pos.shape[1],frags))
for part in range(frags-5):
    for ch in range(ele_pos.shape[1]):
        cor_score[ch,part] = pearsonr(lfp_s[ch_stay,leap*part:leap*part+leng], lfp_s[ch,leap*part:leap*part+leng])[0]
py.figure(figsize=(8,8))
py.subplot(211)
py.title('LFP')
py.imshow(lfp_s, vmin=-.1, vmax=.1, extent=[-5,25,ele_pos.shape[1],1], aspect='auto', cmap='PRGn')
py.ylabel('channels'), py.xlabel('time')
py.subplot(212)
py.title('Correlation')
py.imshow(cor_score, vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],1],aspect='auto', cmap='PiYG')
py.ylabel('channels'), py.xlabel('time')
py.savefig('sov'+loadir[-3:-1]+'_load')
scipy.io.savemat('./mats/sov' +loadir[-3:-1]+'.mat', 
                 dict(crtx=trials_b, th=trials_t,
                      cor=cor_score, ele_pos=ele_pos))
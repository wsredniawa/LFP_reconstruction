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

def ldfs(params, channels=16):
    r = Spike2IO(filename=params)
    seg = r.read_segment()
    t_stop = np.float(seg.t_stop)
    Fs = np.float(seg.analogsignals[0].sampling_rate)
    sig=np.zeros((channels, int(t_stop*Fs)))
    for i in range(channels):
        sig[i] = np.array(seg.analogsignals[0][:int(t_stop*Fs),i]).flatten()
    event_list=[]
    for i in seg.events[1]:
        event_list.append(np.float(i))
    return sig, Fs, t_stop, event_list
#%%
crtx_ch = 22
cor_b = np.array([np.linspace(-2.6, -2.6,crtx_ch),
                  np.linspace(5.7, 4.6,crtx_ch),
                  np.linspace(1.8, 3.6, crtx_ch)])
cor_t = np.array([np.linspace(-2.6, -2.6, 16),
                  np.linspace(3.3, 2.5, 16),
                  np.linspace(5.5, 6.7, 16)])
#%% 
loadir = './sov/sov02/'

b1_name ='01_1500_all.smr'#2_BCX2000.smr'
b2_name ='02_2300_all.smr'#2_BCX2000.smr'
t1_name= '06_5500_all.smr'
sig_B1,Fsb,t_stop, m1a = ldfs(loadir+b1_name, 32)
sig_B2,Fsb,t_stop, m1b = ldfs(loadir+b2_name, 32)
sig_T1,Fst,t_stop, m2a = ldfs(loadir+t1_name, 32)
Fs = int(Fsb)
#%%
back, forward=50, 250
trials_b = np.zeros((crtx_ch,100,back+forward))
trials_t = np.zeros((16,100,back+forward))
shf=-30
for i in range(100):
    t1a=m1a[i]
    t1b=m1b[i]
    t2a=m2a[i]
    trials_b[:14,i] = sig_B1[:14,int(t1a*Fs-back-shf):int(t1a*Fs+forward-shf)]
    trials_b[14:,i] = sig_B2[7:15,int(t1b*Fs-back-shf):int(t1b*Fs+forward-shf)]
    trials_t[:,i] = sig_T1[:16,int(t2a*Fs-back-shf):int(t2a*Fs+forward-shf)]
    # trials_t[:,i] = sig_T2[16:,int(t2b*Fs-Fs/4-shf):int(t2b*Fs+Fs*3/4-shf)][:,:Fs]
trials_b = np.delete(trials_b, [4],axis=0)
trials_t = np.delete(trials_t, [4],axis=0)
cor_b = np.delete(cor_b,[4],axis=1)
cor_t = np.delete(cor_t,[4],axis=1)
#%%
ele_pos = np.concatenate((cor_b,cor_t), axis=1)
lfp_s = np.concatenate((trials_b.mean(axis=1),trials_t.mean(axis=1)))
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
py.imshow(lfp_s, vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],1], aspect='auto', cmap='PRGn')
py.ylabel('channels'), py.xlabel('time')
py.subplot(212)
py.title('Correlation')
py.imshow(cor_score, vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],1],aspect='auto', cmap='PiYG')
py.ylabel('channels'), py.xlabel('time')
py.savefig('sov'+loadir[-3:-1]+'_load')
scipy.io.savemat('./mats/sov' +loadir[-3:-1]+'.mat',
                 dict(crtx=trials_b.mean(axis=1), th=trials_t.mean(axis=1),
                      cor=cor_score, ele_pos=ele_pos))
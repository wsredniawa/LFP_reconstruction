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

def ldfs(params, channels=16, event_ind=1):
    global seg
    r = Spike2IO(filename=params)
    seg = r.read_segment()
    t_stop = np.float(seg.t_stop)
    Fs = np.float(seg.analogsignals[0].sampling_rate)
    sig=np.zeros((channels, int(t_stop*Fs)))
    for i in range(channels):
        sig[i] = np.array(seg.analogsignals[0][:int(t_stop*Fs),i]).flatten()
    event_list=[]
    for i in seg.events[event_ind]:
        event_list.append(np.float(i))
    return sig, Fs, t_stop, event_list
#%%
crtx_ch = 14
cor_b = np.array([np.linspace(-1.8, -1.8,crtx_ch),
                  np.linspace(4.8, 3.8,crtx_ch),
                  np.linspace(1, 3.1, crtx_ch)])
cor_t1 = np.array([np.linspace(-3.3, -3.3, 7),
                   np.linspace(3.4, 3.4, 7),
                   np.linspace(4.1, 6.7, 7)])
cor_t2 = np.array([np.linspace(-3.3, -3.3, 8),
                   np.linspace(2.4, 2.4, 8),
                   np.linspace(4.1, 6.7, 8)]) 
cor_t1 = np.delete(cor_t1,[2],axis=1)
cor_t2 = np.delete(cor_t2,[2,5],axis=1)
cor_t = np.concatenate((cor_t1, cor_t2), axis=1)
cor_b = np.delete(cor_b,[4,-2],axis=1)
#%% 
loadir = './sov/sov06/'
b1_name ='01_2100v2.smr'#2_BCX2000.smr'
t1_name= '01_2100v2.smr'
sig_B1,Fsb,t_stop, m1a = ldfs(loadir+b1_name, 32,2)
sig_T1,Fst,t_stop, m2a = ldfs(loadir+t1_name, 16)
Fs = int(Fsb)
#%%
back, forward=50, 250
trials_b = np.zeros((14,100,back+forward))
trials_t = np.zeros((16,100,back+forward))
shf=5
for i in range(100):
    t1a=m1a[i]
    t2a=m1a[i]
    trials_b[:,i] = sig_B1[17:31,int(t1a*Fs-back+shf):int(t1a*Fs+forward+shf)]
    trials_t[:,i] = sig_T1[:16,int(t2a*Fs-back+shf):int(t2a*Fs+forward+shf)]
trials_b = np.delete(trials_b, [4,-3],axis=0)
trials_t = np.delete(trials_t, [2,7,10,13],axis=0)
# cor_b = np.delete(cor_b,[4],axis=1)
# cor_t = np.delete(cor_t,[4],axis=1)
#%%
ele_pos = np.concatenate((cor_b,cor_t), axis=1)
lfp_s = np.concatenate((trials_b.mean(axis=1),trials_t.mean(axis=1)))
frags = 30
leng= 50
leap=10
ch_stay=4
cor_score = np.zeros((ele_pos.shape[1],frags))
cor_score_crtx, cor_score_th = np.zeros((ele_pos.shape[1],frags)), np.zeros((ele_pos.shape[1],frags))
for part in range(frags-3):
    for ch in range(ele_pos.shape[1]):
        cor_score[ch,part] = pearsonr(lfp_s[ch_stay,leap*part:leap*part+leng], lfp_s[ch,leap*part:leap*part+leng])[0]
py.figure(figsize=(8,8))
py.subplot(211)
py.title('LFP')
py.imshow(lfp_s, vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],0], aspect='auto', cmap='PRGn')
py.ylabel('channels'), py.xlabel('time')
py.subplot(212)
py.title('Correlation')
py.imshow(cor_score, vmin=-1, vmax=1, extent=[-5,25,ele_pos.shape[1],0],aspect='auto', cmap='PiYG')
py.ylabel('channels'), py.xlabel('time')
py.savefig('sov'+loadir[-3:-1]+'_load')
scipy.io.savemat('./mats/sov' +loadir[-3:-1]+'.mat', 
                 dict(crtx=trials_b.mean(axis=1), th=trials_t.mean(axis=1),
                      cor=cor_score, ele_pos=ele_pos))
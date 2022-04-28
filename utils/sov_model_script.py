# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:12:06 2020

@author: Wladek
"""

import numpy as np #wczytanie biblioteki z funkcjami matematycznymi
import pylab as py #wczytanie biblioteki do rysowania
import os #wczytanie biblioteki do dostepu do wybranych folderow
from scipy.signal import filtfilt, butter, coherence, welch
from kcsd import oKCSD3D, KCSD1D
from skimage import filters
from mayavi import mlab
# from neo.io import Spike2IO
from basicfunc import ldfs, remove_points
import matplotlib.gridspec as gridspec

def model_data(ele_pos,charges, dip_cor):
    wave = np.zeros((ele_pos.shape[1],1))
    for i in range(ele_pos.shape[1]):
        for n,q in enumerate(charges):
            wave[i] += q/np.sum((ele_pos[:,i]-dip_cor[:,n])**2)**(1/2)
    return wave, dip_cor
    
def check_3d(ele_pos, est_xyz, dis, save=True):
    mlab.figure(bgcolor=(0.2, 0.2, 0.2), size=(1000, 800))
    mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color = (0, 1, 0), 
                  scale_mode='none', scale_factor=.2)
    # mlab.points3d(0, 0, 1, color = (1, 1, 0.5), scale_mode='none', scale_factor=1)
    mlab.points3d(env_x, env_y, env_z, scale_mode='none', scale_factor=0.03, color = (1, 1, 0.5))
    pt = mlab.points3d(dip_cor[0], dip_cor[1], dip_cor[2], charges, colormap = 'RdBu', 
                  vmin = -1, vmax = 1, scale_mode='none', scale_factor=1)
    # mlab.points3d(est_xyz[0],  est_xyz[1],  est_xyz[2], scale_mode='none', scale_factor=1)
    # mlab.points3d(x_hip, y_hip, z_hip, scale_mode='none', scale_factor=0.1,color = (.5, 0, .5)
    pt.module_manager.scalar_lut_manager.reverse_lut = True
    mlab.view(elevation = 70, azimuth=50, distance = 80-dis)
    
#%% 
# loadir = './sov/sov1/'
# b1_name ='01_1700_all.smr'#2_BCX2000.smr'
# b2_name ='02_2500_all.smr'#2_BCX2000.smr'
# t1_name= '06_5600_all.smr'
# t2_name= '09_6250_all.smr'

# # loadir = './sov/sov2/'
# # b1_name ='01_1500_all.smr'#2_BCX2000.smr'
# # b2_name ='02_2300_all.smr'#2_BCX2000.smr'
# # t1_name= '06_5500_all.smr'
# # t2_name= '07_5750_all.smr'

# # r = Spike2IO(filename=loadir+b_name)
# # seg = r.read_segment()

# sig_B1,Fsb,ch_list_b,t_stop, m1a = ldfs(loadir+b1_name, 16)
# sig_B2,Fsb,ch_list_b,t_stop, m1b = ldfs(loadir+b2_name, 16)
# sig_T1,Fst,ch_list_t,t_stop, m2a = ldfs(loadir+t1_name, 32)
# sig_T2,Fst,ch_list_t,t_stop, m2b = ldfs(loadir+t2_name, 32)
# Fs = int(Fsb)
# # timeline=np.linspace(0, sig_B.shape[1]/Fs,sig_B.shape[1] )
# # b,a = butter(3, [500/(Fs/2), 2000/(Fs/2)], btype='bandpass')
# # sig_B = filtfilt(b,a, sig_B)
# # sig_T = filtfilt(b,a, sig_T)
# #%%
# crtx_ch = 24
# trials_b = np.zeros((crtx_ch,100,Fs))
# trials_t = np.zeros((16,100,Fs))
# for i in range(100):
#     t2a=float(m2a[1][i])
#     t2b=float(m2b[1][i])
#     t1a=float(m1a[1][i])
#     t1b=float(m1b[1][i])
#     trials_b[:16,i] = sig_B1[:,int(t1a*Fs-Fs/4):int(t1a*Fs+Fs*3/4)][:,:Fs]
#     trials_b[16:,i] = sig_B2[8:,int(t1b*Fs-Fs/4):int(t1b*Fs+Fs*3/4)][:,:Fs]
#     trials_t[:8,i] = sig_T1[:8,int(t2a*Fs-Fs/4):int(t2a*Fs+Fs*3/4)][:,:Fs]
#     trials_t[8:,i] = sig_T2[8:16,int(t2b*Fs-Fs/4):int(t2b*Fs+Fs*3/4)][:,:Fs]

# py.figure()
# py.subplot(121)
# py.suptitle(b1_name)
# py.title('BCX')
# time = np.linspace(-250,750,Fs)
# for i in range(crtx_ch):
#     py.plot(time, trials_b[i].mean(axis=0)-i*1, color='blue')
# py.xlim(-10, 40)
# py.grid()
# py.subplot(122)
# py.title('Thalamus')
# for i in range(16):
#     py.plot(time, trials_t[i].mean(axis=0)-i*.2, color='blue')
# py.xlim(-10, 40)
# py.grid()
crtx_ch=24
cor_b = np.array([np.linspace(6,5,crtx_ch),
                  np.linspace(-3,-3.5,crtx_ch),
                  np.linspace(.5, -1.5, crtx_ch)])

cor_t = np.array([np.linspace(3.5,2.5,16),
                  np.linspace(-3,-3.5,16),
                  np.linspace(-4.9, -6.5, 16)])

ele_pos = np.concatenate((cor_b,cor_t),axis=1)

modeldir = '/Users/Wladek/Dysk Google/Tom_data/'
skacz = 4
brain_data = np.load(modeldir+'volume_Brain.npy')[::skacz,::skacz, ::skacz]
brain_data = np.rot90(brain_data, 1, axes=(1,2))
div = 1/(0.05*skacz) #35
brain_c = np.zeros((70,brain_data.shape[1],brain_data.shape[2]))
for i in range(brain_c.shape[0]):
    brain_c[i] =  filters.sobel(brain_data[i])
px2 , pyy2, pz2 = np.where(brain_c>0.6)
xtran,ytran,ztran = 8.1,23,12
env_x = pyy2/div - xtran  
env_y = pz2/div - ytran
env_z = px2/div - ztran
np.save('est_env', np.array([env_x,env_y,env_z]))
px, pyy, pz = np.where(brain_data[:, 45:, 100:120]>100)
skip = 2
xpos = pyy[::skip]/div + 2#4
ypos = pz[::skip]/div - 5
zpos = px[::skip]/div - 12#2.5
est_xyz = np.array([xpos,ypos,zpos])
# px_hip, py_hip, pz_hip = np.where(brain_data[40:, 45:70, 100:120]<240)
# skip = 4
# x_hip = px_hip[::skip]/div + 2
# y_hip= py_hip[::skip]/div - 5
# z_hip= pz_hip[::skip]/div - ztran
# py.figure()
# py.imshow(brain_data[::-1,:,220])
# py.colorbar()

est_xyz=np.array([[6,6,6],[-3,-3,-3], [-.5, -1., -1.5]])
charges = [.5, -1, .5]
wave, dip_cor = model_data(ele_pos,charges, est_xyz)
check_3d(ele_pos, est_xyz, dis=0, save = False)
#%%
# py.close('all')
# wave = np.concatenate((trials_b.mean(axis=1), trials_t.mean(axis=1)))[:,::4]
k = oKCSD3D(ele_pos.T, wave, own_src=est_xyz, sigma=1)
k.L_curve(lambdas=np.logspace(-11,-6,20), Rs=np.linspace(.1,.2,10))
est_csd = k.values('CSD')
py.figure()
py.imshow(k.curve_surf, aspect='auto',vmax=np.max(abs(k.curve_surf)),vmin=-np.max(abs(k.curve_surf)), cmap='bwr')
py.colorbar()
print(est_csd)
#%%
visname = 'anim'#snap, time, anim
tp = 0
roz = 1
# roz = .5
stds = 0
# colors = ['blue', 'navy', 'indianred', 'maroon', 'green', 'darkgreen']
@mlab.animate(delay=100)
def anim():
    tp = 0
    while True:
        est_csdi, xposii, yposii, zposii = remove_points(est_csd, est_xyz[0], est_xyz[1], est_xyz[2], tp+600, stds*np.std(est_csd[:,tp+600]))
        pt.mlab_source.reset(x=xposii, y=yposii, z=zposii, scalars=est_csdi)
        pt.module_manager.scalar_lut_manager.reverse_lut = True
        # lines.mlab_source.reset(y=[tp/100], z=[wave_20[tp]-5])
        tp+=1
        if tp==100: tp=0
        yield
        
mfig=mlab.figure(bgcolor=(0.2, 0.2, 0.2), fgcolor=(0.,0.,0.),size=(1000, 800))
mlab.points3d(env_x[::4], env_y[::4], env_z[::4], scale_mode='none', scale_factor=0.03, color = (1, 1, 0.5))
mlab.points3d(ele_pos[0], ele_pos[1], ele_pos[2], color = (0, 1, 0), scale_mode='none', scale_factor=.2) 
# mlab.points3d(dip_cor[0], dip_cor[1], dip_cor[2], charges, colormap = 'RdBu', 
              # vmin = -1, vmax = 1, scale_mode='none', scale_factor=1)
est_csdi, xposii, yposii, zposii = remove_points(est_csd, est_xyz[0], est_xyz[1], est_xyz[2], 0, stds*np.std(est_csd[:,0]))
pt = mlab.points3d(xposii, yposii, zposii, est_csdi, colormap = 'RdBu', 
                    vmin = -roz, vmax = roz, scale_mode='none', scale_factor=.8)
pt.module_manager.scalar_lut_manager.reverse_lut = True
mlab.colorbar()
# a = anim()
#%%
py.figure()
est_wave = np.dot(k.k_pot, k.beta_new)
py.suptitle('Lambda: '+ str(k.lambd)+' R: '+str(k.R))
py.subplot(121)
py.title('BCX')
py.plot(wave[:24,0], color='black')
py.plot(est_wave[:24,0], color='blue')
# time = np.linspace(-250,750,2500)
# for i in range(crtx_ch): 
#     py.plot(time,est_wave[i]-1.2*i, color='black')
#     py.plot(time,wave[i]-1.2*i, color='blue')
# py.xlim(-10,40)
py.subplot(122)
py.title('Thl')
py.plot(wave[24:,0], color='black')
py.plot(est_wave[24:,0], color='blue')
# for i in range(16): 
     # py.plot(time,est_wave[crtx_ch+i]-.1*i, color='black')
     # py.plot(time,wave[crtx_ch+i]-.1*i, color='blue')
# py.xlim(-10,40)
import matplotlib.patches as mpatches
est = mpatches.Patch(color='blue', label='Estimated potential from CSD')
pot = mpatches.Patch(color='black', label='Measured potential')
py.legend(loc=1, handles=[est, pot],
          ncol=1, frameon = True, fontsize = 15)
# elif visname == 'time':
#     py.close('all')
#     for tp in range(600, 900, 2):
#         if tp<10:
#             save_as = 'proj_000' + str(tp)
#         elif tp<100:
#             save_as = 'proj_00' + str(tp)
#         elif tp<1000:
#             save_as = 'proj_0' + str(tp)
#         else:
#             save_as = 'proj_' + str(tp)
#         plot3dbrain_m(tp, save = True)

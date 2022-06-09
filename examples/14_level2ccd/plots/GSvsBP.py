#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 08:36:58 2022

@author: athulsun
"""
import pickle 
import matplotlib.pyplot as plt
import numpy as np

pklname = 'q8_review.pkl'

with open(pklname,'rb') as handle:
    Results = pickle.load(handle)

transition_results = Results[1]
OutputNames = transition_results['Outputnames']
indpwr = OutputNames.index('SrvD GenPwr, (kW)')

X = transition_results['States']
U = transition_results['Controls']
Y = transition_results['Outputs']
Time = transition_results['Time']

iGenSpeed = 4; iPtfmPitch = 0; iBldPitch = 2;


specific_results = np.zeros((1000,2))

idx = 0

specific_results[:,idx] = U[:,iBldPitch].squeeze()
idx +=1

specific_results[:,idx] = X[:,iGenSpeed].squeeze()
idx +=1

#specific_results[:,idx] = Y[:,indpwr].squeeze()
# idx +=1

#specific_results[:,idx] = U[:,0].squeeze()

mean = np.mean(specific_results,0)
scaled_results = specific_results/mean



fig1,ax1 = plt.subplots(1,1)
ax1.set_ylabel('Scaled Quantities',fontsize = 16)
ax1.set_xlabel('Time [s]',fontsize = 16)
ax1.set_xlim([0,800])
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

labels = ['Blade Pitch','GenSpeed','GenPower','Wind Speed','GenPwr']
col = ['k','r']

for i in range(2):
    ax1.plot(Time,scaled_results[:,i],color = col[i],linewidth = 2,markersize = 10,label = labels[i])

fig1.legend(bbox_to_anchor=(0.5,1), loc='upper center',ncol = 4,fontsize = 12)

# fig2,ax2 = plt.subplots(1,1)
# ax2.set_ylabel('Wind Speed [m/s]',fontsize = 16)
# ax2.set_xlabel('Time [s]',fontsize = 16)
# ax2.set_xlim([0,800])
# ax2.tick_params(axis='x', labelsize=12)
# ax2.tick_params(axis='y', labelsize=12)

# ax2.plot(Time,U[:,0],linewidth = 2,markersize = 10)

#breakpoint()
fig1.savefig('GSvsBP.svg',format = 'svg',dpi = 1200)
#fig2.savefig('WS.svg',format = 'svg',dpi = 1200)


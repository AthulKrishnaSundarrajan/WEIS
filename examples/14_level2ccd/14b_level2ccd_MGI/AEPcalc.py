#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:54:39 2022

@author: athulsun
"""

import numpy as np 
from pCrunch import PowerProduction
import matplotlib.pyplot as plt

turbine_class = 'I'

pp = PowerProduction(turbine_class)

ws =  [4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24.]

ws_prob = pp.prob_WindDist(ws)

# initialize plot
fig1,ax1 = plt.subplots(1,1)
ax1.set_xlabel('Wind Speed [m/s]',fontsize = 16)
ax1.set_ylabel('Weights',fontsize = 16)

ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

#ax1.plot(ws,ws_prob)


wind_prob = pp.prob_WindDist(ws)
nw = len(ws)
val = np.zeros(nw)

I = np.eye(nw)

for i in range(nw):
    val[i] = np.trapz(I[i,:]*wind_prob,ws)


ax1.plot(ws,val,'*')
# ws = [5.8,9.4732,13.009,15.6765,16.866,19.9968]
# #ws = [5,7,9,10,11,13,15,17,19,21,24]
# pwr = [2976.640245,10026.456721,14443.843565,15004.501395,12546.785709,15036.040867]

# wind_prob = pp.prob_WindDist(ws)
# nw = len(ws)
# val = np.zeros(nw)

# I = np.eye(nw)

# for i in range(nw):
#     val[i] = np.trapz(I[i,:]*wind_prob,ws)


# ax1.plot(ws,val,'*')

# P = np.sum(val)
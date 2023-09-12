import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pCrunch.io import load_FAST_out
from ROSCO_toolbox.ofTools.fast_io import output_processing
from ROSCO_toolbox.ofTools.util import spectral


def plot_signal(signal_dict,control_flag):
    
    time = signal_dict['time']
    signals_act = signal_dict['OpenFAST']
    signals_dfsm = signal_dict['DFSM']
    n_signals = signal_dict['n']
    signal_names = signal_dict['names']
    
    dx_flag = [not(name[0] == 'd') for name in signal_names]
    
    t0 = time[0];tf = time[-1]
    
    # plot controls
    fig,ax = plt.subplots(sum(dx_flag),1)
    
    if sum(dx_flag) == 1:
        ax = [ax]
        
    ax[-1].set_xlabel('Time [s]')
    fig.subplots_adjust(hspace = 0.65)
    
    for idx,qty in enumerate(signal_names):
        
        if not(qty[0] == 'd'):
            ax[idx].plot(time,signals_act[:,idx],label = 'OpenFAST')
        
            if not(control_flag):
                ax[idx].plot(time,signals_dfsm[:,idx],label = 'DFSM')
            
            ax[idx].set_title(qty)
            ax[idx].set_xlim([t0,tf])
        
    if not(control_flag):
        ax[0].legend(ncol = 2)
    
    

def plot_dfsm_results(U_list,X_list,dx_list,Y_list,simulation_flag,outputs_flag):
    
    n_results = len(U_list)
    
    for ix in range(n_results):
        
        # plot controls
        u_dict = U_list[ix]
        
       # plot_signal(u_dict,True)
    
        dx_dict = dx_list[ix]
        #plot_signal(dx_dict,False)
        
        if simulation_flag:
            x_dict = X_list[ix]
            plot_signal(x_dict,False)
        
        if outputs_flag:
            y_dict = Y_list[ix]
            plot_signal(y_dict,False)
            
            
        

def plot_inputs(SimulationDetails,index,plot_type,save_flag = False,save_path = 'plots'):
    
    # extract 
    sim_details = SimulationDetails.FAST_sim[index]
    
    time = sim_details['time']
    controls = sim_details['controls']
    states = sim_details['states']
    outputs = sim_details['outputs']
    
    control_names = sim_details['control_names']
    state_names = sim_details['state_names']
    output_names = sim_details['output_names']
    
    n_controls = sim_details['n_controls']
    n_outputs = sim_details['n_outputs']
    n_states = sim_details['n_states']
    
    
    t0 = time[0]; tf = time[-1]
    print(state_names)
    state_flag = [not(name[0] == 'd') for name in state_names]
    n_states_ = sum(state_flag)
    states_ = states[:,state_flag]
    state_names_ = []
    
    for idx,flag in enumerate(state_flag):
        if flag:
            state_names_.append(state_names[idx])
    
    # depending on plot type, plot the time series quantities
    if plot_type == 'vertical':
         
        if not(len(outputs) > 0):
            
            # combine all signals into a single array
            quantities = np.hstack([controls,states_])
            
            # get the names
            quantity_names = control_names + state_names_
     
            
        else:
            
            # combine all signals into a single array
            quantities = np.hstack([controls,states_,outputs])
            
            # names of the quantities
            quantity_names = control_names + state_names_ + output_names
            
        n_qty = len(quantity_names)
        
        # intialize plot
        fig,ax = plt.subplots(n_qty,1)
        
        ax[-1].set_xlabel('Time [s]')
        
        fig.subplots_adjust(hspace = 1)
        
        for idx,qty in enumerate(quantity_names):
            
            if not(qty[0] == 'd'):
                ax[idx].plot(time,quantities[:,idx])
                ax[idx].set_title(qty)
                ax[idx].set_xlim([t0,tf])
                
            if not(idx == n_qty-1):
                ax[idx].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        
        wd = os. getcwd()
        
        plot_path = wd + os.sep + save_path
        
        if save_flag:
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)
                
            fig.savefig(plot_path +os.sep+ 'inputs.svg')
            
    elif plot_type == 'separate':
        
        # plot controls
        fig,axc = plt.subplots(n_controls,1)
        if n_controls == 1:
            axc = [axc]
        axc[-1].set_xlabel('Time [s]')
        fig.subplots_adjust(hspace = 0.65)
        
        for idx,qty in enumerate(control_names):
            
            axc[idx].plot(time,controls[:,idx])
            axc[idx].set_title(qty)
            axc[idx].set_xlim([t0,tf])
            
        # plot states    
        fig,axs = plt.subplots(n_states,1)
        if n_states == 1:
            axs = [axs]
        fig.subplots_adjust(hspace = 1)
        axc[-1].set_xlabel('Time [s]')
        
        for idx,qty in enumerate(state_names):
            
            axs[idx].plot(time,states[:,idx])
            axs[idx].set_title(qty)
            axs[idx].set_xlim([t0,tf])
        
        # plot outputs
        if n_outputs > 0:
            
            fig,axo = plt.subplots(n_outputs,1)
            axc[-1].set_xlabel('Time [s]')
            fig.subplots_adjust(hspace = 0.65)
        
            for idx,qty in enumerate(output_names):
                
                axo[idx].plot(time,outputs[:,idx])
                axo[idx].set_title(qty)
                axo[idx].set_xlim([t0,tf])
                
        
            
        
        
        
        
        
            
            
                
        
    
    